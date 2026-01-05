import os
import random
import re
from typing import Dict, List, Optional, Sequence, Tuple

import torch

# Reuse small helper when available
try:
    from .training_utils import to_device
except Exception:
    def to_device(data, device):
        if isinstance(data, dict):
            return {k: v.to(device) if hasattr(v, "to") else v for k, v in data.items()}
        elif hasattr(data, "to"):
            return data.to(device)
        return data


class ReplayDataLoader:
    """Stream filtered batches from a previous GradDotProd run, optionally reshuffling samples."""

    def __init__(
        self,
        run_dir: str,
        filter_metric: str = "dot_product",
        threshold: float = 0.0,
        rebatch_size: Optional[int] = None,
        drop_last: bool = False,
        shuffle: bool = False,
        shuffle_seed: Optional[int] = None,
        rank: int = 0,
        world_size: int = 1,
        device: str = "cuda",
    ) -> None:
        self.run_dir = os.path.abspath(run_dir)
        self.filter_metric = filter_metric
        self.threshold = threshold
        self.rebatch_size = rebatch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.shuffle_seed = shuffle_seed
        self.rank = rank
        self.world_size = max(1, world_size)
        self.device = device

        self.grad_dir = self._resolve_grad_dir()
        self.valset_path = os.path.join(self.grad_dir, "valset.pt")
        self.valset_cache = None

        self.files = self._sorted_log_files()
        if not self.files:
            raise FileNotFoundError(f"No dot product logs found under {self.grad_dir}")

        # Streaming state
        self._file_idx = 0
        self._entry_idx = 0
        self._sample_buffer: List[Dict] = []
        self._global_kept = 0  # counts kept samples across ranks for striding
        self._exhausted = False
        self._shuffled_samples: Optional[List[Dict]] = None
        self._shuffle_pos = 0
        self._samples_emitted = 0

        if self.shuffle:
            self._build_shuffled_samples()

    def _build_shuffled_samples(self) -> None:
        """Load and shuffle all filtered samples into memory."""
        print("[INFO] Loading replay samples for shuffling; this may use significant RAM.")
        device = torch.device("cpu")
        samples: List[Dict] = []
        for log_path in self.files:
            log = torch.load(log_path, map_location="cpu")
            for entry in log:
                metric = self._compute_metric(entry, device)
                X_train = entry["X_train"]
                Y_train = entry["Y_train"]
                batch_idx = entry.get("batch_idx")
                iter_num = entry.get("iter_num")
                for i in range(metric.shape[0]):
                    if metric[i].item() < self.threshold:
                        continue
                    samples.append(
                        {
                            "X": X_train[i],
                            "Y": Y_train[i],
                            "metric": metric[i],
                            "batch_idx": None if batch_idx is None else batch_idx[i],
                            "iter_num": iter_num,
                        }
                    )

        self._global_kept = len(samples)
        rng = random.Random(self.shuffle_seed)
        rng.shuffle(samples)

        if self.world_size > 1:
            samples = samples[self.rank :: self.world_size]

        self._shuffled_samples = samples
        self._shuffle_pos = 0
        self._samples_emitted = 0
        self._file_idx = len(self.files)
        self._entry_idx = 0
        if not samples:
            self._exhausted = True

    def _resolve_grad_dir(self) -> str:
        """Locate the grad_dotprods directory given a run dir."""
        direct = os.path.join(self.run_dir, "grad_dotprods")
        if os.path.isdir(direct):
            return direct

        # Try one level down (e.g., timestamp -> run folder -> grad_dotprods)
        subdirs = [
            os.path.join(self.run_dir, d)
            for d in os.listdir(self.run_dir)
            if os.path.isdir(os.path.join(self.run_dir, d))
        ]
        for subdir in subdirs:
            candidate = os.path.join(subdir, "grad_dotprods")
            if os.path.isdir(candidate):
                return candidate

        raise FileNotFoundError(
            f"Could not locate grad_dotprods directory inside {self.run_dir}"
        )

    def _sorted_log_files(self) -> List[str]:
        """List dot product logs ordered by iteration."""
        pattern = re.compile(r"dot_prod_log_iter_(-?\d+)\.pt")
        files = []
        for fname in os.listdir(self.grad_dir):
            match = pattern.match(fname)
            if match:
                it = int(match.group(1))
                files.append((it, os.path.join(self.grad_dir, fname)))
        files.sort(key=lambda x: x[0])
        return [p for _, p in files]

    def get_validation_batch(self, batch_size: Optional[int] = None):
        """Return the saved validation batch if present."""
        if self.valset_cache is None:
            if not os.path.isfile(self.valset_path):
                raise FileNotFoundError(
                    f"Validation set not found at {self.valset_path}"
                )
            self.valset_cache = torch.load(self.valset_path)

        X_val, Y_val = self.valset_cache["X_val"], self.valset_cache["Y_val"]
        if batch_size is not None and X_val.shape[0] != batch_size:
            raise ValueError(
                f"Requested val batch size {batch_size} but saved valset has {X_val.shape[0]}"
            )
        return to_device(X_val, self.device), to_device(Y_val, self.device)

    def _load_next_entry(self) -> Optional[Dict]:
        """Load the next log entry (may advance files)."""
        while self._file_idx < len(self.files):
            log_path = self.files[self._file_idx]
            log = torch.load(log_path)
            if self._entry_idx < len(log):
                entry = log[self._entry_idx]
                self._entry_idx += 1
                return entry

            # Move to next file
            self._file_idx += 1
            self._entry_idx = 0
        self._exhausted = True
        return None

    def _compute_metric(
        self, entry: Dict, device: torch.device
    ) -> torch.Tensor:
        dot_product = entry["dot_product"].to(device)
        if self.filter_metric == "dot_product":
            return dot_product
        if self.filter_metric == "cosine":
            if "train_grad_norm" not in entry or "val_grad_norm" not in entry:
                raise ValueError(
                    "Cosine metric requested but train/val gradient norms are missing."
                )
            train_norm = entry["train_grad_norm"].to(device)
            val_norm = torch.tensor(entry["val_grad_norm"], device=device)
            denom = train_norm * val_norm
            eps = torch.finfo(denom.dtype).eps
            return dot_product / torch.clamp(denom, min=eps)
        raise ValueError(f"Unknown filter metric: {self.filter_metric}")

    def _fill_buffer(self, min_fill: int) -> None:
        """Fill local buffer with filtered samples that belong to this rank."""
        if self._exhausted:
            return

        device = torch.device("cpu")

        target = max(1, min_fill)
        while not self._exhausted and len(self._sample_buffer) < target:
            entry = self._load_next_entry()
            if entry is None:
                break

            metric = self._compute_metric(entry, device)
            X_train = entry["X_train"]
            Y_train = entry["Y_train"]
            batch_idx = entry.get("batch_idx")
            iter_num = entry.get("iter_num")

            # Preserve per-sample order inside the batch
            for i in range(metric.shape[0]):
                if metric[i].item() < self.threshold:
                    continue
                if (self._global_kept % self.world_size) != self.rank:
                    self._global_kept += 1
                    continue
                sample = {
                    "X": X_train[i],
                    "Y": Y_train[i],
                    "metric": metric[i],
                    "batch_idx": None if batch_idx is None else batch_idx[i],
                    "iter_num": iter_num,
                }
                self._sample_buffer.append(sample)
                self._global_kept += 1

    def next_batch(
        self, batch_size: Optional[int] = None, return_idx: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Return the next filtered batch; raises StopIteration when exhausted."""
        if self.shuffle:
            return self._next_batch_from_shuffle(batch_size=batch_size, return_idx=return_idx)
        target_size = batch_size or self.rebatch_size
        if target_size is None:
            raise ValueError("Batch size must be provided for replay loader.")

        while len(self._sample_buffer) < target_size and not self._exhausted:
            self._fill_buffer(target_size)

        if len(self._sample_buffer) < target_size:
            if not self.drop_last and self._sample_buffer:
                target_size = len(self._sample_buffer)
            else:
                raise StopIteration("Replay data exhausted.")

        take, self._sample_buffer = (
            self._sample_buffer[:target_size],
            self._sample_buffer[target_size:],
        )

        first_x = take[0]["X"]
        if isinstance(first_x, dict):
            X = {
                k: torch.stack([s["X"][k] for s in take]).to(self.device)
                for k in first_x
            }
        else:
            X = torch.stack([s["X"] for s in take]).to(self.device)

        Y = torch.stack([s["Y"] for s in take]).to(self.device)

        idx_tensor = None
        if return_idx:
            idx_values: List[int] = []
            for s in take:
                idx_val = s["batch_idx"]
                idx_values.append(int(idx_val) if idx_val is not None else -1)
            idx_tensor = torch.tensor(idx_values, device=self.device)

        return X, Y, idx_tensor

    def _next_batch_from_shuffle(
        self, batch_size: Optional[int] = None, return_idx: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Return the next batch from the shuffled replay pool."""
        target_size = batch_size or self.rebatch_size
        if target_size is None:
            raise ValueError("Batch size must be provided for replay loader.")

        if self._shuffled_samples is None:
            self._build_shuffled_samples()

        remaining = len(self._shuffled_samples) - self._shuffle_pos
        if remaining <= 0:
            raise StopIteration("Replay data exhausted.")

        if remaining < target_size:
            if not self.drop_last:
                target_size = remaining
            else:
                raise StopIteration("Replay data exhausted.")

        take = self._shuffled_samples[self._shuffle_pos : self._shuffle_pos + target_size]
        self._shuffle_pos += target_size
        self._samples_emitted += target_size

        first_x = take[0]["X"]
        if isinstance(first_x, dict):
            X = {
                k: torch.stack([s["X"][k] for s in take]).to(self.device)
                for k in first_x
            }
        else:
            X = torch.stack([s["X"] for s in take]).to(self.device)

        Y = torch.stack([s["Y"] for s in take]).to(self.device)

        idx_tensor = None
        if return_idx:
            idx_values: List[int] = []
            for s in take:
                idx_val = s["batch_idx"]
                idx_values.append(int(idx_val) if idx_val is not None else -1)
            idx_tensor = torch.tensor(idx_values, device=self.device)

        return X, Y, idx_tensor

    def stats(self) -> Dict[str, int]:
        """Return simple stats useful for logging."""
        if self.shuffle and self._shuffled_samples is not None:
            return {
                "files_total": len(self.files),
                "files_consumed": len(self.files),
                "buffer_size": max(0, len(self._shuffled_samples) - self._shuffle_pos),
                "samples_emitted": self._samples_emitted,
            }
        return {
            "files_total": len(self.files),
            "files_consumed": self._file_idx + (1 if self._entry_idx else 0),
            "buffer_size": len(self._sample_buffer),
            "samples_emitted": self._global_kept // self.world_size,
        }
