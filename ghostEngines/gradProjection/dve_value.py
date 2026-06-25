"""
Data Value Embedding — value computation and corpus attribution.

Stage 3: dot projected **test** gradients against the stored per-training-point
embeddings (``embed_iter_*.pt`` from :mod:`dve_embedding`) to form a dense
``(n_test x n_train)`` value matrix:

    value(test_j, train_s) = <g_test_j, e_s>

Both operands live in the *same* projected space (same projection ``P`` reconstructed
from the shared ``proj_seed``), so the dot product approximates the exact in-run data
value. The sign convention follows the reference (positive = the training gradient is
aligned with the test gradient under the unrolled-SGD transform).

Stage 4: rank training points by value for selected test points (corpus attribution).
"""

import json
from pathlib import Path
from typing import List, Optional

import torch


def _list_embed_files(embed_dir: Path) -> List[Path]:
    files = sorted(embed_dir.glob('embed_iter_*.pt'))
    if not files:
        raise FileNotFoundError(f"No embed_iter_*.pt files in {embed_dir}")
    return files  # ascending == training order -> columns in training order


def compute_values(embed_dir, test_proj: torch.Tensor,
                   test_ids: Optional[List[int]] = None,
                   save_path=None, device: str = 'cpu') -> dict:
    """Compute the ``(n_test x n_train)`` value matrix.

    Args:
        embed_dir: directory of ``embed_iter_*.pt`` files.
        test_proj: projected test gradients, shape ``[n_test, total_proj_dim]``, on the
            same scale/projection as the captured train gradients.
        test_ids: optional identities for the test rows (window offsets etc.).
        save_path: optional path to ``torch.save`` the result dict.
        device: device for the matmuls.

    Returns dict with ``values`` ``[n_test, n_train]`` (cpu float32), ``train_ids``
    (per-column training-window identity), ``train_order`` (per-column training step),
    and ``test_ids``.
    """
    embed_dir = Path(embed_dir)
    files = _list_embed_files(embed_dir)

    test_proj = test_proj.to(device=device, dtype=torch.float32)
    n_test = test_proj.shape[0]

    value_cols: List[torch.Tensor] = []
    train_ids: List[int] = []
    train_order: List[int] = []

    for f in files:
        d = torch.load(f, map_location=device)
        emb = d['embedding'].to(device=device, dtype=torch.float32)  # [B, total]
        if emb.shape[1] != test_proj.shape[1]:
            raise ValueError(
                f"Projection-dim mismatch: test {test_proj.shape[1]} vs "
                f"embedding {emb.shape[1]} in {f.name}. Same proj_seed/arch required.")

        block = test_proj @ emb.t()  # [n_test, B]
        value_cols.append(block.cpu())

        B = emb.shape[0]
        bidx = d.get('batch_idx', list(range(B)))
        train_ids.extend(int(x) for x in bidx)
        step = int(d.get('order', d.get('iter', 0)))
        train_order.extend([step] * B)

    values = torch.cat(value_cols, dim=1)  # [n_test, n_train]

    result = {
        'values': values,
        'train_ids': train_ids,
        'train_order': train_order,
        'test_ids': test_ids if test_ids is not None else list(range(n_test)),
    }
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(result, save_path)
        print(f"[value] saved value matrix {tuple(values.shape)} to {save_path}")
    return result


def attribute(values_result: dict, top_k: int = 10,
              test_indices: Optional[List[int]] = None) -> dict:
    """Rank training points by value for selected test rows.

    Returns a dict mapping each chosen test row index to ``{'top': [...], 'bottom': [...]}``
    where each entry is ``(train_column_index, train_id, train_order, value)``. Also
    prints a readable summary.
    """
    values = values_result['values']
    train_ids = values_result['train_ids']
    train_order = values_result['train_order']
    test_ids = values_result['test_ids']

    n_test, n_train = values.shape
    if test_indices is None:
        test_indices = list(range(min(n_test, 3)))
    k = min(top_k, n_train)

    out = {}
    for ti in test_indices:
        row = values[ti]
        top = torch.topk(row, k)
        bot = torch.topk(row, k, largest=False)

        def pack(idxs, vals):
            return [(int(j), int(train_ids[j]), int(train_order[j]), float(v))
                    for j, v in zip(idxs.tolist(), vals.tolist())]

        rec = {'top': pack(top.indices, top.values),
               'bottom': pack(bot.indices, bot.values)}
        out[int(ti)] = rec

        print(f"\n=== Test point row {ti} (id={test_ids[ti]}) ===")
        print(f"  Top {k} most valuable training points (col, train_id, step, value):")
        for c, tid, st, v in rec['top']:
            print(f"    col={c:6d}  train_id={tid:>10}  step={st:>5}  value={v:+.6e}")
        print(f"  Bottom {k} (most harmful):")
        for c, tid, st, v in rec['bottom']:
            print(f"    col={c:6d}  train_id={tid:>10}  step={st:>5}  value={v:+.6e}")
    return out


def load_and_attribute(values_path, top_k: int = 10,
                       test_indices: Optional[List[int]] = None) -> dict:
    """Load a saved ``values.pt`` and run :func:`attribute`."""
    result = torch.load(values_path, map_location='cpu')
    return attribute(result, top_k=top_k, test_indices=test_indices)
