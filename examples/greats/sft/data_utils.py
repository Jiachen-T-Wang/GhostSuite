"""Data utilities for the GREATS SFT example.

Ports the LESS instruction-tuning data pipeline (tulu chat format) and the MMLU
validation-target construction, so the example mirrors the upstream
`warmup_train.sh` setup. See docs/plans/greats_sft_phaseB_2026-06-25.md.
"""

import json
import os
import random
from typing import Dict, List, Optional

import pandas as pd
import torch


# --------------------------------------------------------------------------- #
# Instruction (training) data — tulu chat format
# --------------------------------------------------------------------------- #
def concat_messages(messages: List[dict], tokenizer) -> str:
    """Tulu chat formatting (LESS `concat_messages`)."""
    text = ""
    for m in messages:
        role, content = m["role"], m["content"].strip()
        if role == "system":
            text += "<|system|>\n" + content + "\n"
        elif role == "user":
            text += "<|user|>\n" + content + "\n"
        elif role == "assistant":
            text += "<|assistant|>\n" + content + tokenizer.eos_token + "\n"
        else:
            raise ValueError(f"Invalid role: {role}")
    return text


def encode_messages(example: dict, tokenizer, max_seq_length: int) -> Optional[Dict[str, torch.Tensor]]:
    """Tokenize one instruction example. Labels = input_ids (no masking), matching
    the upstream `encode_with_messages_format` whose masking line is commented out."""
    messages = example.get("messages", [])
    if not messages:
        return None
    text = concat_messages(messages, tokenizer)
    enc = tokenizer(text, return_tensors="pt", max_length=max_seq_length, truncation=True)
    input_ids = enc.input_ids.flatten()
    if input_ids.numel() == 0:
        return None
    return {
        "input_ids": input_ids,
        "labels": input_ids.clone(),
        "attention_mask": torch.ones_like(input_ids),
    }


def load_instruction_samples(
    train_files: List[str],
    tokenizer,
    max_seq_length: int,
    percentage: float = 1.0,
    seed: int = 0,
    max_train_samples: Optional[int] = None,
    read_limit_per_file: Optional[int] = None,
) -> List[Dict[str, torch.Tensor]]:
    """Read instruction jsonl files, subsample, and tokenize.

    `read_limit_per_file` caps how many raw lines are read per file (fast smoke
    tests); `percentage` then subsamples; `max_train_samples` is a final cap.
    """
    raw: List[dict] = []
    for path in train_files:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Instruction file not found: {path}")
        with open(path) as f:
            for i, line in enumerate(f):
                if read_limit_per_file is not None and i >= read_limit_per_file:
                    break
                line = line.strip()
                if line:
                    raw.append(json.loads(line))

    rng = random.Random(seed)
    rng.shuffle(raw)
    if percentage < 1.0:
        raw = raw[: int(len(raw) * percentage)]
    if max_train_samples is not None:
        raw = raw[:max_train_samples]

    samples: List[Dict[str, torch.Tensor]] = []
    for ex in raw:
        enc = encode_messages(ex, tokenizer, max_seq_length)
        if enc is not None:
            samples.append(enc)
    if not samples:
        raise ValueError("No instruction samples were loaded; check --data_dir / files.")
    print(f"[INFO] Loaded {len(samples)} instruction samples "
          f"(from {len(raw)} raw, percentage={percentage}).")
    return samples


# --------------------------------------------------------------------------- #
# MMLU validation target (scoring set)
# --------------------------------------------------------------------------- #
def _format_mmlu_prompt(df: pd.DataFrame, idx: int) -> str:
    choices = ["A", "B", "C", "D"]
    prompt = str(df.iloc[idx, 0])
    for j in range(4):
        prompt += "\n{}. {}".format(choices[j], df.iloc[idx, j + 1])
    prompt += "\nAnswer:"
    return prompt


def build_mmlu_val(
    data_dir: str,
    tokenizer,
    subject: str,
    n_val: int,
    max_seq_length: int,
) -> List[Dict[str, torch.Tensor]]:
    """Build the MMLU dev validation target (LESS `get_mmlu_dataset`, tulu format).

    Query ends in "The answer is:"; completion is the answer letter; the prompt is
    masked (label_only) so the val gradient targets predicting the answer.
    """
    dev_path = os.path.join(data_dir, "eval", "mmlu", "dev", subject + "_dev.csv")
    if not os.path.exists(dev_path):
        raise FileNotFoundError(f"MMLU dev file not found: {dev_path}")
    df = pd.read_csv(dev_path, header=None)
    if len(df) < n_val:
        # Fail explicitly: silently truncating would leave val_samples shorter than the
        # engine's val_batch_size, misaligning the [candidate ++ val] split and corrupting
        # the per-candidate scores with no error.
        raise ValueError(
            f"MMLU subject '{subject}' dev set has only {len(df)} rows but n_val={n_val} "
            f"was requested ({dev_path}). Lower --n_val (<= {len(df)}) or choose a subject "
            f"with a larger dev set."
        )
    df = df[:n_val]

    samples: List[Dict[str, torch.Tensor]] = []
    for i in range(len(df)):
        prompt = _format_mmlu_prompt(df, i)
        prompt = "<|user|>\n" + prompt + "\n<|assistant|>\nThe answer is:"
        completion = " " + str(df.iloc[i, df.shape[1] - 1])  # answer letter (last column)
        full_text = prompt + completion

        prompt_ids = tokenizer.encode(prompt, max_length=max_seq_length, truncation=True)
        full_ids = tokenizer.encode(full_text, max_length=max_seq_length, truncation=True)
        input_ids = torch.tensor(full_ids)
        labels = input_ids.clone()
        labels[: len(prompt_ids)] = -100  # label_only: mask the prompt
        samples.append({
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": torch.ones_like(input_ids),
        })
    print(f"[INFO] Built {len(samples)} MMLU '{subject}' validation examples.")
    return samples


# --------------------------------------------------------------------------- #
# Collate (right-pad a list of variable-length per-sample dicts)
# --------------------------------------------------------------------------- #
def collate(samples: List[Dict[str, torch.Tensor]], pad_token_id: int,
            device=None) -> Dict[str, torch.Tensor]:
    max_len = max(s["input_ids"].numel() for s in samples)
    n = len(samples)
    input_ids = torch.full((n, max_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((n, max_len), dtype=torch.long)
    labels = torch.full((n, max_len), -100, dtype=torch.long)
    for i, s in enumerate(samples):
        L = s["input_ids"].numel()
        input_ids[i, :L] = s["input_ids"]
        attention_mask[i, :L] = s["attention_mask"]
        labels[i, :L] = s["labels"]
    batch = {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
    if device is not None:
        batch = {k: v.to(device) for k, v in batch.items()}
    return batch
