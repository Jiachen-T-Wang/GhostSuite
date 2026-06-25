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
    """Tokenize one instruction example, masking the non-assistant tokens (only the
    assistant completions carry a loss). Port of the up-to-date upstream
    `encode_with_messages_format` (the masking is active in current GREATS)."""
    messages = example.get("messages", [])
    if not messages:
        return None
    text = concat_messages(messages, tokenizer)
    enc = tokenizer(text, return_tensors="pt", max_length=max_seq_length, truncation=True)
    input_ids = enc.input_ids  # [1, L]
    if input_ids.numel() == 0:
        return None
    labels = input_ids.clone()

    # Mask every non-assistant span so only assistant-response tokens are learned.
    for idx, message in enumerate(messages):
        if message["role"] == "assistant":
            continue
        if idx == 0:
            start = 0
        else:
            start = tokenizer(concat_messages(messages[:idx], tokenizer),
                              return_tensors="pt", max_length=max_seq_length,
                              truncation=True).input_ids.shape[1]
        if idx < len(messages) - 1 and messages[idx + 1]["role"] == "assistant":
            so_far = concat_messages(messages[:idx + 1], tokenizer) + "<|assistant|>\n"
        else:
            so_far = concat_messages(messages[:idx + 1], tokenizer)
        end = tokenizer(so_far, return_tensors="pt", max_length=max_seq_length,
                        truncation=True).input_ids.shape[1]
        labels[:, start:end] = -100
        if end >= max_seq_length:
            break

    input_ids = input_ids.flatten()
    labels = labels.flatten()
    if (labels != -100).sum() == 0:
        return None  # no assistant tokens survived truncation
    return {
        "input_ids": input_ids,
        "labels": labels,
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


def build_mmlu_examples(
    data_dir: str,
    tokenizer,
    subject: str,
    k: int,
    max_seq_length: int,
    split: str = "dev",
    strict: bool = False,
) -> List[Dict[str, torch.Tensor]]:
    """Build MMLU examples (LESS `get_mmlu_dataset`/`tokenize` format) from the dev or
    test CSV. Query ends in "The answer is:"; completion is the answer letter; the prompt
    is masked, so the loss/gradient is on the answer token only. `strict` (dev/scoring)
    fails if the split has fewer than `k` rows."""
    path = os.path.join(data_dir, "eval", "mmlu", split, f"{subject}_{split}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"MMLU {split} file not found: {path}")
    df = pd.read_csv(path, header=None)
    if strict and len(df) < k:
        # Silently truncating the scoring val would leave val_samples shorter than the
        # engine's val_batch_size, misaligning the [candidate ++ val] split.
        raise ValueError(
            f"MMLU subject '{subject}' {split} set has only {len(df)} rows but k={k} "
            f"was requested ({path}). Lower the request (<= {len(df)}) or pick a subject "
            f"with a larger {split} set."
        )
    df = df[: min(k, len(df))]

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
        labels[: len(prompt_ids)] = -100  # mask the prompt; loss on the answer token
        samples.append({
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": torch.ones_like(input_ids),
        })
    return samples


def build_mmlu_val(data_dir, tokenizer, subject, n_val, max_seq_length):
    """The dev/scoring validation pool (strict: fail if dev set < n_val)."""
    samples = build_mmlu_examples(data_dir, tokenizer, subject, n_val, max_seq_length,
                                  split="dev", strict=True)
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
