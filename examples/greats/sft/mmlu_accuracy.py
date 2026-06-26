"""MMLU few-shot accuracy eval, ported from upstream GREATS `less/train/mmlu_eval.py`.

Builds an in-context (few-shot) prompt from the subject's dev rows, appends each test
question, and reads the next-token probability restricted to the answer-choice tokens
(A/B/C/D), argmax vs the gold answer. This is the metric the upstream README reports.
"""

import os
from typing import List

import pandas as pd
import torch

CHOICES = ["A", "B", "C", "D"]


def _format_subject(subject: str) -> str:
    return "".join(" " + e for e in subject.split("_"))


def _format_example(df: pd.DataFrame, idx: int, include_answer: bool = True) -> str:
    prompt = str(df.iloc[idx, 0])
    k = df.shape[1] - 2
    for j in range(k):
        prompt += "\n{}. {}".format(CHOICES[j], df.iloc[idx, j + 1])
    prompt += "\nAnswer:"
    if include_answer:
        prompt += " {}\n\n".format(df.iloc[idx, k + 1])
    return prompt


def _gen_prompt(dev_df: pd.DataFrame, subject: str, k: int = -1) -> str:
    prompt = ("The following are multiple choice questions (with answers) about{}.\n\n"
              .format(_format_subject(subject)))
    if k == -1:
        k = dev_df.shape[0]
    for i in range(k):
        prompt += _format_example(dev_df, i)
    return prompt


def _build_icl_prompts(tokenizer, dev_df, test_df, subject, n_val, max_seq_length=512) -> List[str]:
    """One ICL prompt per test question (tulu chat format), shrinking the few-shot count
    until the prompt fits `max_seq_length` (mirrors upstream)."""
    prompts = []
    for i in range(test_df.shape[0]):
        prompt_end = _format_example(test_df, i, include_answer=False)
        k = n_val
        while True:
            prompt = _gen_prompt(dev_df, subject, k) + prompt_end
            prompt = "<|user|>\n" + prompt + "\n<|assistant|>\n"
            prompt += "The answer is:" if prompt[-1] in ["\n", " "] else " The answer is:"
            n_tok = len(tokenizer(prompt, truncation=False, add_special_tokens=False).input_ids)
            if n_tok <= max_seq_length or k <= 0:
                break
            k -= 1
        prompts.append(prompt)
    return prompts


@torch.no_grad()
def _next_token_choice(model, tokenizer, prompts, answer_choice_ids, device, batch_size=8):
    preds = []
    prev_side = tokenizer.padding_side
    tokenizer.padding_side = "left"  # so logits[:, -1] is the true last token under padding
    try:
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i + batch_size]
            enc = tokenizer(batch, padding="longest", return_tensors="pt", add_special_tokens=True)
            input_ids = enc.input_ids.to(device)
            attn = enc.attention_mask.to(device)
            logits = model(input_ids=input_ids, attention_mask=attn).logits[:, -1, :]
            choice_logits = logits[:, answer_choice_ids]
            preds.extend(torch.argmax(choice_logits, dim=-1).tolist())
    finally:
        tokenizer.padding_side = prev_side
    return preds


@torch.no_grad()
def compute_mmlu_accuracy(model, tokenizer, data_dir, subject, n_val, n_test, device,
                          batch_size=8, max_seq_length=512):
    mmlu_dir = os.path.join(data_dir, "eval", "mmlu")
    dev_df = pd.read_csv(os.path.join(mmlu_dir, "dev", subject + "_dev.csv"), header=None)
    dev_df = dev_df[: min(n_val, len(dev_df))]
    test_df = pd.read_csv(os.path.join(mmlu_dir, "test", subject + "_test.csv"), header=None)
    test_df = test_df[: min(n_test, len(test_df))]

    answer_choice_ids = [tokenizer.encode(" " + c, add_special_tokens=False)[-1] for c in CHOICES]
    prompts = _build_icl_prompts(tokenizer, dev_df, test_df, subject, n_val, max_seq_length)

    was_training = model.training
    model.eval()
    preds = _next_token_choice(model, tokenizer, prompts, answer_choice_ids, device, batch_size)
    if was_training:
        model.train()

    gold = test_df.iloc[:, -1].values
    cors = [CHOICES[p] == g for p, g in zip(preds, gold)]
    return float(sum(cors)) / max(1, len(cors)), len(cors)
