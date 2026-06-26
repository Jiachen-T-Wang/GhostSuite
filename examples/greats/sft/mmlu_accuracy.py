"""MMLU evals, ported from upstream GREATS.

- `compute_mmlu_perplexity`: eval/test answer perplexity (the metric in the upstream
  README's `trialrun.png`) = exp(mean CE on the answer token) over the dev (eval) and
  test sets, with the prompt masked (LESS `tokenize`/`gctrainer` eval loss).
- `compute_mmlu_accuracy`: few-shot ICL accuracy = next-token argmax over A/B/C/D.
"""

import math
import os
from typing import List

import pandas as pd
import torch

CHOICES = ["A", "B", "C", "D"]


@torch.no_grad()
def _set_perplexity(model, tokenizer, examples, device, batch_size=8):
    from data_utils import collate
    if not examples:
        return float("nan")
    pad_id = tokenizer.pad_token_id
    total_loss, total_tok = 0.0, 0
    for i in range(0, len(examples), batch_size):
        batch = collate(examples[i:i + batch_size], pad_id, device)
        out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                    labels=batch["labels"])
        ntok = int((batch["labels"][..., 1:] != -100).sum().item())  # HF shifts labels
        if ntok > 0:
            total_loss += float(out.loss) * ntok
            total_tok += ntok
    return math.exp(total_loss / max(1, total_tok))


@torch.no_grad()
def compute_mmlu_perplexity(model, tokenizer, data_dir, subject, n_val, n_test, device,
                            max_seq_length=512, batch_size=8):
    """Return (eval_ppl, test_ppl): answer perplexity on the dev (n_val) and test
    (n_test) sets. Matches the upstream eval-loss -> exp() metric in trialrun.png."""
    from data_utils import build_mmlu_examples
    eval_ex = build_mmlu_examples(data_dir, tokenizer, subject, n_val, max_seq_length, split="dev")
    test_ex = build_mmlu_examples(data_dir, tokenizer, subject, n_test, max_seq_length, split="test")
    was_training = model.training
    model.eval()
    eval_ppl = _set_perplexity(model, tokenizer, eval_ex, device, batch_size)
    test_ppl = _set_perplexity(model, tokenizer, test_ex, device, batch_size)
    if was_training:
        model.train()
    return eval_ppl, test_ppl


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
