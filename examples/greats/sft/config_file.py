"""Configuration for the GREATS SFT (LoRA instruction tuning) example.

Mirrors the upstream GREATS/LESS `base_training_args.sh` + `warmup_train.sh`. See
docs/plans/greats_sft_phaseB_2026-06-25.md.
"""

import argparse
import os

# Default local paths (verified available on this cluster).
DEFAULT_MODEL_PATH = (
    "/scratch/gpfs/PMITTAL/tongwu/other/llama/models--meta-llama--Llama-2-7b-hf/"
    "snapshots/6fdf2e60f86ff2481f2241aaee459f85b5b0bbb9"
)
DEFAULT_DATA_DIR = "/scratch/gpfs/PMITTAL/tongwu/other/GREATS/data"
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(_THIS_DIR, "results")


def parse_arguments():
    p = argparse.ArgumentParser(description="GREATS SFT online batch selection (first-order).")

    # Method.
    p.add_argument("--method", type=str, default="GREATS", choices=["GREATS", "Regular"])
    p.add_argument("--select_metric", type=str, default="dot", choices=["dot", "cosine"])

    # Model / data paths.
    p.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH)
    p.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR)
    p.add_argument("--subject", type=str, default="sociology",
                   help="MMLU subject used as the validation/scoring target")
    p.add_argument("--n_val", type=int, default=5,
                   help="number of MMLU dev examples in the scoring/val pool")
    p.add_argument("--val_batchsize", type=int, default=2,
                   help="val mini-batch resampled from the n_val pool each scoring step")
    p.add_argument("--n_test", type=int, default=500,
                   help="number of MMLU test questions for the accuracy eval")

    # Online selection (upstream uses fracinv: candidate pool = fracinv * batch).
    p.add_argument("--batch_size", type=int, default=4, help="trained subset size k")
    p.add_argument("--fracinv", type=float, default=2.0,
                   help="candidate pool size N = round(fracinv * batch_size); N>=k")

    # LoRA (base_training_args.sh).
    p.add_argument("--lora_r", type=int, default=128)
    p.add_argument("--lora_alpha", type=int, default=1)
    p.add_argument("--lora_dropout", type=float, default=0.1)
    p.add_argument("--lora_target_modules", type=str,
                   default="q_proj,k_proj,v_proj,o_proj")

    # Optimization.
    p.add_argument("--learning_rate", type=float, default=2e-5)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--num_train_epochs", type=int, default=3)
    p.add_argument("--max_steps", type=int, default=-1,
                   help="cap on optimizer steps (-1 = full epochs)")
    p.add_argument("--max_seq_length", type=int, default=512)

    # Data sampling.
    p.add_argument("--percentage", type=float, default=0.05,
                   help="fraction of the instruction corpus to use")
    p.add_argument("--max_train_samples", type=int, default=None)
    p.add_argument("--read_limit_per_file", type=int, default=None,
                   help="cap raw lines read per jsonl file (fast smoke tests)")
    p.add_argument("--data_seed", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)

    # Precision.
    p.add_argument("--model_dtype", type=str, default="bfloat16",
                   choices=["float32", "bfloat16", "float16"])

    # Logging / eval.
    p.add_argument("--logging_steps", type=int, default=1)
    p.add_argument("--eval_interval", type=int, default=50,
                   help="log MMLU val loss every N optimizer steps (0=off)")
    p.add_argument("--device", type=str, default="cuda")

    return p.parse_args()


class TrainingConfig:
    def __init__(self, args):
        self.args = args
        self.method = args.method
        self.select_metric = args.select_metric

        self.model_path = args.model_path
        self.data_dir = args.data_dir
        self.subject = args.subject
        self.n_val = args.n_val
        self.val_batchsize = args.val_batchsize
        self.n_test = args.n_test

        self.batch_size = args.batch_size            # k
        self.fracinv = args.fracinv
        self.candidate_batch_size = max(self.batch_size, round(args.fracinv * args.batch_size))

        self.lora_r = args.lora_r
        self.lora_alpha = args.lora_alpha
        self.lora_dropout = args.lora_dropout
        self.lora_target_modules = [s.strip() for s in args.lora_target_modules.split(",") if s.strip()]

        self.learning_rate = args.learning_rate
        self.weight_decay = args.weight_decay
        self.warmup_ratio = args.warmup_ratio
        self.grad_clip = args.grad_clip
        self.num_train_epochs = args.num_train_epochs
        self.max_steps = args.max_steps
        self.max_seq_length = args.max_seq_length

        self.percentage = args.percentage
        self.max_train_samples = args.max_train_samples
        self.read_limit_per_file = args.read_limit_per_file
        self.data_seed = args.data_seed
        self.seed = args.seed

        self.model_dtype = args.model_dtype
        self.logging_steps = args.logging_steps
        self.eval_interval = args.eval_interval
        self.device = args.device

        # Per-step gradient norms are needed for cosine ranking.
        self.log_grad_norms = (self.select_metric == "cosine")

        self.train_files = [
            os.path.join(self.data_dir, "train", "processed", name, f"{name}_data.jsonl")
            for name in ("flan_v2", "cot", "dolly", "oasst1")
        ]

        run_name = (f"{self.method}_{self.subject}_bs{self.batch_size}"
                    f"_frac{self.fracinv}_lr{self.learning_rate}_seed{self.seed}")
        self.result_dir = os.path.join(RESULTS_DIR, run_name)
        os.makedirs(self.result_dir, exist_ok=True)
