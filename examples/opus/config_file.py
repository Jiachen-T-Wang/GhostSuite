"""Configuration for the OPUS online-data-selection pretraining example.

Mirrors examples/greats/pretrain/config_file.py (so it reuses the same shared/ model + data
utilities and stays batch-for-batch comparable with the committed GREATS experiment) and adds
the OPUS-specific knobs: sketch dimensions, the selection method (Boltzmann stochastic-greedy /
greedy / top-k), its temperature, and the optimizer-induced preconditioner mode. "OPUS
reference" defaults refer to the reference implementation at github.com/gszfwsb/OPUS.
"""

import argparse
import os
import sys
from datetime import datetime

# Reuse the shared/ utilities that live under examples/lm/.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_EXAMPLES_DIR = os.path.dirname(_THIS_DIR)                           # .../examples
_REPO_ROOT = os.path.dirname(_EXAMPLES_DIR)                          # repo root
for _p in (os.path.join(_EXAMPLES_DIR, "lm"), _REPO_ROOT, _THIS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from shared.utils import build_result_dir


# Results are anchored to this file's directory so each worktree writes to its
# own results/ tree instead of a shared absolute path.
RESULTS_DIR = os.path.join(_THIS_DIR, "results")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="OPUS online data selection (sketched, diversity-aware) pretraining."
    )

    # Method: OPUS (online selection) or Regular (no selection baseline).
    parser.add_argument("--method", type=str, default="OPUS",
                        choices=["OPUS", "Regular"])

    # Candidate pool (OPUS's buffer): N scored candidates per step, k selected.
    parser.add_argument("--candidate_batch_size", type=int, default=None,
                        help="Candidate pool size N scored each step; the trained subset "
                             "size k is --batch_size (N >= k). Defaults to 2 * batch_size "
                             "(selection ratio 0.5, the OPUS reference default).")

    # OPUS selection knobs.
    parser.add_argument("--opus_selection_method", type=str, default="stochastic",
                        choices=["stochastic", "greedy", "topk"],
                        help="stochastic = Boltzmann stochastic-greedy with the Gram "
                             "redundancy penalty (the OPUS default); greedy = deterministic "
                             "argmax with the penalty; topk = plain top-k by score "
                             "(no diversity term — the GREATS-style ablation arm).")
    parser.add_argument("--opus_temperature", type=float, default=0.9,
                        help="Boltzmann temperature for stochastic selection (the OPUS "
                             "reference default is 0.9; calibrate to the observed score "
                             "scale — see experiments/README.md).")
    parser.add_argument("--opus_preconditioner", type=str, default="adamw_scalar",
                        choices=["none", "adamw_scalar"],
                        help="none = raw gradient inner products (OPUS's 'sgd' mode); "
                             "adamw_scalar = OPUS's per-layer AdamW scalar factors "
                             "C_t/sqrt(numel) on the train side (the element-wise AdamW "
                             "diagonal needs materialized per-sample gradients and is not "
                             "supported — see opus_scorer.py).")

    # Sketch (gradient projection) knobs.
    parser.add_argument("--proj_dim", type=int, default=8192,
                        help="Per-layer sketch budget k_i*k_o (split per layer by the "
                             "engine's aspect-ratio rule). The OPUS reference's CountSketch "
                             "default is 8192 per layer.")
    parser.add_argument("--proj_rank_min", type=int, default=4,
                        help="Minimum per-side projection dimension.")
    parser.add_argument("--proj_seed", type=int, default=4242,
                        help="Seed for the projection matrices (fixed across the run).")
    parser.add_argument("--proj_orthonormal", action="store_true",
                        help="Row-orthonormal (calibrated) projections instead of Gaussian; "
                             "with full per-layer dims this makes the sketch inner products "
                             "EXACT (used by the equivalence tests).")
    parser.add_argument("--proj_layers", type=str, default="c_attn,c_fc,c_proj",
                        help="Comma-separated substrings of the Linear layers to score "
                             "(default: all transformer-block Linears; the tied "
                             "wte/lm_head are excluded — the recommended GREATS setting).")
    parser.add_argument("--score_seq_len", type=int, default=None,
                        help="Score on a prefix window of this many tokens (the OPUS "
                             "reference's score_len efficiency knob). Default: the full "
                             "sequence, matching how the GREATS experiment scores.")

    # Architecture.
    parser.add_argument("--architecture", type=str, default="GPT2-Small",
                        choices=["GPT2-Tiny", "GPT2-Small", "GPT2-Medium", "GPT2-Large"])

    # Training.
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Trained subset size k (samples kept per step).")
    parser.add_argument("--val_batch_size", type=int, default=16,
                        help="Proxy (scoring-target) batch size m.")
    parser.add_argument("--warmup_step", type=int, default=2000)
    parser.add_argument("--learning_rate", type=float, default=6e-4)
    parser.add_argument("--min_lr", type=float, default=None,
                        help="Floor of the cosine LR decay (default: 0.1 * learning_rate)")
    parser.add_argument("--lr_decay_iters", type=int, default=None,
                        help="Cosine LR decay horizon in steps (default: --max_steps)")
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adamw"],
                        help="Optimizer (only AdamW is implemented)")
    parser.add_argument("--max_steps", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)

    # Dataset.
    parser.add_argument("--train_set", type=str, default="pile")
    parser.add_argument("--val_set", type=str, default="pile",
                        help="Validation dataset name; currently not used")

    # Evaluation.
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--eval_interval", type=int, default=200)
    parser.add_argument("--eval_iter", type=int, default=20)
    parser.add_argument("--eval_bs", type=int, default=16)
    parser.add_argument("--eval_seed", type=int, default=1234,
                        help="Fixed RNG seed for the evaluation window set; identical "
                             "across training seeds so val/test loss noise is "
                             "seed-independent")

    # Precision.
    parser.add_argument("--model_dtype", type=str, default="float32",
                        choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--train_dtype", type=str, default="bfloat16",
                        choices=["float32", "bfloat16"])

    # Scoring target population. Default ON: the per-step proxy batch is drawn from the
    # SAME fixed window pool used to compute eval loss, so selection targets exactly the
    # eval population (identical to the GREATS experiment protocol).
    parser.add_argument("--score_val_from_eval_pool",
                        action=argparse.BooleanOptionalAction, default=True,
                        help="Draw the proxy batch from the fixed eval window pool")

    # WandB.
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="GhostSuite")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--wandb_mode", type=str, default="online",
                        choices=["online", "offline", "disabled"])
    parser.add_argument("--wandb_dir", type=str, default=None)

    return parser.parse_args()


class TrainingConfig:
    """Training configuration object derived from parsed CLI args."""

    def __init__(self, args):
        self.args = args
        self.architecture = args.architecture

        # Sequence length (block size) is fixed by the GPT config table.
        try:
            from shared.GPT2_configs import get_model_config
            self.block_size = get_model_config(self.architecture)["block_size"]
        except (ImportError, ValueError, KeyError):
            self.block_size = 1024

        # Core hyperparameters.
        self.batch_size = args.batch_size            # k: trained subset size
        self.val_batch_size = args.val_batch_size    # m: proxy (scoring target) size
        self.learning_rate = args.learning_rate
        self.min_lr = args.min_lr if args.min_lr is not None else self.learning_rate * 0.1
        self.max_steps = args.max_steps
        self.seed = args.seed

        # OPUS selection knobs.
        self.method = args.method
        self.candidate_batch_size = (
            args.candidate_batch_size if args.candidate_batch_size is not None
            else 2 * self.batch_size
        )
        self.opus_selection_method = args.opus_selection_method
        self.opus_temperature = args.opus_temperature
        self.opus_preconditioner = args.opus_preconditioner
        self.proj_dim = args.proj_dim
        self.proj_rank_min = args.proj_rank_min
        self.proj_seed = args.proj_seed
        self.proj_orthonormal = args.proj_orthonormal
        self.proj_layers = args.proj_layers
        self.score_seq_len = args.score_seq_len

        if self.method == "OPUS" and self.candidate_batch_size < self.batch_size:
            raise ValueError(
                f"--candidate_batch_size ({self.candidate_batch_size}) must be >= "
                f"--batch_size ({self.batch_size}) for OPUS selection."
            )
        if self.score_seq_len is not None and self.score_seq_len <= 0:
            raise ValueError(f"--score_seq_len must be positive, got {self.score_seq_len}.")

        # Optimizer settings (AdamW), identical to the GREATS example.
        self.optimizer = args.optimizer
        self.weight_decay = 1e-1
        self.beta1 = 0.9
        self.beta2 = 0.95
        self.grad_clip = 1.0
        self.warmup_iters = args.warmup_step
        self.lr_decay_iters = args.lr_decay_iters if args.lr_decay_iters is not None else args.max_steps
        self.decay_lr = True

        # System.
        self.device = "cuda"
        self.compile = False
        self.backend = "nccl"

        # Precision.
        self.model_dtype = args.model_dtype
        self.train_dtype = args.train_dtype

        # No gradient accumulation in this example.
        self.full_batch_size = args.batch_size
        self.gradient_accumulation_steps = 1

        # Evaluation.
        self.eval_iters = args.eval_iter
        self.eval_interval = args.eval_interval
        self.eval_bs = args.eval_bs
        self.eval_seed = args.eval_seed

        # Scoring options.
        self.score_val_from_eval_pool = args.score_val_from_eval_pool
        # Selection is online: refresh the proxy batch every step.
        self.dynamic_val_batch = True

        # WandB.
        self.use_wandb = args.wandb
        self.wandb_project = args.wandb_project
        if args.wandb_run_name:
            self.wandb_run_name = args.wandb_run_name
        else:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.wandb_run_name = (
                f"{args.method}_{args.architecture}_bs{args.batch_size}"
                f"_cand{self.candidate_batch_size}_{args.opus_selection_method}"
                f"_lr{args.learning_rate}_{ts}"
            )
        self.wandb_mode = args.wandb_mode

        # Result directory.
        self.result_folder = os.path.join(RESULTS_DIR, self.wandb_run_name)
        self.setup_result_directories()
        self.wandb_dir = args.wandb_dir or self.result_dir

    def setup_result_directories(self):
        if not os.path.exists(self.result_folder):
            print(f"Results folder '{self.result_folder}' was created.")
        os.makedirs(self.result_folder, exist_ok=True)
        self.result_dir = build_result_dir(self.result_folder, self.method, self.args)
        if not os.path.exists(self.result_dir):
            print(f"Results directory '{self.result_dir}' was created.")
        os.makedirs(self.result_dir, exist_ok=True)

    def get_result_file_path(self):
        return os.path.join(self.result_dir + "_results.json")
