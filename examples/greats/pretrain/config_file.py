"""Configuration for the GREATS online-batch-selection pretraining example.

This mirrors examples/lm/graddotprod_lm/config_file.py (so it can reuse the same
shared/ model + data utilities) and adds the GREATS-specific knobs: a candidate
pool size and the selection metric.
"""

import argparse
import os
import sys
from datetime import datetime

# Reuse the shared/ utilities that live under examples/lm/.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_EXAMPLES_DIR = os.path.dirname(os.path.dirname(_THIS_DIR))          # .../examples
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
        description="GREATS online batch selection (first-order) pretraining."
    )

    # Method: GREATS (online selection) or Regular (no selection baseline).
    parser.add_argument("--method", type=str, default="GREATS",
                        choices=["GREATS", "Regular"])

    # GREATS-specific knobs.
    parser.add_argument("--candidate_batch_size", type=int, default=None,
                        help="Candidate pool size N scored each step. The trained "
                             "subset size k is --batch_size; requires N >= k. "
                             "Defaults to 2 * batch_size.")
    parser.add_argument("--select_metric", type=str, default="dot",
                        choices=["dot", "cosine"],
                        help="Score used to rank candidates: raw train-val dot "
                             "product (dot) or cosine (requires per-sample grad "
                             "norms; forces --log_grad_norms).")

    # Architecture.
    parser.add_argument("--architecture", type=str, default="GPT2-Small",
                        choices=["GPT2-Tiny", "GPT2-Small", "GPT2-Medium", "GPT2-Large"])

    # Training.
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Trained subset size k (samples kept per step).")
    parser.add_argument("--val_batch_size", type=int, default=16,
                        help="Validation batch size m used as the scoring target.")
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
                        choices=["float32", "float16", "bfloat16"])

    # Scoring target population. Default ON: the per-step scoring val batch is drawn
    # from the SAME fixed window pool used to compute eval loss, so selection targets
    # exactly the eval population (see shared/training_utils.setup_data_functions).
    parser.add_argument("--score_val_from_eval_pool",
                        action=argparse.BooleanOptionalAction, default=True,
                        help="Draw the scoring val batch from the fixed eval window pool")
    parser.add_argument("--log_grad_norms", action="store_true",
                        help="Record per-sample train grad norms (required for cosine)")
    parser.add_argument("--score_exclude_params", type=str, default=None,
                        help="Comma-separated param-name substrings to EXCLUDE from the "
                             "score (e.g. 'wte,lm_head'); training is unaffected")

    # Engine fast path. The decoupled in-graph + torch.compile path is the DEFAULT (same as
    # examples/lm/graddotprod_lm): each GPT-2 block runs compiled in BOTH the scoring and update
    # passes and the dot-product is a small in-graph transient. --eager selects the per-layer-hook
    # engine (needed for --select_metric cosine, which the fast path does not support yet).
    parser.add_argument("--decoupled_fn", dest="decoupled_fn", action="store_true",
                        help="Use the decoupled in-graph fast path (default).")
    parser.add_argument("--eager", dest="decoupled_fn", action="store_false",
                        help="Use the eager per-layer-hook engine instead of the decoupled "
                             "in-graph + torch.compile fast path.")
    parser.add_argument("--decoupled_compile", dest="decoupled_compile", action="store_true",
                        help="Regional-compile the transformer blocks (default on with the "
                             "decoupled path; the speedup lever).")
    parser.add_argument("--no_decoupled_compile", dest="decoupled_compile", action="store_false",
                        help="Keep the decoupled path but skip torch.compile (decoupled-eager).")
    parser.set_defaults(decoupled_fn=True, decoupled_compile=True)
    parser.add_argument("--decoupled_compile_toplevel", action="store_true",
                        help="Also compile the top-level in-graph layers (wpe + final norm); the "
                             "tied wte/lm_head stay on the eager capture path.")
    parser.add_argument("--decoupled_mem_budget", type=float, default=None,
                        help="With --decoupled_compile, Inductor activation-memory budget in (0,1] "
                             "(compile-native activation checkpointing); lower = recompute more.")

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

        # Sequence length (block size) is fixed by the GPT config table; sampled
        # windows must match it (e.g. GPT2-Tiny=64).
        try:
            from shared.GPT2_configs import get_model_config
            self.block_size = get_model_config(self.architecture)["block_size"]
        except (ImportError, ValueError, KeyError):
            self.block_size = 1024

        # Core hyperparameters.
        self.batch_size = args.batch_size            # k: trained subset size
        self.val_batch_size = args.val_batch_size    # m: scoring target size
        self.learning_rate = args.learning_rate
        self.min_lr = args.min_lr if args.min_lr is not None else self.learning_rate * 0.1
        self.max_steps = args.max_steps
        self.seed = args.seed

        # GREATS selection knobs.
        self.method = args.method
        self.candidate_batch_size = (
            args.candidate_batch_size if args.candidate_batch_size is not None
            else 2 * self.batch_size
        )
        self.select_metric = args.select_metric
        # Cosine ranking needs per-sample gradient norms from the engine.
        self.log_grad_norms = args.log_grad_norms or (self.select_metric == "cosine")

        # Engine fast path (decoupled in-graph + torch.compile), default on.
        self.decoupled_fn = args.decoupled_fn
        self.decoupled_compile = args.decoupled_compile
        self.decoupled_compile_toplevel = args.decoupled_compile_toplevel
        self.decoupled_mem_budget = args.decoupled_mem_budget

        if self.method == "GREATS" and self.candidate_batch_size < self.batch_size:
            raise ValueError(
                f"--candidate_batch_size ({self.candidate_batch_size}) must be >= "
                f"--batch_size ({self.batch_size}) for GREATS selection."
            )

        # The decoupled fast path returns only the per-sample dot (no grad norms), so cosine
        # ranking is eager-only for now (future work: add grad-norm outputs to the in-graph path).
        if self.decoupled_fn and (self.select_metric == "cosine" or args.log_grad_norms):
            raise ValueError(
                "--select_metric cosine / --log_grad_norms require per-sample grad norms, which "
                "the decoupled fast path does not produce. Re-run with --eager for cosine ranking."
            )

        # Optimizer settings (AdamW).
        self.optimizer = args.optimizer
        self.weight_decay = 1e-1
        self.beta1 = 0.9
        self.beta2 = 0.95
        self.grad_clip = 1.0
        self.warmup_iters = args.warmup_step
        # Cosine decay horizon; defaults to the full run (nanoGPT convention).
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
        self.score_exclude_params = (
            [s.strip() for s in args.score_exclude_params.split(",") if s.strip()]
            if args.score_exclude_params else []
        )
        # Selection is online: refresh the scoring val batch every step.
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
                f"_cand{self.candidate_batch_size}_lr{args.learning_rate}_{ts}"
            )
        self.wandb_mode = args.wandb_mode

        # Result directory.
        self.result_folder = os.path.join(RESULTS_DIR, self.wandb_run_name)
        self.setup_result_directories()
        self.wandb_dir = args.wandb_dir or self.result_dir

    def setup_result_directories(self):
        # exist_ok: DDP ranks race on the same paths.
        if not os.path.exists(self.result_folder):
            print(f"Results folder '{self.result_folder}' was created.")
        os.makedirs(self.result_folder, exist_ok=True)
        self.result_dir = build_result_dir(self.result_folder, self.method, self.args)
        if not os.path.exists(self.result_dir):
            print(f"Results directory '{self.result_dir}' was created.")
        os.makedirs(self.result_dir, exist_ok=True)

    def get_result_file_path(self):
        return os.path.join(self.result_dir + "_results.json")
