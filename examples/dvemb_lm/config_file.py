"""Configuration for the Data Value Embedding (DVEmb) pipeline.

Mirrors gradproj_lm's config but adds a trainable trajectory (optimizer / LR schedule),
the DVEmb learning-rate mode, and per-stage output directories. The four pipeline stages
are selected by boolean flags (mirrors the reference store_train_grad.py):
    --train_and_store_grad  --compute_embedding  --compute_value  --attribute
"""

import argparse
import os
import sys

# examples/lm/ provides `shared`; the repo root provides `ghostEngines`.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_EXAMPLES_DIR = os.path.dirname(_THIS_DIR)
for _p in (os.path.join(_EXAMPLES_DIR, "lm"), os.path.dirname(_EXAMPLES_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Anchored to this file's directory so each git worktree writes to its own results tree.
RESULTS_DIR = os.path.join(_THIS_DIR, 'results')


def parse_arguments():
    parser = argparse.ArgumentParser(description='Data Value Embedding pipeline')

    # --- Pipeline stage flags ---
    parser.add_argument('--train_and_store_grad', action='store_true',
                        help='Stage 1: train and capture per-step projected gradients')
    parser.add_argument('--compute_embedding', action='store_true',
                        help='Stage 2: reverse-recursion to produce data value embeddings')
    parser.add_argument('--compute_value', action='store_true',
                        help='Stage 3: dot test gradients against embeddings -> value matrix')
    parser.add_argument('--attribute', action='store_true',
                        help='Stage 4: rank training points by value for selected test points')

    # --- Model ---
    parser.add_argument('--architecture', type=str, default='GPT2-Small',
                        choices=['GPT2-Tiny', 'GPT2-Small', 'GPT2-Medium', 'GPT2-Large'])

    # --- Projection (must match across stages so P is reconstructible) ---
    parser.add_argument('--proj_layers', type=str, default='mlp,attn')
    parser.add_argument('--proj_rank_total', type=int, default=256)
    parser.add_argument('--proj_rank_min', type=int, default=8)
    parser.add_argument('--proj_seed', type=int, default=42)
    parser.add_argument('--proj_dtype', type=str, default='float32',
                        choices=['float16', 'bfloat16', 'float32'])
    parser.add_argument('--proj_row_orthonormal', action='store_true')
    parser.add_argument('--include_embeddings', action='store_true')

    # --- DVEmb recursion ---
    parser.add_argument('--lr_mode', type=str, default='scaled', choices=['none', 'scaled'],
                        help="'scaled' folds dynamic per-step lr into the embedding (and 1/B "
                             "Gauss-Newton norm); 'none' reproduces the reference (lr=1).")

    # --- Training trajectory ---
    parser.add_argument('--optimizer', type=str, default='adamw', choices=['adamw', 'sgd'],
                        help="Optimizer for the training run. DVEmb's unrolling is derived for "
                             "SGD; use sgd for the cleanest correctness check.")
    parser.add_argument('--learning_rate', type=float, default=3e-4)
    parser.add_argument('--min_lr', type=float, default=3e-5)
    parser.add_argument('--lr_schedule', type=str, default='constant',
                        choices=['constant', 'linear', 'cosine'])
    parser.add_argument('--lr_decay_steps', type=int, default=-1,
                        help="LR-decay horizon (step at which linear->0 / cosine->min_lr). "
                             "-1 uses max_steps. Decouples the schedule from the run length to "
                             "match the reference (hardcoded num_training_steps=10000).")
    parser.add_argument('--warmup_steps', type=int, default=0)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--momentum', type=float, default=0.0, help='SGD momentum')
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.95)
    parser.add_argument('--max_steps', type=int, default=100,
                        help='Number of training steps to run and capture')

    # --- Data ---
    parser.add_argument('--data_source', type=str, default='pile',
                        choices=['pile', 'synthetic'])
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--block_size', type=int, default=1024)
    parser.add_argument('--n_test', type=int, default=32,
                        help='Number of test examples for the value matrix')
    parser.add_argument('--test_batch_size', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)

    # --- Capture fast path (decoupled in-graph + torch.compile) — DEFAULT ON ---
    # The in-graph decoupled manager + regional block compile is the default capture path: it
    # numerically matches the hook engine and, at the default bf16 train_dtype, is ~25% faster
    # (measured on H200, GPT2-Small). Use
    # --no_decoupled_compile to fall back to the eager hook engine (e.g. Conv1D models, which the
    # decoupled path does not support, or CPU debugging).
    parser.add_argument('--decoupled_compile', dest='decoupled_compile', action='store_true',
                        help='Capture via the in-graph decoupled manager + regional block compile '
                             '(default on). Numerically matches the hook engine; ~25%% faster under '
                             'bf16 autocast.')
    parser.add_argument('--no_decoupled_compile', dest='decoupled_compile', action='store_false',
                        help='Fall back to the eager hook engine (GradProjLoraEngine). Required for '
                             'Conv1D-based models (unsupported by the decoupled path).')
    parser.set_defaults(decoupled_compile=True)
    parser.add_argument('--ac_budget', type=float, default=-1.0,
                        help='With decoupled_compile, Inductor min-cut activation-memory budget in '
                             '(0,1] (compile-native activation checkpointing): lower recomputes more '
                             'in backward to save peak memory. -1 disables (save everything).')

    # --- Precision / system ---
    # Default mixed precision: fp32 master weights + bf16 autocast compute (the common LM setting,
    # and the regime where decoupled_compile wins). proj_dtype stays fp32 for projection fidelity.
    parser.add_argument('--model_dtype', type=str, default='float32',
                        choices=['float32', 'float16', 'bfloat16'])
    parser.add_argument('--train_dtype', type=str, default='bfloat16',
                        choices=['float32', 'float16', 'bfloat16'])
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--verbose', action='store_true')

    # --- Output ---
    parser.add_argument('--output_dir', type=str, default=RESULTS_DIR)
    parser.add_argument('--top_k', type=int, default=10,
                        help='Stage 4: number of top/bottom training points to print')

    return parser.parse_args()


class DVEmbConfig:
    """Configuration object for the DVEmb pipeline.

    Derives ``run_dir`` from every parameter that determines the projection P *and* the
    training trajectory (arch, proj_*, optimizer / lr / schedule / warmup / steps, batch
    size, seed, data source, train dtype), so a later stage invoked with a mismatched
    config resolves to a different dir and fails loudly instead of silently mixing
    artifacts from incompatible runs.
    """

    def __init__(self, args):
        self.args = args

        # Stages
        self.train_and_store_grad = args.train_and_store_grad
        self.compute_embedding = args.compute_embedding
        self.compute_value = args.compute_value
        self.attribute = args.attribute

        # Model
        self.architecture = args.architecture

        # Projection
        self.proj_layers = args.proj_layers
        self.proj_rank_total = args.proj_rank_total
        self.proj_rank_min = args.proj_rank_min
        self.proj_seed = args.proj_seed
        self.proj_dtype = args.proj_dtype
        self.proj_row_orthonormal = args.proj_row_orthonormal
        self.include_embeddings = args.include_embeddings

        # DVEmb
        self.lr_mode = args.lr_mode

        # Training
        self.optimizer = args.optimizer
        self.learning_rate = args.learning_rate
        self.min_lr = args.min_lr
        self.lr_schedule = args.lr_schedule
        self.lr_decay_steps = args.lr_decay_steps
        self.warmup_steps = args.warmup_steps
        self.weight_decay = args.weight_decay
        self.momentum = args.momentum
        self.beta1 = args.beta1
        self.beta2 = args.beta2
        self.max_steps = args.max_steps

        # Data
        self.data_source = args.data_source
        self.batch_size = args.batch_size
        self.n_test = args.n_test
        self.test_batch_size = args.test_batch_size
        self.seed = args.seed

        # Clamp block size to the model's block size (e.g. GPT2-Tiny=64).
        self.block_size = args.block_size
        try:
            from shared.GPT2_configs import get_model_config
            model_block = get_model_config(self.architecture)['block_size']
            self.block_size = min(self.block_size, model_block)
        except (ImportError, ValueError, KeyError):
            pass

        # Capture fast path
        self.decoupled_compile = args.decoupled_compile
        self.ac_budget = args.ac_budget

        # Precision / system
        self.model_dtype = args.model_dtype
        self.train_dtype = args.train_dtype
        self.device = args.device
        self.verbose = args.verbose
        self.top_k = args.top_k

        # Output layout: a run-specific root with per-stage subdirs so all stages
        # (possibly separate invocations) agree on paths from the same config.
        # The name must encode EVERY parameter that determines the projection P *and*
        # the training trajectory, so a later stage run with a mismatched config
        # resolves to a different dir and fails loudly (missing checkpoint/embeddings)
        # rather than silently dotting test gradients against embeddings built under an
        # incompatible P or a different trajectory.
        # proj_rank_min / include_embeddings change the projected dims (also caught by
        # the dim check in dvemb_value), while proj_row_orthonormal / proj_dtype change
        # P's values at identical dims (NOT caught there) -- all are included here.
        proj_id = (f"rank_{self.proj_rank_total}_rmin_{self.proj_rank_min}"
                   f"_seed_{self.proj_seed}_ortho_{int(self.proj_row_orthonormal)}"
                   f"_pdt_{self.proj_dtype}_emb_{int(self.include_embeddings)}")
        # lr_decay_steps only affects the trajectory (not P); encode it when it decouples the
        # decay horizon from the run length so a frozen-tail run does not collide with a plain one.
        decay_tag = f"_decay_{self.lr_decay_steps}" if self.lr_decay_steps > 0 else ""
        # train_dtype changes the training trajectory (bf16 diverges from fp32) and thus the captured
        # gradients, though not P. Tag non-fp32 so a bf16 run (now the default) does not silently
        # overwrite / read a stale fp32 capture at the same path. fp32 stays untagged (back-compat).
        dtype_tag = f"_tdt_{self.train_dtype}" if self.train_dtype != 'float32' else ""
        # Trajectory identity: seed (model init + data stream), warmup_steps, and lr_schedule
        # each change the captured trajectory, so they are encoded unconditionally; min_lr /
        # momentum / weight_decay are tagged only when they differ from the argparse defaults
        # (mirrors the decay_tag / dtype_tag back-compat pattern). NOTE: adding these tags
        # intentionally changes run-dir names vs pre-2026-07 captures — a rerun resolves to a
        # fresh dir (and re-trains) rather than silently mixing with an older trajectory.
        minlr_tag = f"_minlr_{self.min_lr}" if self.min_lr != 3e-5 else ""
        mom_tag = f"_mom_{self.momentum}" if self.momentum != 0.0 else ""
        wd_tag = f"_wd_{self.weight_decay}" if self.weight_decay != 0.0 else ""
        run_name = (f"arch_{self.architecture}_layers_{self.proj_layers}"
                    f"_{proj_id}"
                    f"_opt_{self.optimizer}_lr_{self.learning_rate}"
                    f"_sched_{self.lr_schedule}_warm_{self.warmup_steps}"
                    f"{minlr_tag}{mom_tag}{wd_tag}"
                    f"_steps_{self.max_steps}{decay_tag}_bs_{self.batch_size}"
                    f"_seed_{self.seed}"
                    f"_lrmode_{self.lr_mode}_data_{self.data_source}{dtype_tag}")
        self.run_dir = os.path.join(args.output_dir, run_name)
        self.capture_dir = os.path.join(self.run_dir, 'capture')
        self.embed_dir = os.path.join(self.run_dir, 'embedding')
        self.value_dir = os.path.join(self.run_dir, 'value')
        self.checkpoint_path = os.path.join(self.capture_dir, 'final_model.pt')
        os.makedirs(self.run_dir, exist_ok=True)

    def __repr__(self):
        return (f"DVEmbConfig(arch={self.architecture}, opt={self.optimizer}, "
                f"steps={self.max_steps}, lr_mode={self.lr_mode}, data={self.data_source})")
