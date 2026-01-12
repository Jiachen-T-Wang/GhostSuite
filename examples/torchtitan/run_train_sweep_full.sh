#!/bin/bash

#SBATCH --job-name=torchtitan       # Job name
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-type=fail
#SBATCH --mail-user=tw8948@princeton.edu
#SBATCH --output=/scratch/gpfs/PMITTAL/tianhao/slurm_output/slurm-%j.out
#SBATCH --error=/scratch/gpfs/PMITTAL/tianhao/slurm_output/slurm-%j.err
#SBATCH --time=13:59:59             
#SBATCH --nodes=1                    # Number of nodes
#SBATCH --ntasks=1                   # Number of tasks
#SBATCH --cpus-per-task=4            # CPU cores per task
#SBATCH --mem-per-cpu=16G            

#SBATCH --gres=gpu:1
#SBATCH --partition=ailab

# Flexible hyperparameter sweep launcher.
# Usage examples:
#   ./run_train_sweep_full.sh --grid optimizer.lr=3e-4,1e-3 --grid training.local_batch_size=4,8
#   ./run_train_sweep_full.sh --dataset recipe_18 --grid optimizer.lr=3e-4,1e-3

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Default to repository root on scratch to avoid /var/spool paths copied by Slurm.
BASE_DIR=${BASE_DIR:-"/scratch/gpfs/PMITTAL/tianhao/GhostSuite/examples/torchtitan"}
RUN_SCRIPT="${RUN_SCRIPT:-${BASE_DIR}/run_train.sh}"
DUMP_ROOT="${DUMP_ROOT:-${BASE_DIR}/results}"

CONFIG_FILE=${CONFIG_FILE:-"${BASE_DIR}/torchtitan/models/pythia/pythia_70m.toml"}
TRAIN_FILE=${TRAIN_FILE:-"torchtitan.train"}

DEFAULT_WANDB_PROJECT="torchtitan-sweep-debug"
WANDB_PROJECT=${WANDB_PROJECT:-"${DEFAULT_WANDB_PROJECT}"}
WANDB_GROUP=${WANDB_GROUP:-"sweep-$(date +%Y%m%d)"}
JOB_DESCRIPTION=${JOB_DESCRIPTION:-""}
WANDB_PROJECT_FROM_CLI=0

DATASET="c4_500gb"
DATASET_PATH=""
DRY_RUN=0
MAX_RUNS=0   # 0 = no limit

declare -a EXTRA_ARGS=()
declare -a GRID_KEYS=()
declare -A GRID_VALUES=()

add_grid() {
    local key="$1"
    local values="$2"
    GRID_KEYS+=("$key")
    GRID_VALUES["$key"]="$values"
}

# Defaults if no grid is provided via CLI.
add_default_grid() {
    if [[ ${#GRID_KEYS[@]} -eq 0 ]]; then
        add_grid "optimizer.lr" "7e-4,1e-3"
        add_grid "lr_scheduler.warmup_steps" "100"
    fi
}

parse_time_to_seconds() {
    local t="$1"
    if [[ -z "${t}" || "${t}" == "NOT_SET" ]]; then
        echo 0
        return
    fi
    if [[ "${t}" == "UNLIMITED" ]]; then
        echo $((10#3153600000))  # Treat unlimited as very long.
        return
    fi

    local days=0
    if [[ "${t}" == *-* ]]; then
        days="${t%%-*}"
        t="${t#*-}"
    fi

    local hours=0 minutes=0 seconds=0
    IFS=: read -r hours minutes seconds <<<"${t}"
    hours=${hours:-0}
    minutes=${minutes:-0}
    seconds=${seconds:-0}
    echo $((10#$days * 86400 + 10#$hours * 3600 + 10#$minutes * 60 + 10#$seconds))
}

slurm_timelimit_seconds() {
    local limit="${SLURM_TIMELIMIT:-${SBATCH_TIMELIMIT:-}}"
    if [[ -z "${limit}" ]]; then
        if [[ -n "${SLURM_JOB_ID:-}" ]] && command -v scontrol >/dev/null 2>&1; then
            limit=$(scontrol show job "${SLURM_JOB_ID}" 2>/dev/null | awk -F= '/TimeLimit=/{print $2; exit}' || true)
        fi
    fi
    limit="${limit%% *}"  # drop anything after the first space (e.g., TimeLimitRaw=...)
    parse_time_to_seconds "${limit}"
}

is_slurm_job() {
    [[ -n "${SLURM_JOB_ID:-}" || -n "${SLURM_JOB_NAME:-}" || -n "${SLURM_CLUSTER_NAME:-}" ]]
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset)
            DATASET="$2"; shift 2 ;;
        --dataset=*)
            DATASET="${1#*=}"; shift ;;
        --dataset-path)
            DATASET_PATH="$2"; shift 2 ;;
        --dataset-path=*)
            DATASET_PATH="${1#*=}"; shift ;;
        --config-file)
            CONFIG_FILE="$2"; shift 2 ;;
        --config-file=*)
            CONFIG_FILE="${1#*=}"; shift ;;
        --train-file)
            TRAIN_FILE="$2"; shift 2 ;;
        --train-file=*)
            TRAIN_FILE="${1#*=}"; shift ;;
        --wandb-project)
            WANDB_PROJECT="$2"; WANDB_PROJECT_FROM_CLI=1; shift 2 ;;
        --wandb-project=*)
            WANDB_PROJECT="${1#*=}"; WANDB_PROJECT_FROM_CLI=1; shift ;;
        --wandb-group)
            WANDB_GROUP="$2"; shift 2 ;;
        --wandb-group=*)
            WANDB_GROUP="${1#*=}"; shift ;;
        --grid)
            add_grid "${2%%=*}" "${2#*=}"; shift 2 ;;
        --grid=*)
            kv="${1#*=}"; add_grid "${kv%%=*}" "${kv#*=}"; shift ;;
        --max-runs)
            MAX_RUNS="$2"; shift 2 ;;
        --max-runs=*)
            MAX_RUNS="${1#*=}"; shift ;;
        --dry-run)
            DRY_RUN=1; shift ;;
        --)
            shift; EXTRA_ARGS+=("$@"); break ;;
        *)
            EXTRA_ARGS+=("$1"); shift ;;
    esac
done

add_default_grid

if is_slurm_job; then
    SLURM_TIMELIMIT_SECONDS=$(slurm_timelimit_seconds)
    if [[ ${SLURM_TIMELIMIT_SECONDS} -gt 7200 && ${WANDB_PROJECT_FROM_CLI} -eq 0 && "${WANDB_PROJECT}" == "${DEFAULT_WANDB_PROJECT}" ]]; then
        echo "Error: Slurm job with time limit > 2 hours requires explicit --wandb-project (default '${DEFAULT_WANDB_PROJECT}' is for debug)." >&2
        exit 1
    fi
fi

if [[ -n "${DATASET}" ]]; then
    clean_dataset="${DATASET//[^A-Za-z0-9_.-]/_}"
    WANDB_GROUP="${clean_dataset}"
fi

if [[ -z "${WANDB_RUN_GROUP:-}" ]]; then
    WANDB_RUN_GROUP="${WANDB_GROUP}"
fi

if [[ -z "${JOB_DESCRIPTION}" ]]; then
    JOB_DESCRIPTION="sweep-${WANDB_GROUP}"
fi

DUMP_BASE="${DUMP_ROOT}/${WANDB_PROJECT}/${WANDB_GROUP}"

cd "${BASE_DIR}"

# Build cartesian product of grid options.
combos=("")
for key in "${GRID_KEYS[@]}"; do
    IFS=',' read -ra values <<<"${GRID_VALUES[$key]}"
    new_combos=()
    for combo in "${combos[@]}"; do
        for val in "${values[@]}"; do
            new_combos+=("${combo} --${key}=${val}")
        done
    done
    combos=("${new_combos[@]}")
done

mkdir -p "${DUMP_BASE}"

run_count=0
for combo in "${combos[@]}"; do
    run_count=$((run_count + 1))
    if [[ ${MAX_RUNS} -gt 0 && ${run_count} -gt ${MAX_RUNS} ]]; then
        echo "Reached max runs (${MAX_RUNS}); stopping."
        break
    fi

    read -ra combo_tokens <<<"${combo}"
    args=("${combo_tokens[@]}")

    if [[ -n "${DATASET}" ]]; then
        args+=(--training.dataset="${DATASET}")
    fi
    if [[ -n "${DATASET_PATH}" ]]; then
        args+=(--training.dataset_path="${DATASET_PATH}")
    fi

    # Compose a readable run suffix.
    suffix_parts=()
    for tok in "${combo_tokens[@]}"; do
        clean="${tok#--}"
        clean="${clean//./_}"
        clean="${clean//=/}"
        suffix_parts+=("${clean}")
    done
    RUN_SUFFIX=$(IFS=_; echo "${suffix_parts[*]}")
    RUN_SUFFIX=${RUN_SUFFIX:-"run${run_count}"}

    export WANDB_PROJECT
    export WANDB_GROUP
    export WANDB_RUN_GROUP
    export WANDB_RUN_NAME="${RUN_SUFFIX}"

    dump_folder="${DUMP_BASE}/${WANDB_RUN_NAME}"
    mkdir -p "${dump_folder}"

    echo "============================================================"
    echo "Run ${run_count}/${#combos[@]}: ${WANDB_RUN_NAME}"
    echo "Args: ${args[*]} ${EXTRA_ARGS[*]} --job.config_file=${CONFIG_FILE} --job.description=\"${JOB_DESCRIPTION}\""
    echo "Dump: ${dump_folder}"
    echo "============================================================"

    if [[ ${DRY_RUN} -eq 1 ]]; then
        continue
    fi

    CONFIG_FILE="${CONFIG_FILE}" TRAIN_FILE="${TRAIN_FILE}" "${RUN_SCRIPT}" \
        --job.dump_folder="${dump_folder}" \
        --job.config_file="${CONFIG_FILE}" \
        --job.description="${JOB_DESCRIPTION}" \
        "${args[@]}" \
        "${EXTRA_ARGS[@]}"
done
