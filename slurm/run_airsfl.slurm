#!/bin/bash
# slurm/run_airsfl.slurm — submit examples/AirSFL/run_airsfl.py to a GPU node.
#
# AirSFL vs baselines (CIFAR-10, ResNet-18 GroupNorm; two-server draft + roadmap).
# Environment variables (all optional):
#   EXP        one or more of: bench lr main snr nsweep cuts tau nr pathloss   (default "bench")
#   PARTITION  "iid", "dirichlet" or "iid dirichlet"                      (default both)
#   SEEDS      paired seeds, e.g. "11 22 33 44 55"                        (default 11)
#   METHODS    subset, e.g. "airsfl" (to split a sweep over parallel jobs)
#   EPOCHS     global-epoch equivalents per run                           (default 100)
#   LR_EPOCHS  budget of each LR-calibration run                          (default 20)
#   SNRS       SNR list for EXP=snr                                       (default "-20 -10 0 10")
#   RESULTS    results root, e.g. examples/AirSFL/results_quick_aug       (default .../results)
#   TARGETS    validation targets for the figures, e.g. "0.75 0.8"        (default: 95% rule)
#   EXTRA      any other run_airsfl.py flags, e.g. "--early-evals", "--spreads 10 20 30 40",
#              "--path-gain-spread 20", "--augment", "--engine loop", "--deterministic"
#              (defaults: no augmentation, vmap engine, fast cuDNN; a run that hits GPU
#              out-of-memory restarts on the loop engine by itself)
#   PLOT_EXTRA any other plots.py flags, e.g. "--budget-mult 1 3 --sweep-curves epochs uplink"
# Every run is identified by a hash of its full configuration: a finished run with the
# same configuration is skipped (resubmit a timed-out job with the same variables), any
# changed setting produces a new run. Figures/tables are rebuilt at the end of each job.
#
# QUICK GPU TEST (1 seed, 50 epochs, new folder):
#   Q=examples/AirSFL/results_quick_v2
#   J=$(sbatch --parsable --export=ALL,EXP="lr",RESULTS=$Q slurm/run_airsfl.slurm)
#   for P in iid dirichlet; do
#     sbatch --dependency=afterok:$J --export=ALL,EXP="main",PARTITION=$P,EPOCHS=50,RESULTS=$Q,EXTRA="--early-evals" slurm/run_airsfl.slurm
#     for M in airsfl sun_fdma_aircomp aircomp_fl; do
#       sbatch --dependency=afterok:$J --export=ALL,EXP="snr",PARTITION=$P,METHODS=$M,EPOCHS=50,RESULTS=$Q,EXTRA="--early-evals" slurm/run_airsfl.slurm
#       sbatch --dependency=afterok:$J --export=ALL,EXP="pathloss",PARTITION=$P,METHODS=$M,EPOCHS=50,RESULTS=$Q,EXTRA="--early-evals" slurm/run_airsfl.slurm
#     done
#   done
#
# FULL RUNS (5 seeds) -- after the quick test is checked; see README.
#
# EDIT the two sections marked below for your cluster before submitting:
#   1. #SBATCH directives (partition / gres / account names differ per site)
#   2. Environment setup (module names / conda env / venv path)

#SBATCH --job-name=airsfl
#SBATCH --gres=gpu:1            # Quadro RTX 6000 node; other node types offer a100 / a30 / L4 / L40S
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=16:00:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err
# Optional extras — uncomment and edit if your cluster requires/supports them:
# #SBATCH --account=your_account
# #SBATCH --qos=your_qos
# #SBATCH --mail-type=END,FAIL
# #SBATCH --mail-user=you@example.com

set -euo pipefail

# ---------------------------------------------------------------------------
# 1. Environment setup — EDIT for your cluster
# ---------------------------------------------------------------------------
source /global/home/hpc6175/venvs/env1/bin/activate

# Run from the repo root regardless of where sbatch was invoked from.
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
mkdir -p slurm/logs
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

# Workaround: this cluster's ldconfig cache points libcuda.so.1 at
# /usr/lib64/nvidia/libcuda.so.1, which doesn't exist on these GPU nodes —
# the real file lives at /usr/lib64/libcuda.so.1. Without this, torch silently
# falls back to CPU (torch.cuda.is_available() == False) even though the GPU
# and driver are fine.
#
# Only shadow libcuda.so.1, not the whole /usr/lib64 directory: this cluster's
# system binaries (mkdir, etc.) resolve libc.so.6 from a newer CVMFS software
# stack, not from /usr/lib64 — prepending /usr/lib64 wholesale shadows that
# newer libc with the older plain-OS one and breaks every command afterward
# ("mkdir: ... GLIBC_2.34 not found").
CUDA_SHIM_DIR="${SLURM_TMPDIR:-/tmp}/cuda_shim_${SLURM_JOB_ID:-manual}"
mkdir -p "$CUDA_SHIM_DIR"
ln -sf /usr/lib64/libcuda.so.1 "$CUDA_SHIM_DIR/libcuda.so.1"
export LD_LIBRARY_PATH="$CUDA_SHIM_DIR:${LD_LIBRARY_PATH:-}"

# ---------------------------------------------------------------------------
# 2. Record which GPU this job actually landed on.
#    A generic `--gres=gpu:1` request doesn't guarantee a specific model, so
#    log the real hardware (name, memory, driver, compute capability) here
#    for reproducibility — one file per job under slurm/logs/.
# ---------------------------------------------------------------------------
GPU_INFO_FILE="slurm/logs/gpu_${SLURM_JOB_ID:-manual}.txt"
{
    echo "Job ID:  ${SLURM_JOB_ID:-N/A}"
    echo "Node:    $(hostname)"
    echo "Date:    $(date)"
    echo "---- nvidia-smi GPU query ----"
    nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv
} > "$GPU_INFO_FILE" 2>&1 || echo "nvidia-smi not available on this node" > "$GPU_INFO_FILE"

echo "===== GPU info (also saved to $GPU_INFO_FILE) ====="
cat "$GPU_INFO_FILE"
echo "===================================================="

# Fail loudly instead of silently training on CPU for 16 hours.
python -c "
import sys, torch
ok = torch.cuda.is_available()
print('torch:', torch.__version__,
      '| CUDA available:', ok,
      '| device:', torch.cuda.get_device_name(0) if ok else 'cpu (no GPU visible to torch!)')
sys.exit(0 if ok else 1)
"

# ---------------------------------------------------------------------------
# 3. Run the experiment
# ---------------------------------------------------------------------------
EXP="${EXP:-bench}"
PARTITION="${PARTITION:-iid dirichlet}"
SEEDS="${SEEDS:-11}"
ARGS=(--exp ${EXP} --partition ${PARTITION} --seeds ${SEEDS})
[[ -n "${METHODS:-}" ]]   && ARGS+=(--methods ${METHODS})
[[ -n "${EPOCHS:-}" ]]    && ARGS+=(--epochs ${EPOCHS})
[[ -n "${LR_EPOCHS:-}" ]] && ARGS+=(--lr-epochs ${LR_EPOCHS})
[[ -n "${SNRS:-}" ]]      && ARGS+=(--snrs ${SNRS})
[[ -n "${RESULTS:-}" ]]   && ARGS+=(--results ${RESULTS})
[[ -n "${EXTRA:-}" ]]     && ARGS+=(${EXTRA})
PLOT_ARGS=()
[[ -n "${RESULTS:-}" ]]   && PLOT_ARGS+=(--results ${RESULTS})
[[ -n "${TARGETS:-}" ]]   && PLOT_ARGS+=(--targets ${TARGETS})
[[ -n "${PLOT_EXTRA:-}" ]] && PLOT_ARGS+=(${PLOT_EXTRA})
echo "AirSFL: ${ARGS[*]}"
python examples/AirSFL/run_airsfl.py "${ARGS[@]}"
python examples/AirSFL/plots.py "${PLOT_ARGS[@]}"

echo "Job finished at $(date)"
