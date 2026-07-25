#!/bin/bash
# slurm/run_parallelsfl.slurm — submit examples/parallelsfl_experiment.py
# (ParallelSFL vs FL / SFLv2 / SAFSL on MNIST + MnistCNN, 50 clients, data +
# system heterogeneity) to a GPU node.
#
# Usage:
#   sbatch slurm/run_parallelsfl.slurm
#
# All four paradigms run on the GPU automatically (ParallelSFLSimulator,
# SplitSimulator, the sync/async Simulators, SplitClient relay, evaluation move
# models/data to torch.device("cuda" if available else "cpu"); aggregation
# tensors inherit that device — GPU-safe). No code changes needed.
#
# HEADS-UP ON RUNTIME: 50 workers x split-relay training over MAX_GLOBAL_ROUNDS
# (150) rounds x four methods is many GPU-hours. To shrink for a trial run,
# lower NUM_CLIENTS / MAX_GLOBAL_ROUNDS at the top of the experiment file.
#
# EDIT the two sections marked below for your cluster before submitting:
#   1. #SBATCH directives (partition / gres / account names differ per site)
#   2. Environment setup (module names / conda env / venv path)

#SBATCH --job-name=parallelsfl_experiment
#SBATCH --gres=gpu:1            # Quadro RTX 6000 node; other node types offer a100 / a30 / L4 / L40S
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
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
# 2. Record which GPU this job actually landed on (one file per job).
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

# Assert torch actually sees the GPU, and FAIL LOUDLY if it doesn't — a silent
# CPU fallback would make this (relay-heavy, 50-worker) job take days instead of
# hours, so it's much better to stop here than waste the allocation.
python3 -c "
import sys, torch
ok = torch.cuda.is_available()
print('torch:', torch.__version__,
      '| CUDA available:', ok,
      '| device:', torch.cuda.get_device_name(0) if ok else 'cpu (no GPU visible to torch!)')
sys.exit(0 if ok else 1)
"

# ---------------------------------------------------------------------------
# 3. Run the experiment
#
# Compares four paradigms on MNIST / MnistCNN, 50 clients, Dirichlet(alpha=0.3)
# data heterogeneity + per-device system heterogeneity, all on ONE physical
# base with IDENTICAL max local work (H = ParallelSFL tau_max):
#   ParallelSFL / FL / SFLv2 / SAFSL
# Produces: accuracy-vs-time & accuracy-vs-round curves, energy(J)- and
# overhead(MB)-to-accuracy bars, an average-waiting-time curve, and a
# time-to-accuracy table. All knobs (NUM_CLIENTS, DIRICHLET_ALPHA, LOCAL_ITERS,
# heterogeneity ranges, cut layer) are at the top of
# examples/parallelsfl_experiment.py — edit there, not here.
# ---------------------------------------------------------------------------
python examples/parallelsfl_experiment.py

echo "Job finished at $(date)"
