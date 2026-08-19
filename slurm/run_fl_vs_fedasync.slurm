#!/bin/bash
# slurm/run_fl_vs_fedasync.slurm — submit examples/fl_vs_fedasync_experiment.py
# (FL sync FedAvg vs FedAsync on ResNet-34 / CIFAR-10, 20 clients, system
# heterogeneity, at TWO data-heterogeneity levels: IID and high non-IID) to a
# GPU node.
#
# Usage:
#   sbatch slurm/run_fl_vs_fedasync.slurm
#
# All four runs (FL/FedAsync × IID/non-IID) run on the GPU automatically (the
# sync/async Simulators, Client training, and evaluation move models/data to
# torch.device("cuda" if available else "cpu")). No code changes needed.
#
# HEADS-UP ON RUNTIME: ResNet-34 @ 64x64 × 20 clients × 300 rounds × H=16 local
# steps, done for four runs, is several GPU-hours. To shrink for a trial run,
# lower NUM_CLIENTS / MAX_GLOBAL_ROUNDS / LOCAL_ITERS at the top of the
# experiment file (or run `python examples/fl_vs_fedasync_experiment.py --smoke`
# locally first).
#
# NOTE: CIFAR-10 auto-downloads on first use (~170 MB) to ~/.flsim/data — make
# sure the compute node has network access, or pre-download it on the login node.
#
# EDIT the two sections marked below for your cluster before submitting:
#   1. #SBATCH directives (partition / gres / account names differ per site)
#   2. Environment setup (module names / conda env / venv path)

#SBATCH --job-name=fl_vs_fedasync_experiment
#SBATCH --gres=gpu:1            # Quadro RTX 6000 node; other node types offer a100 / a30 / L4 / L40S
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=12:00:00
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
# CPU fallback would make ResNet-34 training take days instead of hours, so it's
# much better to stop here than waste the allocation.
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
#
# Compares FL (sync FedAvg) vs FedAsync (polynomial staleness) on ResNet-34 /
# CIFAR-10, 20 clients, per-device system heterogeneity, at two data-
# heterogeneity levels (IID and Dirichlet(alpha=0.1) high non-IID). Compute
# workload C is auto-measured from the model's MACs (system.cycles_per_sample_mode
# = "model_macs", the default), device q_n = 1 — paper-faithful timing.
# Produces: accuracy-vs-round & accuracy-vs-time curves, cumulative energy,
# FedAsync staleness curve, best-accuracy / energy / time bars, and a
# time-to-accuracy table. All knobs (NUM_CLIENTS, MAX_GLOBAL_ROUNDS, LOCAL_ITERS,
# NONIID_ALPHA, heterogeneity ranges) are at the top of
# examples/fl_vs_fedasync_experiment.py — edit there, not here.
# ---------------------------------------------------------------------------
python examples/fl_vs_fedasync_experiment.py

echo "Job finished at $(date)"
