#!/bin/bash
# slurm/run_splitfed.slurm — submit examples/splitfed_experiment.py (Split
# Learning / SplitFed V1 / V2 vs FL vs Normal, Figure-2-style comparison) to a
# GPU node.
#
# Usage:
#   sbatch slurm/run_splitfed.slurm
#
# The split-learning code runs on the GPU automatically: SplitSimulator,
# SplitClient.train_local, the FedAvg aggregation, and the combined-model
# evaluation all move their models/data to
# `torch.device("cuda" if torch.cuda.is_available() else "cpu")` and the
# aggregation tensors inherit that device — verified GPU-safe (no CPU-forcing
# or numpy->tensor creation anywhere in the split path). No code changes are
# needed; this file only needs to get you onto a node that has a GPU.
#
# EDIT the two sections marked below for your cluster before submitting:
#   1. #SBATCH directives (partition / gres / account names differ per site)
#   2. Environment setup (module names / conda env / venv path)

#SBATCH --job-name=splitfed_experiment
#SBATCH --gres=gpu:1            # Quadro RTX 6000 node; other node types offer a100 / a30 / L4 / L40S
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=06:00:00
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

# Assert torch actually sees the GPU, and FAIL LOUDLY if it doesn't — a silent
# CPU fallback would make this (relay-heavy) job take many hours instead of
# minutes, so it's much better to stop here than to waste the allocation.
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
# Compares 5 learning paradigms (Figure-2-style) on MNIST + MnistCNN:
#   Normal (centralized) / FL (FedAvg) / SL / SFLV1 / SFLV2
# The split cut layer is learning.cut_layer in flsim/configs/base.yaml
# (default 6 = MnistCNN's features/classifier boundary); other knobs
# (num_clients=5, global_rounds=30, local_epochs, lr) live at the top of
# examples/splitfed_experiment.py. See that file's docstring for how this
# differs from the paper's exact HAM10000 + ResNet18 Figure 2 setup.
# ---------------------------------------------------------------------------
python examples/splitfed_experiment.py

echo "Job finished at $(date)"
