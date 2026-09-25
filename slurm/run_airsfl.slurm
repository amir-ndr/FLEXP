#!/bin/bash
# slurm/run_airsfl.slurm — submit examples/AirSFL/run_airsfl.py to a GPU node.
#
# AirSFL vs baselines (CIFAR-10, ResNet-18 GroupNorm). Pick the experiment with
# EXP and the data partition(s) with PARTITION (default: both "iid dirichlet"):
#   lr      LR calibration on the noiseless reference -> RUN THIS FIRST
#   main    5 methods x {IID, Dirichlet-0.5} at 20 dB
#   snr     analog methods x {0,10,30} dB
#   nr      AirSFL x Nr {32,48,64,128} at 10 dB (+ digital SFL-V1 reference)
#   nsweep  5 methods x N {20,40} at 20 dB (N=30 reused from main)
#
# Usage (submit lr first; the rest wait for it via --dependency):
#   LR=$(sbatch --parsable --export=ALL,EXP="lr" slurm/run_airsfl.slurm)
#   sbatch --dependency=afterok:$LR --export=ALL,EXP="main",PARTITION="iid"         slurm/run_airsfl.slurm
#   sbatch --dependency=afterok:$LR --export=ALL,EXP="main",PARTITION="dirichlet"   slurm/run_airsfl.slurm
#   sbatch --dependency=afterok:$LR --export=ALL,EXP="snr",PARTITION="iid"          slurm/run_airsfl.slurm
#   sbatch --dependency=afterok:$LR --export=ALL,EXP="snr",PARTITION="dirichlet"    slurm/run_airsfl.slurm
#   sbatch --dependency=afterok:$LR --export=ALL,EXP="nr"                           slurm/run_airsfl.slurm
#   sbatch --dependency=afterok:$LR --export=ALL,EXP="nsweep",PARTITION="iid"       slurm/run_airsfl.slurm
#   sbatch --dependency=afterok:$LR --export=ALL,EXP="nsweep",PARTITION="dirichlet" slurm/run_airsfl.slurm
#
# Every run is resumable: finished runs are skipped, so a job that hits --time
# can simply be resubmitted. Figures/tables are rebuilt at the end of each job
# (examples/AirSFL/results/figures, .../tables).
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
EXP="${EXP:-lr}"
PARTITION="${PARTITION:-iid dirichlet}"
echo "AirSFL: --exp ${EXP} --partition ${PARTITION}"
python examples/AirSFL/run_airsfl.py --exp ${EXP} --partition ${PARTITION}
python examples/AirSFL/plots.py

echo "Job finished at $(date)"
