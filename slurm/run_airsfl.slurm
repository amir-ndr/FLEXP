#!/bin/bash
# slurm/run_airsfl.slurm — AirSFL vs baselines (CIFAR-10, ResNet-18 GroupNorm).
#
# Experiments (examples/AirSFL/run_airsfl.py --exp ...), all resumable (finished
# runs are skipped, so a timed-out job can simply be resubmitted):
#   lr    LR calibration on the noiseless reference, grid {0.01,0.03,0.1} (~20 min)
#         -> RUN THIS FIRST: every other experiment uses the chosen LR.
#   main  5 methods x {IID, Dirichlet-0.5} at 20 dB   (~10 runs, ~25-30 min each)
#   snr   3 analog methods x {0,10,30} dB x partitions (~18 runs)
#   nr    AirSFL x Nr {32,48,64,128} at 10 dB (+ digital SFL-V1 reference)
#   nsweep 5 methods x N {20,40} at 20 dB (N=30 reused from main) (~10 runs/partition)
# Then build figures/tables:  python examples/AirSFL/plots.py
#
# Usage (one experiment per job is recommended):
#   sbatch --export=ALL,EXP="lr"   slurm/run_airsfl.slurm
#   sbatch --export=ALL,EXP="main",PARTITION="iid"       slurm/run_airsfl.slurm
#   sbatch --export=ALL,EXP="main",PARTITION="dirichlet" slurm/run_airsfl.slurm
#   sbatch --export=ALL,EXP="snr",PARTITION="iid"       slurm/run_airsfl.slurm
#   sbatch --export=ALL,EXP="snr",PARTITION="dirichlet" slurm/run_airsfl.slurm
#   sbatch --export=ALL,EXP="nr"   slurm/run_airsfl.slurm
#   sbatch --export=ALL,EXP="nsweep",PARTITION="iid"       slurm/run_airsfl.slurm
#   sbatch --export=ALL,EXP="nsweep",PARTITION="dirichlet" slurm/run_airsfl.slurm
# Estimated on one modern GPU: ~30-35 min per 50-epoch run (N=30, tau=5, B=16).
# A job that hits --time can simply be resubmitted: finished runs are skipped.
#
# EDIT the two sections marked below for your cluster before submitting.

#SBATCH --job-name=airsfl
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err

set -euo pipefail

# ---------------------------------------------------------------------------
# 1. Environment setup — EDIT for your cluster
# ---------------------------------------------------------------------------
source /global/home/hpc6175/venvs/env1/bin/activate

cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
mkdir -p slurm/logs
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

# libcuda shim (same workaround as the other jobs on this cluster)
CUDA_SHIM_DIR="${SLURM_TMPDIR:-/tmp}/cuda_shim_${SLURM_JOB_ID:-manual}"
mkdir -p "$CUDA_SHIM_DIR"
ln -sf /usr/lib64/libcuda.so.1 "$CUDA_SHIM_DIR/libcuda.so.1"
export LD_LIBRARY_PATH="$CUDA_SHIM_DIR:${LD_LIBRARY_PATH:-}"

# ---------------------------------------------------------------------------
# 2. Record the GPU and fail loudly if torch cannot see it
# ---------------------------------------------------------------------------
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || true
python -c "
import sys, torch
ok = torch.cuda.is_available()
print('torch:', torch.__version__, '| CUDA:', ok, '|', torch.cuda.get_device_name(0) if ok else 'cpu')
sys.exit(0 if ok else 1)
"

# ---------------------------------------------------------------------------
# 3. Run
# ---------------------------------------------------------------------------
EXP="${EXP:-lr}"
PARTITION="${PARTITION:-iid dirichlet}"
echo "AirSFL: --exp ${EXP} --partition ${PARTITION}"
python examples/AirSFL/run_airsfl.py --exp ${EXP} --partition ${PARTITION}
python examples/AirSFL/plots.py

echo "Job finished at $(date)"
