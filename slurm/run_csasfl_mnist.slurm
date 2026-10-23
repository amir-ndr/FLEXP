#!/bin/bash
# slurm/run_csasfl_mnist.slurm — "mnist_exp": the CSA-SFL paper's simulation setup on a GPU node
# (examples/CSASFL/run_mnist_exp.py, then examples/CSASFL/plots.py).
#
# CSA-SFL vs FL / SFLv2 / SAFSL / ParallelSFL on MNIST + 12-layer LeNet (split 8), K = 50 devices,
# s_k ~ LogNormal(0, 2^2), t_0 = 0.2 s, E = 5, B = 64, eta = 0.01, N = 7, H = 20. Every method trains
# until its simulated clock reaches the same TIME_BUDGET (default 50,000 s ~ 2000 CSA-SFL rounds,
# ~1000 SAFSL / ~2000 ParallelSFL / ~30 FL-SFLv2 rounds, median over seeds); ROUNDS caps own rounds.
# Experiments (EXP, any subset):
#   main      5 methods x delta {0.1, 10}                       (comparison)
#   nsweep    CSA-SFL x N {3,5,7,9,11}, delta 0.1, H 20         (paper Fig. a/b)
#   hsweep    CSA-SFL x H {10,20,30,40,50}, delta 0.1, N 7      (paper Fig. c)
#   ablation  CSA-SFL full / uniform phi / random clustering, delta 0.1
#   variants  CSA-SFL improvement screen: paper / +system-aware clustering / +time slot /
#             +update-based aggregation (sweeps staleness) / +lazy re-clustering / all four / all but (1)
#             (EXTRA: --variants ..., --variant-deltas 0.1 10, --sys-lambda 1, --deadline-q 0.5)
#   plots     only rebuild the figures/tables from the finished runs
# Environment variables (all optional):
#   EXP          default "main"  (sweeps/ablation: EXP="nsweep hsweep ablation")
#   SEEDS        paired seeds, default "1 2 3 4 5" (the paper averages 20 runs: SEEDS="$(seq -s ' ' 1 20)")
#   METHODS      subset of "fl sflv2 safsl parallelsfl csasfl" for EXP=main
#   DELTAS       default "0.1 10"
#   TIME_BUDGET  simulated seconds every method trains for, default 50000 (0 = exactly ROUNDS rounds)
#   ROUNDS       safety cap on each method's own global rounds, default 50000 (time-stopped runs are reused)
#   RESULTS    results root, default examples/CSASFL/results_mnist_exp
#   WORKERS    concurrent run processes sharing this job's GPU (default 3); each takes a
#              cost-balanced slice of the run list (MNIST/LeNet groups do not fill a GPU alone)
#   EXTRA      other run_mnist_exp.py flags, e.g. "--tf32" (TF32 tensor cores: ~2-3x faster
#              convolutions on A100/A30, default exact FP32), "--vmap-chunk 16" (less GPU memory)
#   PLOT_EXTRA other plots.py flags, e.g. "--targets 0.9 0.95 0.97 --tmax 20000"
# GPU use: MNIST lives on the GPU, every group of devices trains in lock-step through one batched
# vmap kernel, cuDNN autotuning on; a run out of GPU memory restarts with chunked vmap, then the
# loop engine. Every run is identified by a hash of its configuration: finished runs are skipped
# (resubmit a timed-out job with the same variables), shared runs are trained once.
# Job arrays: the run list is split over (array tasks x WORKERS) disjoint slices.
#
# COST scales with TIME_BUDGET: the async/clustered methods keep training fast devices for the whole
# simulated window. Per run at 50,000 s (seeds 1-5): ParallelSFL ~4,200, SAFSL ~3,650, CSA-SFL ~1,400,
# FL/SFLv2 ~400 MNIST-epochs of local training (346k / 215k / 347k / 19k batched GPU steps); seeds 1-5 of
# main = ~100,000 MNIST-epochs (~12-16 h on 4 Quadro RTX 6000). 20,000 s needs 2.5x less, 10,000 s 5x less.
# All methods but FL reach 97% well before 20,000 s. A finished run with a LONGER budget covers a
# shorter one (skipped, no training; plots.py cuts every run at the common budget), so lowering
# TIME_BUDGET later never retrains. --tf32 needs an Ampere+ GPU (A30/A100/L40S), not the RTX 6000.
#
# MAIN, FIRST SEED (both deltas, 2 GPUs x 3 processes), then the figures:
#   J=$(sbatch --parsable --array=0-1 --export=ALL,SEEDS=1 slurm/run_csasfl_mnist.slurm)
#   sbatch --dependency=afterany:$J --export=ALL,EXP=plots --time=00:30:00 slurm/run_csasfl_mnist.slurm
# CSA-SFL IMPROVEMENT SCREEN (seeds 1-2, both deltas, 10,000 s; the paper variant and the baselines are
# reused from the finished main runs). An efficient CSA-SFL keeps devices busy, so it trains ~5x more per
# simulated second than the paper version (time-slot variants ~1,900 MNIST-epochs per run at 10,000 s):
#   J=$(sbatch --parsable --array=0-3 --export=ALL,EXP=variants,TIME_BUDGET=10000,SEEDS="1 2",EXTRA="--variant-deltas 0.1 10" slurm/run_csasfl_mnist.slurm)
#   sbatch --dependency=afterany:$J --export=ALL,EXP=plots,PLOT_EXTRA="--time-budget 10000" --time=00:30:00 slurm/run_csasfl_mnist.slurm
# SWEEPS + ABLATION OF THE TWO CHOSEN VARIANTS, in a NEW results folder (delta 0.1, 5,000 s):
#   slot_batch_lazy_q25 (time slot q=0.25 + batched arrivals + lazy re-clustering) and sys (system-aware clustering,
#   lambda=1): N sweep {3,5,7,9,11} (H=20), H sweep {10,20,30,40,50} (N=7), and each variant's ablation (paper reference,
#   each improvement removed, uniform phi, random clustering; sys also: speed-only clusters). 27 runs per seed.
#   R=examples/CSASFL/results_sweeps_ablation
#   J=$(sbatch --parsable --array=0-7 --export=ALL,RESULTS=$R,EXP="nsweep hsweep ablation",TIME_BUDGET=5000,SEEDS="1 2 3",EXTRA="--csa-variant slot_batch_lazy_q25 sys" slurm/run_csasfl_mnist.slurm)
#   sbatch --dependency=afterany:$J --export=ALL,RESULTS=$R,EXP=plots,PLOT_EXTRA="--csa-variant slot_batch_lazy_q25 sys --targets 0.9 0.95 0.97 0.98" --time=00:30:00 slurm/run_csasfl_mnist.slurm
#   (more seeds later: the same lines with SEEDS="4 5" -- finished runs are skipped)
# TIME-VARYING DEVICE SPEEDS (log s_k AR(1), rho = 0.9 per 100 s: LogNormal(0, 2^2) at every instant, correlation
# time 1000 s; identical drift for every method of a seed), NEW folder, delta 0.1, 5,000 s, seeds 1-2. H sweep {10..50}
# + ablation (full / w/o weighted aggregation / random clustering) of sys and sys_lazy -- 13 runs per seed, 26 in all:
#   R=examples/CSASFL/results_drift
#   D="--speed-drift-rho 0.9 --speed-drift-dt 100"
#   J=$(sbatch --parsable --array=0-5 --export=ALL,RESULTS=$R,EXP="hsweep ablation",TIME_BUDGET=5000,SEEDS="1 2",EXTRA="--csa-variant sys sys_lazy --ablation-parts full uniform random $D" slurm/run_csasfl_mnist.slurm)
#   sbatch --dependency=afterany:$J --export=ALL,RESULTS=$R,EXP=plots,PLOT_EXTRA="--csa-variant sys sys_lazy --targets 0.9 0.95 0.97 0.98" --time=00:30:00 slurm/run_csasfl_mnist.slurm
#   (more seeds later: the same lines with SEEDS="3" -- finished runs are skipped)
# FASTER SPEED FLUCTUATION for the H sweep (rho = 0.9 per 20 s, correlation time 200 s: at H = 50 the clustering lives
# ~one correlation time, at H = 10 the recollection halts dominate), sys only, seeds 1-4, NEW folder -- 28 runs:
#   R=examples/CSASFL/results_drift_tc200
#   D="--speed-drift-rho 0.9 --speed-drift-dt 20"
#   J=$(sbatch --parsable --array=0-5 --export=ALL,RESULTS=$R,EXP="hsweep ablation",TIME_BUDGET=5000,SEEDS="1 2 3 4",EXTRA="--csa-variant sys --ablation-parts full uniform random $D" slurm/run_csasfl_mnist.slurm)
#   sbatch --dependency=afterany:$J --export=ALL,RESULTS=$R,EXP=plots,PLOT_EXTRA="--csa-variant sys --targets 0.9 0.95 0.97 0.98" --time=00:30:00 slurm/run_csasfl_mnist.slurm
# IMPROVED CSA-SFL (winner of the screens: slot q=0.25 + batched arrivals + lazy re-clustering). Set RESULTS to
# the ONE folder that holds all finished runs (finished runs are reused, the figures compare against them):
#   A) main comparison, seeds 3-5 (seeds 1-2 exist), both deltas, 10,000 s -- 6 runs:
#   J1=$(sbatch --parsable --array=0-1 --export=ALL,RESULTS=examples/CSASFL/results_mnist_exp,EXP=variants,TIME_BUDGET=10000,SEEDS="3 4 5",EXTRA="--variants slot_batch_lazy_q25 --variant-deltas 0.1 10" slurm/run_csasfl_mnist.slurm)
#   B) N sweep + H sweep + ablation of the winner, delta 0.1, seeds 1-3, 5,000 s -- 38 new runs:
#   J2=$(sbatch --parsable --array=0-5 --export=ALL,RESULTS=examples/CSASFL/results_mnist_exp,EXP="nsweep hsweep ablation",TIME_BUDGET=5000,SEEDS="1 2 3",EXTRA="--csa-variant slot_batch_lazy_q25" slurm/run_csasfl_mnist.slurm)
#   figures (CSA-SFL = the winner everywhere; staleness figures for the N and H sweeps):
#   sbatch --dependency=afterany:$J1:$J2 --export=ALL,RESULTS=examples/CSASFL/results_mnist_exp,EXP=plots,PLOT_EXTRA="--time-budget 5000 --csa-variant slot_batch_lazy_q25 --targets 0.9 0.95 0.97 0.98" --time=00:30:00 slurm/run_csasfl_mnist.slurm
# SECOND SCREEN (sys+lazy / slot+batched arrivals / +lazy / +lazy with a shorter slot; seeds 1-2, both deltas,
# 10,000 s). Put all finished runs in ONE folder first so the figures compare against the baselines:
#   cp -n examples/CSASFL/results_mnist_exp_old/runs/* examples/CSASFL/results_mnist_exp/runs/
#   J=$(sbatch --parsable --array=0-3 --export=ALL,EXP=variants,TIME_BUDGET=10000,SEEDS="1 2",EXTRA="--variants screen2 --variant-deltas 0.1 10" slurm/run_csasfl_mnist.slurm)
#   sbatch --dependency=afterany:$J --export=ALL,EXP=plots,PLOT_EXTRA="--time-budget 10000 --targets 0.9 0.95 0.97 0.98" --time=00:30:00 slurm/run_csasfl_mnist.slurm
# MAIN, SEEDS 1-5 (the default; finished runs are skipped, resubmit the same line after a time-out):
#   J=$(sbatch --parsable --array=0-3 slurm/run_csasfl_mnist.slurm)
#   sbatch --dependency=afterany:$J --export=ALL,EXP=plots --time=00:30:00 slurm/run_csasfl_mnist.slurm
#
# EDIT the two sections marked below for your cluster before submitting:
#   1. #SBATCH directives (partition / gres / account names differ per site)
#   2. Environment setup (module names / conda env / venv path)

#SBATCH --job-name=csasfl_mnist
#SBATCH --gres=gpu:1            # Quadro RTX 6000 node; other node types offer a100 / a30 / L4 / L40S
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH --output=slurm/logs/%x_%A_%a.out
#SBATCH --error=slurm/logs/%x_%A_%a.err
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
# the real file lives at /usr/lib64/libcuda.so.1. Only shadow libcuda.so.1
# (prepending /usr/lib64 wholesale breaks the CVMFS libc of system binaries).
CUDA_SHIM_DIR="${SLURM_TMPDIR:-/tmp}/cuda_shim_${SLURM_JOB_ID:-manual}"
mkdir -p "$CUDA_SHIM_DIR"
ln -sf /usr/lib64/libcuda.so.1 "$CUDA_SHIM_DIR/libcuda.so.1"
export LD_LIBRARY_PATH="$CUDA_SHIM_DIR:${LD_LIBRARY_PATH:-}"

RESULTS="${RESULTS:-examples/CSASFL/results_mnist_exp}"
PLOT_ARGS=(--results "${RESULTS}")
[[ -n "${PLOT_EXTRA:-}" ]] && PLOT_ARGS+=(${PLOT_EXTRA})
EXP="${EXP:-main}"

if [[ "${EXP}" == "plots" ]]; then
    python examples/CSASFL/plots.py "${PLOT_ARGS[@]}"
    echo "Job finished at $(date)"
    exit 0
fi

# ---------------------------------------------------------------------------
# 2. Record which GPU this job landed on (one file per job) and fail loudly if
#    torch cannot see it (a silent CPU fallback would take days).
# ---------------------------------------------------------------------------
GPU_INFO_FILE="slurm/logs/gpu_${SLURM_JOB_ID:-manual}.txt"
{
    echo "Job ID:  ${SLURM_JOB_ID:-N/A} (array task ${SLURM_ARRAY_TASK_ID:-none})"
    echo "Node:    $(hostname)"
    echo "Date:    $(date)"
    echo "---- nvidia-smi GPU query ----"
    nvidia-smi --query-gpu=name,memory.total,memory.used,driver_version,compute_cap --format=csv
} > "$GPU_INFO_FILE" 2>&1 || echo "nvidia-smi not available on this node" > "$GPU_INFO_FILE"
echo "===== GPU info (also saved to $GPU_INFO_FILE) ====="
cat "$GPU_INFO_FILE"
echo "===================================================="

python -c "
import sys, torch
ok = torch.cuda.is_available()
print('torch:', torch.__version__, '| CUDA available:', ok,
      '| device:', torch.cuda.get_device_name(0) if ok else 'cpu (no GPU visible to torch!)',
      '| torch.func (vmap engine):', hasattr(torch, 'func'))
sys.exit(0 if ok else 1)
"

# ---------------------------------------------------------------------------
# 3. Run: WORKERS processes on this GPU, each a disjoint slice of the run list
# ---------------------------------------------------------------------------
SEEDS="${SEEDS:-1 2 3 4 5}"
DELTAS="${DELTAS:-0.1 10}"
WORKERS="${WORKERS:-3}"
ARGS=(--exp ${EXP} --seeds ${SEEDS} --deltas ${DELTAS} --results "${RESULTS}")
[[ -n "${METHODS:-}" ]] && ARGS+=(--methods ${METHODS})
[[ -n "${ROUNDS:-}" ]]  && ARGS+=(--rounds ${ROUNDS})
[[ -n "${TIME_BUDGET:-}" ]] && ARGS+=(--time-budget ${TIME_BUDGET})
[[ -n "${EXTRA:-}" ]]   && ARGS+=(${EXTRA})

if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    TASK=$(( SLURM_ARRAY_TASK_ID - ${SLURM_ARRAY_TASK_MIN:-0} ))
    NTASK=$(( ${SLURM_ARRAY_TASK_MAX:-0} - ${SLURM_ARRAY_TASK_MIN:-0} + 1 ))
else
    TASK=0; NTASK=1
fi
NSHARD=$(( NTASK * WORKERS ))
echo "mnist_exp: ${ARGS[*]} | array task ${TASK}/${NTASK} x ${WORKERS} workers = shards of ${NSHARD}"
python examples/CSASFL/run_mnist_exp.py "${ARGS[@]}" --list --shard "$(( TASK * WORKERS ))/${NSHARD}" | head -3 || true

PIDS=()
for (( w = 0; w < WORKERS; w++ )); do
    SH=$(( TASK * WORKERS + w ))
    LOG="slurm/logs/csasfl_mnist_${SLURM_JOB_ID:-manual}_shard${SH}of${NSHARD}.log"
    OMP_NUM_THREADS=2 python examples/CSASFL/run_mnist_exp.py "${ARGS[@]}" --shard "${SH}/${NSHARD}" > "$LOG" 2>&1 &
    PIDS+=($!)
    echo "  shard ${SH}/${NSHARD} -> pid $! (log $LOG)"
done
FAIL=0
for pid in "${PIDS[@]}"; do
    wait "$pid" || FAIL=1
done
nvidia-smi --query-gpu=memory.used,memory.total --format=csv || true
[[ $FAIL -ne 0 ]] && echo "WARNING: at least one shard failed (see the shard logs); figures use the finished runs"

python examples/CSASFL/plots.py "${PLOT_ARGS[@]}" || echo "plots skipped (no complete runs yet?)"
echo "Job finished at $(date)"
exit $FAIL
