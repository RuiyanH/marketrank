#!/bin/bash
# Step R.4 — train the two towers at full scale on misha, ON A GPU.
#
# The CPU sibling of this script is jobs/train_towers.sbatch. Keep both: the CPU
# one is the fallback when the GPU queue is long, and the comparison of the two
# wall clocks is itself worth recording.
#
# ---------------------------------------------------------------------------
# RUN THIS RECON FIRST. The two directives below are the only thing in this file
# that is guessed, because the 2026-08-14 recon table in docs/SETUP_MISHA.md
# lists day / week / devel / bigmem and NO GPU partition. Resolve them, edit
# them in, and note them in the recon table so the next person does not repeat
# this:
#
#     sinfo -o "%20P %10G %10m %10c %10l %D" | grep -i gpu
#       -> partition name, the gres string, mem, cores, MaxTime, node count
#
#     scontrol show partition <NAME> | grep -E "MaxTime|DefaultTime|MaxNodes"
#       -> the time limits, which differ from the CPU partitions
#
# Slurm accepts either `--gpus=1` (newer) or `--gres=gpu:1` (older); the `%10G`
# column above tells you which vocabulary this cluster speaks, and whether the
# GPUs are typed (`gpu:a100:1`). Ask for ONE. This model is 13.5M parameters at
# d=128 and does not shard.
# ---------------------------------------------------------------------------

#SBATCH --job-name=marketrank-towers-gpu
#SBATCH --partition=REPLACE_FROM_RECON
#SBATCH --gpus=1
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/%x-%j.out

# --time is mandatory; CPU partitions here default to 1h and kill at it, and GPU
# partitions are usually stricter, not looser. 4h is a placeholder chosen to be
# generous against an UNMEASURED wall clock — replace it with ~1.5x the probe
# below once you have one. Shorter jobs schedule sooner, and on a GPU queue that
# matters much more than it does on `day`.
#
# --cpus-per-task=8, not 64: on a GPU run the CPU only feeds the device. Asking
# for a whole node's cores makes the job wait for a whole node's cores.
#
# --mem=64G covers the exported arrays held in host RAM at full scale (~28M
# positives x recent-article lists + numerics, tens of GB). It is host memory,
# unrelated to the GPU's own.

set -euo pipefail

source "$HOME/marketrank/env.misha.sh"
cd "$HOME/marketrank"
mkdir -p logs

# Feeds the dataloader side only; the matmuls are on the device.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

# ---------------------------------------------------------------------------
# What this rung changes against the reduced laptop configuration. Everything
# else is held at the value the ablation ladder established, so this run
# measures SCALE and nothing else.
#
#   training positives   2,900,248  ->  the full `train` slice (~28M)
#   customer cohort        300,000  ->  all 1,371,980
#   embedding dim               64  ->  128
#   training window   2020-02-14..  ->  2018-09-20 .. 2020-08-11
#
# N_UNIFORM and RECENCY_HALF_LIFE MUST carry the values the ladder chose, not
# fresh guesses -- otherwise this run confounds scale with a hyperparameter
# change and stops being a rung.
# ---------------------------------------------------------------------------
DIM=128
EPOCHS=12
BATCH=4096            # larger than the laptop's 1024: in-batch negatives scale
                      # with batch size, so a bigger batch is a better sampled
                      # softmax as well as better GPU utilisation. It also
                      # changes the negative distribution -- report it.
LR=3e-3
TEMPERATURE=0.07
N_UNIFORM=64          # R.3's chosen value -- CONFIRM against the ladder
RECENCY_HALF_LIFE=30  # days -- R.2's measured winner (+0.873 recall@100)
LABEL="r4_scale_gpu"

echo "=== marketrank two-tower (GPU) ======================================"
echo "job              : ${SLURM_JOB_ID:-interactive} on $(hostname)"
echo "gpu              : ${SLURM_JOB_GPUS:-${SLURM_STEP_GPUS:-unset}}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
echo "cores / mem      : ${SLURM_CPUS_PER_TASK:-?} / ${SLURM_MEM_PER_NODE:-?} MB"
echo "git commit       : $(git rev-parse --short HEAD)"
echo "dim / epochs     : ${DIM} / ${EPOCHS}   batch ${BATCH}"
echo "n_uniform        : ${N_UNIFORM}"
echo "recency half-life: ${RECENCY_HALF_LIFE}"
echo "started          : $(date -Is)"
echo "====================================================================="

# Fail early and loudly rather than silently training on CPU for four hours.
# torch's device pick order is MPS -> CUDA -> CPU; on this node only CUDA can
# win, so if it does not, the environment is wrong (CPU-only wheel is the usual
# cause -- see the install note at the bottom).
python - <<'PY'
import sys, torch
print("torch", torch.__version__, "| cuda build", torch.version.cuda)
if not torch.cuda.is_available():
    sys.exit("CUDA not available to torch -- refusing to run a GPU job on CPU. "
             "Most likely a CPU-only pyspark-era wheel; see the install note in "
             "this script.")
print("device:", torch.cuda.get_device_name(0))
PY

# ---------------------------------------------------------------------------
# VERIFY — the wall clock at this scale is UNMEASURED on any GPU.
#
# The laptop reference is Apple MPS at reduced scale: ~90-110 s/epoch on 2.9M
# positives at d=64. This run is ~10x the positives and 2x the dim on different
# silicon, and I will not assert a scaling factor. Measure ONE epoch first:
#
#     sbatch --time=00:45:00 --export=ALL,PROBE=1 jobs/train_towers_gpu.sbatch
#
# Read `seconds` from the epoch line, multiply by EPOCHS, add ~20% and set
# --time from that. The probe exists to produce a wall clock, not a model.
# ---------------------------------------------------------------------------
if [[ "${PROBE:-0}" == "1" ]]; then
    echo "PROBE RUN — one epoch, for wall clock only"
    EPOCHS=1
    LABEL="r4_probe_gpu"
fi

python -m marketrank.jobs.train_towers \
    --label "${LABEL}" \
    --dim "${DIM}" \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH}" \
    --lr "${LR}" \
    --temperature "${TEMPERATURE}" \
    --n-uniform "${N_UNIFORM}" \
    --recency-half-life "${RECENCY_HALF_LIFE}" \
    --article-volume \
    --threads "${OMP_NUM_THREADS}"

echo "finished         : $(date -Is)"

# ---------------------------------------------------------------------------
# INSTALL NOTE — the one thing that will bite.
#
# docs/SETUP_MISHA.md Phase 1 installs `pyspark==3.5.9` and the project, and
# nothing there pulls a CUDA-enabled torch. A default `pip install torch` on a
# login node can resolve to a CPU-only wheel, which fails the guard above rather
# than training silently on CPU -- deliberately. Install torch to match the
# cluster's CUDA:
#
#     module avail CUDA                 # find the toolkit the nodes carry
#     pip install torch --index-url https://download.pytorch.org/whl/cu121
#                                       # match cu### to the module
#
# Do this on a node that can see the GPU, or at least verify with the guard
# above in an interactive session before queueing a long job.
#
# NOTE ALSO: the eval scores a (20,000 x 105,542) matrix in chunks of 512
# customers. On the GPU that is ~216 MB per chunk in fp32 -- comfortable, but it
# is why chunking exists; do not raise it to "use the memory".
# ---------------------------------------------------------------------------
