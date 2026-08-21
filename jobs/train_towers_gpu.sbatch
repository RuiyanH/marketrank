#!/bin/bash
# Step R.4 — train the two towers at full scale on misha, ON A GPU.
#
# The CPU sibling of this script is jobs/train_towers.sbatch. Keep both: the CPU
# one is the fallback when the GPU queue is long, and the comparison of the two
# wall clocks is itself worth recording.
#
# ---------------------------------------------------------------------------
# RECON RESOLVED 2026-08-18 (the 2026-08-14 table in docs/SETUP_MISHA.md listed
# no GPU partition; this is that gap filled).
#
#   sinfo -o "%20P %10G %10m %10c %10l %D" | grep -i gpu
#
#   gpu         gpu:a100:4  gpu:h100:4  gpu:h200:4  gpu:a40:4  gpu:l40s:4
#               30 nodes, 32-48 cores, ~1 TB RAM, MaxTime INFINITE
#   gpu_devel   gpu:a40:4, 3 nodes
#
# GPUs are TYPED, so `--gres=gpu:a100:1` is available -- and deliberately not
# used. This model is 13.5M parameters at d=128; every card on that list is
# oversized for it, so naming one only narrows the queue and delays the start.
# `--gpus=1` takes whichever is free first. If Slurm here is old enough to
# reject `--gpus`, the equivalent is `--gres=gpu:1`.
#
# MaxTime is infinite, so --time is a scheduling choice rather than a cap:
# shorter jobs start sooner. That makes the probe below worth more here, not
# less.
# ---------------------------------------------------------------------------

#SBATCH --job-name=marketrank-towers-gpu
#SBATCH --partition=gpu
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
# THERE IS NO EARLY STOPPING. `train()` runs a fixed epoch count and `best` is
# selected from the history afterwards, so epochs past the peak cost wall clock
# and change nothing -- they do not stop, and they do not corrupt the reported
# number either.
#
# MEASURED (R.4, r4_scale_gpu, 2026-08-19): recall@100 peaks at **epoch 3**
# (6.748%) and falls monotonically to 6.260% by epoch 11 -- a drop of 0.488,
# nearly ten noise spreads -- while training loss improves every single epoch
# (6.973 -> 6.768). That run spent 9 of its 12 epochs making retrieval worse.
#
# 5 leaves margin over the observed peak. Raise it only with a curve that says
# so, and read the curve rather than the loss: this build has now been misled by
# loss/recall anti-correlation three times (R.2's volume leak, R.3's wider
# denominator, R.4's over-training).
EPOCHS=5
BATCH=4096            # larger than the laptop's 1024: in-batch negatives scale
                      # with batch size, so a bigger batch is a better sampled
                      # softmax as well as better GPU utilisation. It also
                      # changes the negative distribution -- report it.
LR=3e-3
TEMPERATURE=0.07
N_UNIFORM=16          # R.3's nominal best -- AND ITS DELTA IS INSIDE THE NOISE
                      # FLOOR. Measured on the laptop: +0.051 recall@100 against
                      # n_uniform=0, where 3 seeds of one config spread 0.051.
                      # Carried under "keep the best; record all", because the
                      # cost here is negligible and the tail-inflation mechanism
                      # has more room on the full catalog than on the cohort.
                      # DO NOT attribute any R.4 gain to this value: R.3 is a
                      # null, and a second full-scale run to ablate it is
                      # explicitly not worth its cost. See BUILD_NOTES R.3.
RECENCY_HALF_LIFE=30  # days -- R.2's measured winner (+0.873 recall@100)
LABEL="r4_scale_gpu"

# NO --article-volume. R.2 measured it as actively harmful: -0.462 recall@100
# alone, -1.715 on top of recency, and it produced the LOWEST training loss of
# the four arms while giving the WORST retrieval. See BUILD_NOTES step R.2 for
# the day-fingerprint mechanism. If you re-enable it, you are re-enabling a
# measured leak.

echo "=== marketrank two-tower (GPU) ======================================"
echo "job              : ${SLURM_JOB_ID:-interactive} on $(hostname)"
echo "gpu              : ${SLURM_JOB_GPUS:-${SLURM_STEP_GPUS:-unset}}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
echo "cores / mem      : ${SLURM_CPUS_PER_TASK:-?} / ${SLURM_MEM_PER_NODE:-?} MB"
echo "git commit       : $(git rev-parse --short HEAD)"
# Tracked-file status, so the sha above is self-adjudicating: `none` means
# this log reproduces from that commit, anything else names what differed.
# --untracked-files=no deliberately -- output under artifacts/ is not code
# drift, and counting it made metrics report dirty=True on every run
# (r4_scale_gpu did, from artifact files alone). `|| true` because
# `set -e` would otherwise abort the header on a git hiccup.
_tracked="$(git status --porcelain --untracked-files=no | tr '\n' ';' | sed 's/;$//' || true)"
echo "tracked mods     : ${_tracked:-none}"
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
