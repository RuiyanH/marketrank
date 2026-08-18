#!/bin/bash
# Week 5 — train + calibrate the LightGBM ranker on misha.
#
# Submit from the repo root, and make the log directory FIRST:
#     mkdir -p logs && sbatch jobs/train_ranker.sbatch
#
# Prerequisite: week 4's candidate table exists in the warehouse. This job reads
# it; it does not build it. See jobs/build_candidates.sbatch.

#SBATCH --job-name=marketrank-rank
#SBATCH --partition=day
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=400G
#SBATCH --output=logs/%x-%j.out

# --time is mandatory: the `day` partition's DefaultTime is 1 hour and jobs are
# killed at it with no warning. 8h is generous for LightGBM on a downsampled
# candidate table; drop it once you have a measured wall clock, since shorter
# jobs schedule sooner. If it ever needs >24h, move to --partition=week.

set -euo pipefail

source "$HOME/marketrank/env.misha.sh"
cd "$HOME/marketrank"

# Thread count follows the ALLOCATION, not the node. LightGBM and OpenMP both
# default to "every core I can see" — on a shared node that oversubscribes and
# runs slower than single-threaded. Passing it explicitly is the fix.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-64}"

# ---------------------------------------------------------------------------
# The negative sampling rate. Week 4 kept only a fraction of the ~100 negatives
# per positive, so p̂ is inflated and needs Elkan's correction before week 7
# multiplies it by price. Declaring it here means the value lands in the job log
# next to the run that used it — which is the cheapest possible defence against
# the "forgotten w" failure the implementation guide warns about in step 4.2.
# It MUST match what week 4 actually wrote. Do not guess it here.
# ---------------------------------------------------------------------------
NEG_SAMPLE_RATE=0.10

echo "=== marketrank ranker ==============================================="
echo "job            : ${SLURM_JOB_ID:-interactive} on $(hostname)"
echo "cores / mem    : ${SLURM_CPUS_PER_TASK:-?} / ${SLURM_MEM_PER_NODE:-?} MB"
echo "git commit     : $(git rev-parse --short HEAD)"
echo "neg sample rate: ${NEG_SAMPLE_RATE}"
echo "started        : $(date -Is)"
echo "====================================================================="

python -m marketrank.jobs.train_ranker \
    --objective binary \
    --neg-sample-rate "${NEG_SAMPLE_RATE}" \
    --calibration isotonic \
    --threads "${OMP_NUM_THREADS}" \
    --out artifacts/ranker

echo "finished       : $(date -Is)"

# ---------------------------------------------------------------------------
# What the Python entrypoint is expected to do (steps 5.1-5.3):
#
#   1. Spark reads the week-4 candidate table, filters to the `train` slice,
#      and writes it out as Parquet for LightGBM. Spark is only the reader here
#      -- Iceberg is not a format LightGBM speaks. Stop the session before
#      training so the JVM is not holding heap while LightGBM allocates.
#
#   2. Train LightGBM with objective=binary. NOT lambdarank: the decision layer
#      multiplies p̂ by price and treats it as expected revenue, so the score
#      has to be a probability. Report NDCG@12 from the binary model's scores;
#      ranking only needs order. Groups for NDCG are (customer_id, day).
#      Early stopping on `val_tune`.
#
#   3. Apply Elkan's correction with --neg-sample-rate, then fit isotonic
#      regression on `val_calib` -- NOT on val_tune, which early stopping has
#      already seen. Emit ECE before and after, plus the reliability diagram.
#      That plot is gate 2's calibration criterion.
#
#   4. Write to artifacts/ranker/: model.txt, calibrator.pkl, metrics.json,
#      reliability.png. Everything week 6 and week 7 need to load.
#
# Slices come from marketrank.splits -- do not re-derive the dates here, and do
# not let anything touch `ope_env`, which is reserved for week 8's reward model.
# ---------------------------------------------------------------------------
