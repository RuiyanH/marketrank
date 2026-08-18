#!/bin/bash
# The CPU prerequisite for GPU training: rebuild week-2 features with the spine,
# then export the two-tower dataset at FULL scale.
#
#     mkdir -p logs && sbatch jobs/build_export.sbatch
#
# Both stages are Spark. They are in one job because they are sequential and
# share a warehouse, and because two queue waits are worse than one. Training is
# a SEPARATE job (jobs/train_towers_gpu.sbatch) for a real reason: holding JVM
# heap while torch allocates is how a large allocation becomes an OOM.
#
# Prerequisite: docs/SETUP_MISHA.md Phases 1-5 have been run, so the warehouse
# holds all 734 days of local.raw.transactions.

#SBATCH --job-name=marketrank-build-export
#SBATCH --partition=day
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=400G
#SBATCH --output=logs/%x-%j.out

# --time is mandatory; `day` defaults to 1h and kills at it. 8h is generous
# against laptop measurements that do not transfer: the reduced-cohort feature
# rebuild was 318.8 s and the reduced export 230.0 s on 8 cores, but this runs
# the FULL cohort (1,371,980 customers, ~28M positives vs 2.9M). Replace with
# ~1.5x the first measured run.

set -euo pipefail

source "$HOME/marketrank/env.misha.sh"
cd "$HOME/marketrank"
mkdir -p logs

echo "=== marketrank build + export ======================================="
echo "job         : ${SLURM_JOB_ID:-interactive} on $(hostname)"
echo "cores / mem : ${SLURM_CPUS_PER_TASK:-?} / ${SLURM_MEM_PER_NODE:-?} MB"
echo "git commit  : $(git rev-parse --short HEAD)"
echo "started     : $(date -Is)"
echo "====================================================================="

# --- Stage 1: week-2 features, with the R.1 spine ---------------------------
# The spine covers every customer and every article on the SCORING days, unioned
# with natural transaction-day rows. Not a dense customer x calendar spine --
# that is 1,371,980 x 734 = 1.006 billion rows before a feature is attached, and
# it is never the right object at any scale.
echo ">>> stage 1: rebuild_features"
python -m marketrank.jobs.rebuild_features

# --- Stage 2: the two-tower export, full cohort -----------------------------
# COHORT 0 means "no cap". Read that literally: `training_events` guards on
# `is not None`, so before the CLI learned to map <=0 to None, passing 0 here
# produced .limit(0) -- an EMPTY training set, exported without complaint.
# If n_train_rows comes back small, that is the bug to suspect first.
echo ">>> stage 2: export (full cohort, article volume)"
python -m marketrank.retrieval.dataset 0 --article-volume

echo "finished    : $(date -Is)"

# ---------------------------------------------------------------------------
# CHECKPOINT -- read these off the log before submitting the GPU job.
#
#   EXPORT n_articles          expect 105,542
#   EXPORT n_train_rows        expect ~10x the laptop's 2,900,248 -- this is the
#                              number that proves the cohort cap actually lifted
#   EXPORT n_eval_customers    expect 20,000
#   EXPORT n_eval_truth_pairs  expect 70,715 EXACTLY
#
# The last one is not approximate. The eval cohort is hash-derived and
# feature-independent, so it must reproduce the laptop's denominator to the
# pair. If it does not, every recall number stops being comparable with the
# ablation ladder and the GPU run is measuring something else. Stop and find out
# why rather than proceeding.
# ---------------------------------------------------------------------------
