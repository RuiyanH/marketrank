#!/bin/bash
# Step R.4 — train the two towers at full scale on misha.
#
# Submit from the repo root, and make the log directory FIRST:
#     mkdir -p logs && sbatch jobs/train_towers.sbatch
#
# Prerequisite: the two-tower export exists in artifacts/twotower/, built with
# the R.1 spine and the R.2 article-volume columns:
#     python -m marketrank.retrieval.dataset 0 --article-volume
# (cohort 0 meaning "no cap" — the full 1,371,980 customers; this only became
# true when the CLI learned to map <=0 to None. Before that it meant .limit(0),
# an empty training set.) That export is a
# Spark job and belongs in its own submission, not in this one; this job reads
# parquet and never starts a session.

#SBATCH --job-name=marketrank-towers
#SBATCH --partition=day
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=200G
#SBATCH --output=logs/%x-%j.out

# --time is mandatory: the `day` partition's DefaultTime is 1 hour and jobs are
# killed at it with no warning. 12h is deliberately generous because the CPU
# wall clock for this job is UNMEASURED — see the VERIFY note below. Drop it to
# roughly 1.5x the measured full run once one exists, since shorter jobs
# schedule sooner. If it ever needs >24h, move to --partition=week.

set -euo pipefail

source "$HOME/marketrank/env.misha.sh"
cd "$HOME/marketrank"

# Thread count follows the ALLOCATION, not the node. torch's intra-op pool and
# OpenMP both default to "every core I can see" — on a shared node that
# oversubscribes and runs slower than single-threaded. Passing it explicitly is
# the fix, and --threads hands the same number to torch.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-64}"

# ---------------------------------------------------------------------------
# The configuration R.4 is testing, declared here so it lands in the job log
# next to the run that used it.
#
# Scale is the LAST and WEAKEST of the recovery plan's four hypotheses, and it
# is run last on purpose: the reduced run's per-epoch curve plateaued at epoch 4,
# and a 20-epoch run on less data was WORSE than a 5-epoch one (overfitting).
# What this job changes against the reduced configuration:
#
#   training positives   2,900,248  ->  the full `train` slice
#   customer cohort        300,000  ->  all 1,371,980
#   embedding dim               64  ->  128
#   training window   2020-02-14..  ->  2018-09-20 .. 2020-08-11
#
# Everything else is held at the value the ablation ladder established, so this
# rung measures scale and nothing else.
# ---------------------------------------------------------------------------
DIM=128
# NO EARLY STOPPING here either -- fixed count, `best` chosen from history
# afterwards. Left at 12 rather than following the GPU sibling down to 5,
# deliberately: that curve was measured at FULL scale, where the peak was epoch
# 3, while this script's reduced-scale ladder peaked at epochs 4-7 (r2_recency
# 4, r3_uniform16 4, r3_uniform64 7). Cutting to 5 on the strength of a
# different scale's curve would risk truncating before the peak. If you measure
# this one, lower it and cite the run.
EPOCHS=12
BATCH=1024
LR=3e-3
TEMPERATURE=0.07
N_UNIFORM=64          # R.3's best; MUST match what the ablation ladder chose
RECENCY_HALF_LIFE=30  # days — R.2
LABEL="r4_scale"

echo "=== marketrank two-tower ============================================"
echo "job              : ${SLURM_JOB_ID:-interactive} on $(hostname)"
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
echo "dim / epochs     : ${DIM} / ${EPOCHS}"
echo "n_uniform        : ${N_UNIFORM}"
echo "recency half-life: ${RECENCY_HALF_LIFE}"
echo "started          : $(date -Is)"
echo "====================================================================="

# ---------------------------------------------------------------------------
# VERIFY — CPU-node training at this scale has NOT been measured.
#
# Everything in the reference build trained on Apple MPS at reduced scale
# (~90-110 s/epoch on 2.9M positives at d=64). This job is CPU, ~10x the
# positives and 2x the dim, and the scaling is not linear in any direction I am
# willing to assert. So: measure ONE epoch before committing to the full run.
#
# Run the probe first, read `seconds` out of the EPOCH line, and only then
# submit the real job with an --time you can defend:
#
#     sbatch --time=01:00:00 --export=ALL,PROBE=1 jobs/train_towers.sbatch
#
# The probe is one epoch with no early stopping — it exists to produce a wall
# clock, not a model.
# ---------------------------------------------------------------------------
if [[ "${PROBE:-0}" == "1" ]]; then
    echo "PROBE RUN — one epoch, for wall clock only"
    EPOCHS=1
    LABEL="r4_probe"
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
# What the Python entrypoint does (marketrank.jobs.train_towers):
#
#   1. Loads the exported parquet — no Spark session. The export is a separate
#      job for exactly this reason: holding JVM heap while torch allocates is
#      how a 200G allocation becomes an OOM.
#
#   2. Trains with in-batch sampled softmax plus N_UNIFORM uniform negatives,
#      all logQ-corrected. The logQ term is subtracted AFTER temperature
#      scaling; it is a correction in logit units. Getting that wrong cost the
#      reference build a 17x difference in recall@500 with the WORSE model
#      holding the BETTER training loss — see the ablation table in the README.
#
#   3. Evaluates recall@{12,100,500} on `val_tune` every epoch and checkpoints
#      the BEST epoch, not the last. val_tune is the slice splits.py allocates
#      for early stopping, so selecting on it is the intended use, not a leak.
#      Nothing here may read `ope_env`.
#
#   4. Writes artifacts/twotower/runs/${LABEL}/: model.pt, metrics.json (the
#      full per-epoch history plus the selected epoch and the exact args).
#
# The eval cohort is FIXED at the 20,000 hash-sampled val_tune customers and
# 70,715 true pairs every other number in this build uses; the entrypoint fails
# the run if the denominator moves, because a rung of the ablation ladder
# measured against a different denominator is not a comparison.
#
# After this finishes, the tower becomes a candidate SOURCE and is judged on
# marginal ceiling per slot, not on beating the baseline union solo:
#     python -m marketrank.jobs.ann_candidates --run ${LABEL} --top-n 50
#     python -m marketrank.jobs.candidate_ceiling --ann artifacts/twotower/runs/${LABEL}/ann_candidates.parquet
# ---------------------------------------------------------------------------
