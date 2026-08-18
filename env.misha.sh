# Sourced at the start of every misha session and inside every Slurm job script.
# The laptop never sources this and keeps config.py's defaults.
#
# Every path here is on scratch, which is PURGED AFTER 60 DAYS. That is
# deliberate: both the warehouse and the raw CSVs are reproducible (re-run the
# load; re-run the Kaggle download). The project fileset looks like the safer
# home and is not -- a load died there with `Disk quota exceeded` on 2026-08-18.
module load Java/17.0.4
export MARKETRANK_DATA_RAW="$HOME/scratch/marketrank/data/raw"
export MARKETRANK_WAREHOUSE="$HOME/scratch/marketrank/warehouse"
export MARKETRANK_SPARK_TMP="${TMPDIR:-/tmp}/marketrank-spark"
export SPARK_CONF_DIR="$HOME/marketrank/conf"
source "$HOME/marketrank/.venv/bin/activate"
