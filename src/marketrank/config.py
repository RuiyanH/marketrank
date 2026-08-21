import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_RAW = Path(os.environ.get("MARKETRANK_DATA_RAW", PROJECT_ROOT / "data" / "raw"))
WAREHOUSE = Path(os.environ.get("MARKETRANK_WAREHOUSE", PROJECT_ROOT / "warehouse"))
REPORTS = Path(os.environ.get("MARKETRANK_REPORTS", PROJECT_ROOT / "reports"))
# BULK DERIVED TABLES -- week 4's candidate set and week 5's ranker training
# frame. Split out from the other paths because they are the only outputs
# measured in tens of GB: Rider 1 prices the training set at ~55.7 GB compressed
# / ~479 GB uncompressed at the shipped 159.4-candidate budget.
#
# Every other path here derives from PROJECT_ROOT, and on misha PROJECT_ROOT is
# $HOME -- the SMALLEST filesystem on that machine (125 GiB quota, ~19 GiB free)
# and the only one with no room for either number. That is not hypothetical: the
# 27M-row two-tower export landed there (`EXPORT out_dir .../marketrank/
# artifacts/twotower` in job 2303121), and a project-fileset write was already
# killed by quota on 2026-08-18. Default keeps the laptop's behaviour; misha
# points this at scratch via env.misha.sh.
#
# NOTE: nothing reads this yet -- it is introduced ahead of week 5's candidate
# job (C1), whose output is the first thing that would otherwise land in $HOME.
# Stated because "plumbed and inert" is a bug class this build keeps finding;
# unlike a model hyperparameter it cannot raise on use, so it is flagged here.
TABLES = Path(os.environ.get("MARKETRANK_TABLES", PROJECT_ROOT / "artifacts" / "tables"))
# Where Spark spills. Overridable so misha can point it at node-local /tmp
# (SETUP_MISHA Phase 3b -- GPFS handles small-file shuffle churn badly) while
# the laptop keeps it inside the repo. Left unset entirely, Spark spills to the
# OS temp dir where it is neither bounded nor visible: that is how R.5's
# co-visitation job retained 5.2 GB of orphaned shuffle across two crashes and
# took this machine to 332 MiB free.
SPARK_TMP = Path(os.environ.get("MARKETRANK_SPARK_TMP", PROJECT_ROOT / ".spark-tmp"))

# JAVA_HOME. `setdefault` so a real environment value always wins -- on misha
# `module load Java/17.0.4` sets it, and that must not be overridden.
#
# But the fallback is a **macOS** path, and setting it unconditionally is worse
# than leaving it unset: on Linux it made Spark launch
# `/Library/Java/.../bin/java`, which fails with "No such file or directory"
# and surfaces as JAVA_GATEWAY_EXITED -- an error that names neither Java nor
# the path. That is exactly what happens in a Jupyter kernel on the cluster,
# which does not inherit the shell's `module load`.
#
# So only claim the default when it actually exists. Elsewhere JAVA_HOME stays
# unset and Spark's own "JAVA_HOME is not set" is the error you get, which is
# the true one.
_MACOS_JDK = Path("/Library/Java/JavaVirtualMachines/temurin-17.jdk/Contents/Home")
if _MACOS_JDK.exists():
    os.environ.setdefault("JAVA_HOME", str(_MACOS_JDK))
os.environ.setdefault(
    "PYSPARK_PYTHON",
    str(PROJECT_ROOT / ".venv" / "bin" / "python"),
)

CATALOG = "local"
ICEBERG_VERSION = "1.11.0"

# Everything that imports marketrank -- a bare pytest, a notebook, any REPL that
# calls get_spark() -- needs SPARK_CONF_DIR set BEFORE session construction,
# because that is where the Iceberg catalog definition now lives. An export in
# the Makefile reaches only Makefile-launched processes, which is not how we
# actually run things. setdefault, not assignment, so a real export from
# env.misha.sh or CI wins.
os.environ.setdefault("SPARK_CONF_DIR", str(PROJECT_ROOT / "conf"))
SPARK_CONF_DIR = Path(os.environ["SPARK_CONF_DIR"])

SPARK_DEFAULTS_TEMPLATE = PROJECT_ROOT / "conf" / "spark-defaults.conf.template"
SPARK_DEFAULTS = SPARK_CONF_DIR / "spark-defaults.conf"


def render_spark_defaults() -> Path:
    """
    Render conf/spark-defaults.conf from its committed template.

    Spark performs no variable substitution in spark-defaults.conf, so a
    committed static file cannot carry the warehouse location -- and the
    warehouse location is environment-dependent by design (MARKETRANK_WAREHOUSE;
    env.misha.sh sets its own). Rendering keeps ONE source of truth for the
    catalog -- this module -- while still handing dbt a plain file it can read
    without importing marketrank.

    Idempotent: rewrites only when the content would change, so importing
    marketrank does not churn the file or fight a concurrent reader.
    """
    if not SPARK_DEFAULTS_TEMPLATE.exists():
        return SPARK_DEFAULTS
    text = SPARK_DEFAULTS_TEMPLATE.read_text().format(
        catalog=CATALOG,
        iceberg_version=ICEBERG_VERSION,
        warehouse=WAREHOUSE,
    )
    SPARK_CONF_DIR.mkdir(parents=True, exist_ok=True)
    if not SPARK_DEFAULTS.exists() or SPARK_DEFAULTS.read_text() != text:
        SPARK_DEFAULTS.write_text(text)
    return SPARK_DEFAULTS


render_spark_defaults()


if __name__ == "__main__":  # `python -m marketrank.config` -- used by the Makefile
    print(render_spark_defaults())
