import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_RAW = Path(os.environ.get("MARKETRANK_DATA_RAW", PROJECT_ROOT / "data" / "raw"))
WAREHOUSE = Path(os.environ.get("MARKETRANK_WAREHOUSE", PROJECT_ROOT / "warehouse"))
REPORTS = Path(os.environ.get("MARKETRANK_REPORTS", PROJECT_ROOT / "reports"))

os.environ.setdefault(
    "JAVA_HOME",
    "/Library/Java/JavaVirtualMachines/temurin-17.jdk/Contents/Home",
)
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
