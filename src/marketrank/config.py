import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_RAW = PROJECT_ROOT / "data" / "raw"
WAREHOUSE = PROJECT_ROOT / "warehouse"
REPORTS = PROJECT_ROOT / "reports"

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