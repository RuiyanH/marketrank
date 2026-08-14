# Runbook — Bring `marketrank` up on Yale misha (HPC)

**Audience:** an agent (or a human) with a shell on `misha.ycrc.yale.edu`.
**Goal:** the Iceberg data layer running on misha, with the repo working *unchanged* on both laptop and cluster.
**Estimated time:** 30–45 min, most of it the Kaggle download.

Execute phases in order. Each phase ends with a **CHECKPOINT** stating the expected result.
**If a checkpoint does not match, STOP and report — do not continue.**

---

## Facts about this cluster (verified 2026-08-14, do not re-derive)

| Fact | Value |
|---|---|
| Login | `rh849@misha.ycrc.yale.edu` (Yale VPN required) |
| Group | `dijk` |
| `~/project` | → `/gpfs/radev/project/dijk/rh849` — 4 TiB, never purged, **no backup** |
| `~/scratch` | → `/gpfs/radev/scratch/dijk/rh849` — 10 TiB, **purged after 60 days** |
| `~` (home) | 125 GiB quota, **~26 GiB free**, backed up. File limit 500k, ~138k used |
| `/tmp` | 3.4 TB node-local NVMe, 3.3 TB free — **not** GPFS |
| `$TMPDIR` | per-job dir under `/tmp`, auto-deleted at job end |
| System python | `/usr/bin/python3` = 3.11.13 (laptop runs 3.11.15 — close enough) |
| Java | `module load Java/17.0.4`. **Default is Java 21 — never `module load Java` bare** |
| Nodes | 64 cores, ~480 GiB RAM (`day`, `week`, `devel`); `bigmem` has ~1.9 TiB |
| Partitions | `devel` (2 nodes), `day` (14, **DefaultTime 1h**), `week` (10), `bigmem` (2) |
| Internet | Compute nodes reach Maven Central (HTTP 200) — `spark.jars.packages` works |

**Do not use the `Spark/3.5.4-foss-2022b-Scala-2.13` module.** It is a Scala 2.13 build; our
Iceberg coordinate is `_2.12`. Use pip-installed `pyspark==3.5.9`, which ships its own complete
Spark distribution (including `sbin/` scripts for multi-node) and matches the laptop exactly.

---

## Phase 0 — On the LAPTOP: push pending work

The repo has uncommitted changes. Misha will clone from GitHub, so anything unpushed is invisible there.

```bash
cd ~/Developer/marketrank && git add -A && git commit -m "Environment-aware paths; snapshot comparison helpers" && git push
```

**CHECKPOINT 0:** `git status --short` prints nothing.

---

## Phase 1 — On MISHA: clone and build the environment

```bash
cd ~ && git clone https://github.com/RuiyanH/marketrank.git && cd ~/marketrank && python3 -m venv .venv && .venv/bin/pip install --upgrade pip && .venv/bin/pip install -e . && .venv/bin/pip install pyspark==3.5.9 kaggle
```

**CHECKPOINT 1:**

```bash
~/marketrank/.venv/bin/python -c "import pyspark, sys; print(sys.version.split()[0], pyspark.__version__)"
```

Expect `3.11.13 3.5.9`.

**If `python3 -m venv` fails** (system Python missing `ensurepip`), fall back to conda:

```bash
module load miniconda/24.3.0 && conda create -y -p ~/marketrank/.venv python=3.11 && ~/marketrank/.venv/bin/pip install -e ~/marketrank && ~/marketrank/.venv/bin/pip install pyspark==3.5.9 kaggle
```

The `.venv` path is deliberate — `config.py` derives `PYSPARK_PYTHON` from
`PROJECT_ROOT/.venv/bin/python`, so the interpreter must live there under either method.

---

## Phase 2 — On MISHA: create directories

```bash
mkdir -p ~/project/marketrank/data/raw ~/project/marketrank/warehouse ~/.kaggle
```

**Why `~/project` and not `~` or `~/scratch`:** home has only ~26 GiB free and the warehouse will
outgrow it; scratch is purged at 60 days and this project runs ~10 weeks. `project` is 4 TiB and
never purged. It has no backup, but the warehouse is *derived* data — reproducible by re-running
the load — so backup is not the relevant property.

**CHECKPOINT 2:** `ls -d ~/project/marketrank/{data/raw,warehouse}` lists both without error.

---

## Phase 3 — Code changes

### 3a. `src/marketrank/config.py` — add `SPARK_TMP`

The three path lines already read the environment. Add a fourth. Target state of the path block:

```python
DATA_RAW = Path(os.environ.get("MARKETRANK_DATA_RAW", PROJECT_ROOT / "data" / "raw"))
WAREHOUSE = Path(os.environ.get("MARKETRANK_WAREHOUSE", PROJECT_ROOT / "warehouse"))
REPORTS = Path(os.environ.get("MARKETRANK_REPORTS", PROJECT_ROOT / "reports"))
SPARK_TMP = Path(os.environ.get("MARKETRANK_SPARK_TMP", PROJECT_ROOT / ".spark-tmp"))
```

Note `os.environ.get` (read with a fallback) rather than `os.environ.setdefault` (write into the
environment for a child process to inherit). `JAVA_HOME` and `PYSPARK_PYTHON` below need
`setdefault` because Spark's launcher reads them out of the environment; these four are only ever
read by our own Python. `setdefault` on `JAVA_HOME` is also what makes `module load Java/17.0.4`
win on misha while the Temurin path still works on the laptop — leave it alone.

### 3b. `src/marketrank/spark.py` — route shuffle spill to node-local disk

Add one line to the builder chain, after `spark.driver.memory`:

```python
        .config("spark.local.dir", str(config.SPARK_TMP))
```

**Why this matters:** shuffle spill is thousands of small files written and deleted at high
frequency. GPFS is optimized for large sequential reads across many nodes and handles small-file
churn badly. `/tmp` is node-local NVMe. This is also per-node by design — under multi-node Spark
each executor spills to its own `/tmp`, which is correct; no other node reads it.

Contrast with `WAREHOUSE`, which **must** stay on GPFS: all executors write partitions of the same
Iceberg table and must see one shared filesystem.

### 3c. `.gitignore` — add the local-mode spill dir

Append under the Spark section:

```
.spark-tmp/
```

### 3d. New file `env.misha.sh` in the repo root — **commit this**

```bash
module load Java/17.0.4
export MARKETRANK_DATA_RAW="$HOME/project/marketrank/data/raw"
export MARKETRANK_WAREHOUSE="$HOME/project/marketrank/warehouse"
export MARKETRANK_SPARK_TMP="${TMPDIR:-/tmp}/marketrank-spark"
source "$HOME/marketrank/.venv/bin/activate"
```

`source env.misha.sh` at the start of every misha session and inside every Slurm job script.
The laptop never sources it and keeps using `config.py`'s defaults.

**CHECKPOINT 3:**

```bash
cd ~/marketrank && source env.misha.sh && python -c "from marketrank import config; print(config.WAREHOUSE); print(config.SPARK_TMP)" && java -version 2>&1 | head -1
```

Expect the warehouse under `/gpfs/radev/project/dijk/rh849/...`, spark tmp under `/tmp/...`,
and `openjdk version "17.0.4"`. **If Java reports 21, the module load failed — stop.**

Commit and push these changes.

---

## Phase 4 — Data

### 4a. On the LAPTOP: copy the Kaggle token (38 bytes, not the 3.5 GB of CSVs)

```bash
scp ~/.kaggle/access_token rh849@misha.ycrc.yale.edu:~/.kaggle/
```

The cluster has fast direct internet, so re-downloading there beats pushing 3.5 GB up a home
connection over VPN.

### 4b. On MISHA: download the three files

```bash
source ~/marketrank/env.misha.sh && cd ~/project/marketrank/data/raw && for f in transactions_train.csv articles.csv customers.csv; do kaggle competitions download -c h-and-m-personalized-fashion-recommendations -f "$f"; done && unzip -o '*.zip' && rm -f *.zip
```

**The `-f` flag is required.** Without it Kaggle serves the entire competition, including ~25 GB of
product images this project never uses.

**CHECKPOINT 4:**

```bash
ls -la ~/project/marketrank/data/raw/
```

Expect roughly `transactions_train.csv` 3.2 G, `customers.csv` 198 M, `articles.csv` 34 M.

If the Kaggle CLI reports `401` or a missing token, the OAuth token may be laptop-bound; re-run the
export from the Kaggle account page and copy the new file.

---

## Phase 5 — Build and verify the Iceberg table

```bash
cd ~/marketrank && source env.misha.sh && python
```

Then, interactively:

```python
from marketrank.spark import get_spark
from marketrank import ingest, checks

spark = get_spark(driver_memory="64g", master="local[16]")
ingest.create_tables(spark)
ingest.load_transactions(spark, "2018-09-20", "2020-09-22")
```

`driver_memory="64g"` and `local[16]` are conservative for a 480 GiB / 64-core node and polite on
the shared `devel` node. Raise them for the real backfill in a batch job.

**CHECKPOINT 5a:** the table has data.

```python
spark.sql("SELECT COUNT(*) FROM local.raw.transactions").show()
```

Expect **31,788,324** rows.

**CHECKPOINT 5b:** the load is idempotent — re-running changes nothing.

```python
before = checks.snapshot_ids(spark, ingest.TRANSACTIONS_TABLE)
a = checks.read_snapshot(spark, ingest.TRANSACTIONS_TABLE)
ingest.load_transactions(spark, "2019-01-01", "2019-01-31")
b = checks.read_snapshot(spark, ingest.TRANSACTIONS_TABLE)
checks.assert_identical(a, b)
```

`assert_identical` must pass. It uses `exceptAll` in **both** directions — duplicate-preserving,
unlike `EXCEPT`, so a row appearing twice where it should appear once is still caught.

Exit with `exit()`. **Do not leave the Spark session open** — an idle session holds its spill
directory. On `/tmp` that is now harmless, but the habit matters.

---

## Phase 6 — Reclaim laptop space (only after CHECKPOINT 5b passes)

```bash
rm -rf ~/Developer/marketrank/data ~/Developer/marketrank/warehouse
```

Frees ~4.4 GB. Both are gitignored and both are reproducible: `data` from Kaggle, `warehouse` by
re-running Phase 5. The laptop clone stays useful for editing and for dbt/DuckDB work.

---

## Appendix A — Slurm batch template (for Week 2's backfill, not needed yet)

`jobs/backfill.sbatch`:

```bash
#!/bin/bash
#SBATCH --job-name=marketrank-backfill
#SBATCH --partition=day
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=400G
#SBATCH --output=logs/%x-%j.out

source "$HOME/marketrank/env.misha.sh"
cd "$HOME/marketrank"
python -m marketrank.jobs.backfill
```

**Always pass `--time`.** The `day` partition's DefaultTime is 1 hour and jobs are killed at it.

---

## Appendix B — Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `java: command not found` | `env.misha.sh` not sourced | `source ~/marketrank/env.misha.sh` |
| Java reports 21 | bare `module load Java` | `module load Java/17.0.4` explicitly |
| `PYTHON_VERSION_MISMATCH` | worker Python found via PATH, not venv | confirm `config.PYSPARK_PYTHON` points into `.venv` |
| Ivy / Maven resolution failure | no internet on the node | pre-fetch the JAR on a login node, switch to `spark.jars` with a local path |
| `NoSuchMethodError` / `NoClassDefFound` on Iceberg | Scala 2.12 vs 2.13 | you loaded the Spark module; use pip `pyspark` instead |
| `TABLE_OR_VIEW_NOT_FOUND` | `create_tables()` not run | run it; the plan shows `'UnresolvedRelation` and fails at planning time |
| `BindException` | stale hostname → DHCP address | already handled: loopback bind under `master="local*"` |
| Job killed at 1 hour | no `--time` | add `#SBATCH --time=HH:MM:SS` |
| Disk full during shuffle | spill landed on GPFS or home | verify `spark.local.dir` resolves under `/tmp` |

---

## Appendix C — What is deliberately NOT here

- **No orchestrator.** Data is static historical files; nothing arrives on a schedule.
- **No extra tools.** The stack is Spark + Iceberg + dbt plus a CI file. Redis, Kafka, Great
  Expectations, and a second warehouse were each evaluated and cut for adding surface without
  adding a semantic.
- **No multi-node Spark yet.** One node has 64 cores and 480 GiB. Whether multi-node is justified
  is a **Week 2 decision made with a measured single-node wall-clock**, not an assumption. Running
  distributed on a job that fits comfortably on one node is the manufactured-scale trap.
