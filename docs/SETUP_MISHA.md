# Runbook — Bring `marketrank` up on Yale misha (HPC)

**Status:** NOT YET EXECUTED — deferred to **Week 4** (decided 2026-08-14).
Weeks 1–3 run on the laptop: ~18 GB free is enough for the data layer's <5 GB footprint, and the
local REPL loop is faster and needs no VPN. Week 4's candidate generation needs 40–160 GB, which
is the trigger to move. Run this then, or earlier if local disk gets tight.

**Audience:** an agent (or a human) with a shell on `misha.ycrc.yale.edu`.
**Goal:** the Iceberg data layer running on misha, with the repo working *unchanged* on both laptop and cluster.
**Estimated time:** 30–45 min, most of it the Kaggle download.

**Recon date: 2026-08-14.** Everything in the facts table below was verified by hand that day.
Re-verify only quotas and free space — those drift. The rest (module names, paths, node specs,
partition limits) is stable cluster configuration.

Execute phases in order. Each phase ends with a **CHECKPOINT** stating the expected result.
**If a checkpoint does not match, STOP and report — do not continue.**

---

## Facts about this cluster (verified 2026-08-14, do not re-derive)

| Fact | Value |
|---|---|
| Login | `rh849@misha.ycrc.yale.edu` (Yale VPN required) |
| Group | `dijk` |
| `~/project` | → `/gpfs/radev/project/dijk/rh849` — never purged, **no backup**. 4 TiB is the *fileset* size, NOT your quota: a 2026-08-18 write died with `Disk quota exceeded` here. **Do not put bulk data here** |
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
mkdir -p ~/scratch/marketrank/data/raw ~/scratch/marketrank/warehouse ~/.kaggle
```

**Why `~/scratch` and not `~` or `~/project`** — this reverses the original choice, on evidence.
Home is out on size (125 GiB quota, most of it already spoken for by unrelated work). `project`
was the first choice because the fileset is 4 TiB and never purged; on 2026-08-18 a transactions
load died there with `Disk quota exceeded` anyway. **`df` cannot see this** — it reports the whole
3.7 PiB GPFS filesystem, not your quota — and `getquota` reported the wrong group, so the binding
limit was never visible. See [Appendix B](#appendix-b--troubleshooting).

That leaves scratch: 10 TiB, 15M inodes, empty. The cost is the **60-day purge**, and it is
affordable because *everything* under this directory is reproducible — the warehouse by re-running
the load, the CSVs by re-running the Kaggle download in 4b. Nothing here is a source of truth;
the source of truth is the git repo, which lives in home. If you return to this project after two
idle months, expect to redo Phase 4b and Phase 5, and nothing else.

**CHECKPOINT 2:** `ls -d ~/scratch/marketrank/{data/raw,warehouse}` lists both without error.

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

### 3d. `env.misha.sh` in the repo root — **already committed**, no action needed

```bash
module load Java/17.0.4
export MARKETRANK_DATA_RAW="$HOME/scratch/marketrank/data/raw"
export MARKETRANK_WAREHOUSE="$HOME/scratch/marketrank/warehouse"
export MARKETRANK_SPARK_TMP="${TMPDIR:-/tmp}/marketrank-spark"
export SPARK_CONF_DIR="$HOME/marketrank/conf"
source "$HOME/marketrank/.venv/bin/activate"
```

`source env.misha.sh` at the start of every misha session and inside every Slurm job script.
The laptop never sources it and keeps using `config.py`'s defaults.

**On `SPARK_CONF_DIR`** — added 2026-08-14, owned by [`IMPLEMENTATION.md`](IMPLEMENTATION.md) step 1.5.
`conf/spark-defaults.conf` holds the Iceberg jar coordinate and the catalog config, because
dbt-spark's session method builds its *own* SparkSession and cannot receive them from
`profiles.yml`. `get_spark()` reads the same file, so both consumers see one catalog definition.

`config.py` sets this same value with `os.environ.setdefault`, so anything importing `marketrank`
is already covered and the export here is deliberate redundancy — it costs a line and it means the
variable is visible in the shell where you will be debugging. `setdefault` also means this export
wins if the two ever disagree. What the export genuinely adds is **dbt**, which never imports
`marketrank`: run `dbt` on misha without it and the session comes up with no `local` catalog.

**If `conf/` does not exist when you run this**, step 1.5 hasn't landed yet — drop the line and
Phase 5 still works, because `get_spark()` still carries the catalog config itself at that point.
Do **not** move `spark.local.dir` (3b) into that file to "match" it: spill location is
mode- and machine-dependent and stays in the builder by design, for the same reason the
loopback bind-address workaround does. Engine-agnostic config goes in the file; runtime and
topology config stays in `get_spark()`.

**CHECKPOINT 3:**

```bash
cd ~/marketrank && source env.misha.sh && python -c "from marketrank import config; print(config.WAREHOUSE); print(config.SPARK_TMP)" && java -version 2>&1 | head -1
```

Expect the warehouse under `/gpfs/radev/scratch/dijk/rh849/...`, spark tmp under `/tmp/...`,
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
source ~/marketrank/env.misha.sh && cd ~/scratch/marketrank/data/raw && for f in transactions_train.csv articles.csv customers.csv; do kaggle competitions download -c h-and-m-personalized-fashion-recommendations -f "$f"; done && unzip -o '*.zip' && rm -f *.zip
```

**The `-f` flag is required.** Without it Kaggle serves the entire competition, including ~25 GB of
product images this project never uses.

**CHECKPOINT 4:**

```bash
ls -la ~/scratch/marketrank/data/raw/
```

Expect roughly `transactions_train.csv` 3.2 G, `customers.csv` 198 M, `articles.csv` 34 M.

If the Kaggle CLI reports `401` or a missing token, the OAuth token may be laptop-bound; re-run the
export from the Kaggle account page and copy the new file.

---

## Phase 5 — Build and verify the Iceberg table

**Get an allocation first.** This step is a 31.8M-row load with a 48 GiB JVM; a login node will
not give you the memory and running it there is against cluster etiquette. From a fresh login:

```bash
salloc -p day -c 16 --mem=64G -t 2:00:00
```

Do **not** run `salloc` from inside an existing allocation — Slurm rejects the nested call with
`SLURM_MEM_PER_CPU, SLURM_MEM_PER_GPU, and SLURM_MEM_PER_NODE are mutually exclusive`. Reconnect
with a fresh `ssh` instead. Check `hostname`: `login2` means the `salloc` did not take.

`salloc` hands you a **fresh shell that has not sourced anything**, so the next line is mandatory
in every new allocation — the single most common way this phase fails:

```bash
cd ~/marketrank && source env.misha.sh && python
```

Then, interactively:

```python
from marketrank.spark import get_spark
from marketrank import ingest, checks

spark = get_spark(driver_memory="48g", master="local[16]")
ingest.create_tables(spark)
ingest.load_transactions(spark, "2018-09-20", "2020-09-22")
```

**`driver_memory` must be strictly less than `--mem`.** It sets the JVM *heap*; resident memory is
heap plus metaspace, off-heap buffers and the Python workers. A 64 GiB heap under a 64 GiB cgroup
is killed the moment the heap fills — the same failure as the `mem=2G` kill, just later. `48g`
under `--mem=64G` leaves headroom. Raise both together for the real backfill in a batch job.

**CHECKPOINT 5a:** the table has data.

```python
spark.sql("SELECT COUNT(*) FROM local.raw.transactions").show()
```

Expect **31,788,324** rows.

**CHECKPOINT 5b:** the load is idempotent — re-running changes nothing.

```python
TABLE = ingest.TRANSACTIONS_TABLE
snap_a = checks.snapshot_ids(spark, TABLE)[-1]
ingest.load_transactions(spark, "2019-01-01", "2019-01-31")
snap_b = checks.snapshot_ids(spark, TABLE)[-1]

a = checks.read_snapshot(spark, TABLE, snap_a)
b = checks.read_snapshot(spark, TABLE, snap_b)
checks.assert_identical(a, b)
```

`assert_identical` must pass. It uses `exceptAll` in **both** directions — duplicate-preserving,
unlike `EXCEPT`, so a row appearing twice where it should appear once is still caught.

**On the full table this is four scans and two 31.8M-row shuffles.** It is not hung, but it looks
hung, and interrupting it kills the py4j connection and costs you the whole session. Scope it to
the window the MERGE actually touched and add a total-count guard for everything outside it:

```python
assert a.count() == b.count(), "row count changed"
w = "t_dat >= date'2019-01-01' AND t_dat <= date'2019-01-31'"
checks.assert_identical(a.filter(w), b.filter(w))
```

This is not a weaker check. A duplicate can only appear where the MERGE wrote, and damage outside
that window necessarily moves the total.

**Both reads must pin an explicit snapshot id** (corrected 2026-08-14). Reading the table without
one and holding the DataFrame across the reload does not capture a "before" — Spark DataFrames are
lazy, so an unpinned read is resolved when the action runs, and both sides can end up scanning the
same post-reload state. The check then passes by comparing the table to itself, which is the worst
possible outcome for a test whose whole job is to fail when the load isn't idempotent. Capture the
ids first; read afterwards.

**Signatures** (owned by [`IMPLEMENTATION.md`](IMPLEMENTATION.md) steps 1.3 and 1.6): `read_snapshot(spark, table, snapshot_id)`
and `assert_identical(a, b, ignore_cols=("_ingested_at",))`, taking two DataFrames. The current
`checks.py` still has `assert_identical(spark, table, snap_a, snap_b)` — the refactor lands with
step 1.3, which is also what makes `ignore_cols` necessary, since `_ingested_at` differs between
two otherwise-identical loads. If you reach this phase before step 1.3, call the old four-argument
form with the two ids and skip the two `read_snapshot` lines.

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
| `Disk quota exceeded` on a warehouse write | bulk data on `project` or home, not `scratch` | check `config.WAREHOUSE` is under `~/scratch`; `df` will **not** show this — it reports the 3.7 PiB filesystem, not your quota |
| `getquota` numbers contradict the error | report is per-group and may name a group your data is not in (it printed `timmermans`; this account's data is under `dijk`), and its Usage Details block is a once-daily snapshot | trust the failure over the table; confirm the real fileset with `readlink -f ~/scratch` |
| Iceberg table unreadable after moving the warehouse | Hadoop catalog stores **absolute** paths in `.metadata.json` and every manifest | a moved warehouse is not portable — delete it and re-run the load |

---

## Appendix C — What is deliberately NOT here

- **No orchestrator.** Data is static historical files; nothing arrives on a schedule.
- **No extra tools.** The stack is Spark + Iceberg + dbt plus a CI file. Redis, Kafka, Great
  Expectations, and a second warehouse were each evaluated and cut for adding surface without
  adding a semantic.
- **No multi-node Spark yet.** One node has 64 cores and 480 GiB. Whether multi-node is justified
  is a decision made with a **measured single-node wall-clock**, not an assumption. Running
  distributed on a job that fits comfortably on one node is the manufactured-scale trap.

---

## Appendix D — Open items carried from the 2026-08-14 recon

**1. ~~How long does this account live?~~ RESOLVED 2026-08-15 — access runs through ~2027-08.**
Storage sits under the `dijk` *group* allocation, not a personal one, and the Stanford move ends
the Yale affiliation eventually — but not for another year. The whole 10-week build finishes with
roughly nine months of margin, so **this is no longer a sequencing constraint**: weeks 4–8 can all
assume misha, and the serving path can be planned on its merits rather than around an expiry date.

Two things this does *not* resolve, so don't let the margin turn into complacency:

- Item 2 below still stands on its own — a demo behind Yale's VPN is not demoable to a recruiter
  regardless of how long the account lives.
- Re-confirm before any work lands after ~2027-05. A year of margin quietly becomes three months,
  and group allocations can change ahead of the affiliation ending.

Mitigation already in place either way: code is in GitHub, and the warehouse is derived data
rebuilt by re-running Phase 5 against Kaggle. Losing access costs a re-run, not an archaeology
project.

**2. The flagship must stay demoable.** "Deployed and pokeable" is one of the plan's deliverables,
and nothing behind Yale's VPN is pokeable by a recruiter. Weeks 5–6's serving path should land
somewhere public regardless of where training runs.

**3. `pyproject.toml` declares no dependencies.** ~~Move the pins into `[project] dependencies`
before Week 4.~~ **Superseded 2026-08-14** — this is now [`IMPLEMENTATION.md`](IMPLEMENTATION.md)
step 1.0, pulled forward to Week 1 because the CI job in step 1.7 cannot pass without it: with no
dependency block, `pip install -e .` installs nothing, `import pyspark` fails, and the
point-in-time leakage test errors instead of running. Phase 1 of this runbook installs
`pyspark==3.5.9` explicitly and so is unaffected either way.

**4. Multi-node Spark on Slurm, if it turns out to be warranted.** Slurm does not run Spark
natively. The pattern is: allocate N nodes, start a standalone master on the first, start workers
on the rest, point `master=` at the master's URL. pip-installed pyspark ships the `sbin/` scripts
for this. Do not attempt it before the single-node measurement justifies it.
