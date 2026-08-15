# BUILD_NOTES.md

Build log for `docs/IMPLEMENTATION.md`, executed on branch `build/implementation`
in an isolated worktree at `/Users/test/Developer/marketrank-build`.

Every number below was produced by a command that actually ran. Where a number
was not produced, the entry says **not run** rather than an estimate.

## Environment this was built on

| Fact | Value |
|---|---|
| Machine | Apple Silicon macOS 25.5.0, **8 cores, 16 GiB RAM** |
| Free disk at start | **15 GiB** on the volume holding the worktree |
| Python | 3.11.15 (`/opt/homebrew/bin/python3.11` → worktree `.venv`) |
| Java | Temurin 17.0.20 |
| `MARKETRANK_DATA_RAW` | `/Users/test/Developer/marketrank/data/raw` (read-only) |
| `MARKETRANK_WAREHOUSE` | `/Users/test/Developer/marketrank-build/warehouse` (built from scratch) |

**The single most important fact about this build:** 8 cores / 16 GiB / 15 GiB
free disk is roughly two orders of magnitude below what week 4's candidate
generation needs (§5 of the spec sizes it at 40–160 GB of *output*). Everything
through week 2 ran at full scale. From week 3 on, full-scale code was written and
a **reduced configuration** was executed; each reduction is stated at the step and
carried into every number derived from it.

---

## Step 1.0 — Declare your dependencies

**Checkpoint:** fresh venv + `pip install -e ".[dev]"` gives working `pytest` and `dbt`.

Observed, on a venv created from scratch with `python3.11 -m venv .venv`:

```
pytest 9.1.1
dbt Core: 1.12.2   plugins: duckdb 1.11.0, spark 1.11.0
```

Resolved versions worth recording (they are not pinned beyond `pyspark`):
pyspark 3.5.9, dbt-core 1.12.2, dbt-spark 1.11.0, dbt-duckdb 1.11.0, duckdb 1.5.5,
py4j 0.10.9.9. **PASS.**

Deviation: `lightgbm`, `torch`, `hnswlib` are *not* in the `dev` extra at this
commit. The doc marks them "(later)", so they are added in the commit for the week
that first needs them, to keep the git history honest about when each arrived.

## Step 1.1 — Load `articles` and `customers`

**Checkpoint (marked VERIFY in the doc):** `local.raw.articles` = 105,542 and
`local.raw.customers` = 1,371,980.

Observed, on a warehouse built from scratch:

```
articles count:   105542      (load wall clock 4.0 s)
customers count: 1371980      (load wall clock 2.8 s)
```

**VERIFY resolved: the doc's widely-cited counts are correct.** They also match
`wc -l` on the CSVs minus the header (105,543 / 1,371,981 lines), which is the
independent check worth doing — it also proves no field in `articles.csv`
contains an embedded newline, so no `multiLine` read option is needed.

So §2's résumé claim should read **105,542 articles / 1,371,980 customers**.

Zero-padding check: `article_id` reads back as `0108775015`, string, leading zero
intact.

Deviation: `create_tables()` declares DDL only for `raw.transactions`. `articles`
and `customers` are created by `createOrReplace()` in their loaders, which owns
their schema. That is deliberate — §0's problem #1 is a table and a `CREATE TABLE`
statement drifting apart, and the cheapest way to not have that problem for a
snapshot-replaced table is to not have a second declaration of its schema.

## Step 1.2 — Decide the write strategy, deliberately

No checkpoint; the step is a comment and a decision.

**Think-first answer, confirmed by reading the code path rather than guessing:**
`writeTo(...).overwritePartitions()` is Spark's dynamic partition overwrite, so a
corrections file containing 3 rows for `2019-06-01` leaves exactly 3 rows in that
partition. The doc is right. Nothing to run here — the same fact is *demonstrated*
in step 9.1, and that is where the number goes.

Recorded so week 9 does not have to re-derive it: the mechanism the merge case
needs is a key, and the raw transaction rows have none. That is what step 1.4 is
for.

## Step 1.3 — `_ingested_at`, using schema evolution for real

Note on §0's problem #1 (`promo_flag`): it does not exist in this build. The
divergence between `create_tables()` and the live table came from the warehouse
having been evolved by hand; a warehouse rebuilt from the committed DDL simply
does not have the column. So `_ingested_at` is the *first* schema evolution here,
and it is the one with a reason. If you are redoing this on the existing
warehouse, drop `promo_flag` in the same commit.

**Base load, for the record** (fresh warehouse, full `transactions_train.csv`):

```
TXN_LOAD_SECONDS 49.3
TXN_COUNT        31,788,324
TXN_SPAN         2018-09-20 .. 2020-09-22   DAYS 734
warehouse size after load: 559 MB
```

31.79M is the number to use wherever the spec writes "~32M".

### VERIFY — what `overwritePartitions()` does when the DataFrame lacks a column

The doc asks whether Iceberg fills null or refuses. **It refuses**, at analysis
time, before any work:

```
AnalysisException [INCOMPATIBLE_DATA_FOR_TABLE.CANNOT_FIND_DATA]
Cannot write incompatible data for the table `local`.`raw`.`transactions`:
Cannot find data for the output column `_ingested_at`.
```

Worth stating as two directions rather than one rule, because they differ:

- **Read-side drift is tolerated.** Data files written before the column existed
  read back as null — that is the metadata-only evolution working.
- **Write-side drift is rejected.** `DataFrameWriterV2` resolves the incoming
  DataFrame against the *current* table schema by name and fails on a missing
  column. Producer and table drifting apart is a loud failure at write, not a
  silent column of nulls.

That asymmetry is the actual answer to "what happens when the producer and the
table drift apart", and it is a better one than the doc's either/or implies.

### Evidence the evolution was free

`transactions.snapshots` after the ALTER contains **3** snapshots — the initial
load and the two step-1.3 day reloads. The ALTER produced a
`metadata_log_entries` row at `23:11:39.998` and **no snapshot at all**, and
`latest_schema_id` went 0 → 1 only at the next write. No data files were touched:

```
snapshot_id          committed_at              operation  added      total
2968744110833297794  2026-08-14 23:10:28.944   overwrite  31788324   31788324
4827918704589656546  2026-08-14 23:12:39.522   overwrite  62619      31788324
5931284054021401369  2026-08-14 23:12:56.625   overwrite  62619      31788324
```

### Checkpoint — both facts, together

Day used: `2019-06-01` (62,619 rows), loaded twice back to back.

```
distinct _ingested_at, run a: 2026-08-14 19:12:21.582137
distinct _ingested_at, run b: 2026-08-14 19:12:41.626972
assert_identical(a, b)                -> ignored ['_ingested_at']; 0 / 0; IDENTICAL
assert_identical(a, b, ignore_cols=()) -> 62619 / 62619; AssertionError
rows still NULL in _ingested_at elsewhere: 31,725,705 of 31,788,324
```

31,725,705 + 62,619 = 31,788,324, so exactly the reloaded day carries a
timestamp. **PASS**, both halves.

### Deviation — `assert_identical`'s default

The doc specifies `assert_identical(a, b, ignore_cols=("_ingested_at",))`, but
its own prose asks for "anything else you later add with a leading underscore"
and for a rule rather than a hardcoded column name. Those two are in tension:
week 9 adds `_source_file`, and with the literal default in place `assert_identical`
would go red the first time a corrections batch lands, for a reason that has
nothing to do with idempotency.

Implemented as `ignore_cols=None` meaning *every column whose name starts with
`_`*, with an explicit tuple accepted as an override and `()` demanding a
whole-row match. `()` is what the second half of this checkpoint uses, so the
override is not speculative API — the checkpoint needs it.

## Step 1.4 — Resolve the grain

Measured on the full log before deciding (all four numbers from one Spark job):

```
source rows                                              31,788,324
distinct (customer, article, t_dat, channel)             28,583,889
distinct (customer, article, t_dat, channel, price)      28,813,419
groups with >1 distinct price                               223,068  (0.78%)
max distinct prices inside one group                              8
distinct price values                                         9,857
distinct price values cast to DECIMAL(10,8)                   9,857
price range                          1.694915254237288e-05 .. 0.5915254237288136
```

**Chosen: option (b), price OUT of the key.** Grain
`(customer_id, article_id, t_dat, sales_channel_id)`, 28,583,889 rows.

**Multi-quantity purchase rate = 10.08%** (3,204,435 of 31,788,324 source rows
collapse). With price kept in the key it would have been 9.36% — the gap between
those two, 229,530 rows, is the mid-day-markdown population.

Reasoning recorded because the doc leaves this decision open: a markdown does not
make a second basket line, and carving 0.78% of groups into extra rows to
preserve a price difference is a modeling artifact rather than a business
distinction. `fact_transaction` instead carries `price_mean = sum(price)/qty`,
`price_min`, `price_max` (so the markdown stays visible) and
`revenue = sum(price)`. Note `revenue = sum(price)` is exact and identical to
`qty * price_mean`, so nothing is lost even in the multi-price groups — this is
strictly better than the doc's "accept the loss" framing of the mean-price
variant.

**On the `DECIMAL(10,8)` advice:** with price out of the key it stops mattering,
exactly as the doc predicts. Measured anyway, since it is cheap and the doc gives
no number: all 9,857 distinct prices survive the cast without collision, so the
`DECIMAL(10,8)` recommendation would have been safe. Worth knowing that 8 decimal
places is *not* obviously enough by inspection — the raw values carry ~18
significant digits (`0.050830508474576264`) — and the reason it works is that the
underlying price grid is coarse: 9,857 values over a 4-decade range.

**Checkpoint — measured half.** The count query ran and is quoted above.
`count(distinct customer_id, article_id, t_dat, sales_channel_id)` = 28,583,889,
which is by construction the row count a `GROUP BY` on those four columns
produces, so the grain is unique. Source minus fact = 31,788,324 − 28,583,889 =
**3,204,435 rows**, the multi-quantity purchase population.

**Checkpoint — not yet discharged as a property of a built table.** At this commit
there is no `dbt/` directory and no materialized `fact_transaction`. The number
above is a property of a query over `raw.transactions`, not of a table that
exists. Step 1.5 below records the materialized row count and the dbt uniqueness
test result when they have actually been run.

## Step 1.5 — Stand up dbt on Spark

This step needed four corrections to the doc, three of them because the doc's
design is right but incomplete about *which* consumer gets what. All four are
below with the exact error each produced, because each one costs an hour to
diagnose from scratch.

### Correction 1 — a static `spark-defaults.conf` cannot carry the warehouse path

The doc says to create `conf/spark-defaults.conf` holding, among other things,
"the `local` catalog config". The catalog config includes
`spark.sql.catalog.local.warehouse`, and the warehouse location is
environment-dependent by design: `config.py` reads `MARKETRANK_WAREHOUSE`, and
`SETUP_MISHA.md`'s `env.misha.sh` sets a different one. **Spark does not
interpolate environment variables in `spark-defaults.conf`.** Tested directly:

```
spark.sql.catalog.local.warehouse   ${env:MARKETRANK_WAREHOUSE}

-> s.conf.get(...) returns the literal string "${env:MARKETRANK_WAREHOUSE}"
-> IllegalArgumentException: java.net.URISyntaxException:
   Relative path in absolute URI: ${env:MARKETRANK_WAREHOUSE%7D
```

So a committed static file would have to hardcode one machine's path, which
contradicts the runbook it is supposed to serve.

**Resolution:** `conf/spark-defaults.conf.template` is committed;
`marketrank.config.render_spark_defaults()` renders `conf/spark-defaults.conf`
from it at import time (idempotent, rewrites only on change), and the rendered
file is gitignored. `make render-conf` / `python -m marketrank.config` is the
entry point for the one consumer that does not import `marketrank` — dbt. This
keeps the doc's actual goal (one definition of the catalog, visible to both
consumers) while letting the warehouse move per machine.

### Correction 2 — dbt-spark's session has no driver bind address, and dies

The doc's split-by-kind rule puts `spark.driver.bindAddress` /
`spark.driver.host` in `get_spark()` because they are machine- and mode-
dependent. Correct — but dbt-spark's session method never calls `get_spark()`,
so on this laptop (hostname resolves to a stale DHCP address) `dbt build` failed
before running a single model:

```
WARN Utils: Service 'sparkDriver' could not bind on a random free port.
  ... io.netty.channel.AbstractChannel.bind ... make: *** [dbt] Error 2
```

**Resolution that preserves the split:** export `SPARK_LOCAL_IP=127.0.0.1` from
the Makefile. It is the environment-level spelling of the same two settings, so
it stays out of the committed conf file (where it would break a multi-node
cluster) and still reaches dbt. The doc should say this; without it, step 1.5's
checkpoint is unreachable on any machine that needs the loopback workaround —
which is the machine the doc was written on.

### Correction 3 — Iceberg has no views, so staging cannot be views

The doc says staging models are views. On the Spark target every one of them
failed:

```
ERROR creating sql view model staging.stg_articles
  Replacing a view is not supported by catalog: local
```

Iceberg's `SparkCatalog` (1.11.0, Hadoop catalog) does not implement
`ViewCatalog`. **Resolution: `ephemeral`.** Staging here is pure projection
(rename + cast, no business logic), so inlining it as a CTE stores nothing and
costs nothing. It is also better than the obvious alternative of "table on
Spark, view on DuckDB", because that would make dev and CI materialise
differently — the opposite of what step 1.6 wants from CI.

### Correction 4 — seeds cannot build on the Spark target

```
ERROR loading seed file raw_seed.seed_articles
  [DATATYPE_MISSING_SIZE] DataType "VARCHAR" requires a length parameter
```

dbt-spark emits unsized `varchar` for seed columns typed as strings, which Spark
rejects. Seeds exist so CI can run without the 3.5 GB extract, so they have no
business on the Spark target at all. **Resolution:**
`+enabled: "{{ target.type == 'duckdb' }}"` in `dbt_project.yml`.

### VERIFY — does `spark.sql.defaultCatalog=local` do the routing?

**Yes.** With it set and no catalog qualification anywhere in `dbt_project.yml`
or `sources.yml`, dbt's unqualified `marts.fact_transaction` landed in the
Iceberg catalog:

```
SHOW NAMESPACES IN local  ->  staging, marts, raw_seed, raw
```

The fallback the doc names (fully qualifying the catalog in `dbt_project.yml`)
was not needed. Note the flip side: because the default catalog is Iceberg,
*everything* dbt does goes there, which is exactly how correction 3 surfaced.

### Checkpoint

`make dbt` (Spark/Iceberg target), from the real 31.8M-row warehouse:

```
1 of 16 OK created sql table model marts.dim_article        OK in  4.64s
2 of 16 OK created sql table model marts.dim_customer       OK in  3.38s
3 of 16 OK created sql table model marts.fact_transaction   OK in 59.39s
... 13 data tests ...
Done. PASS=16 WARN=0 ERROR=0 SKIP=0 NO-OP=0 REUSED=0 TOTAL=16
wall clock: 2 min 09 s
```

Is it genuinely Iceberg?

```
SELECT * FROM local.marts.fact_transaction.snapshots
  7609064981163123657  overwrite  total-records = 28583889

SHOW TBLPROPERTIES local.marts.fact_transaction
  format          iceberg/parquet
  format-version  2
  write.parquet.compression-codec  zstd
```

The `.snapshots` metadata table only resolves for a real Iceberg table. **PASS.**

**This also discharges step 1.4's pending half:** `fact_transaction` materialised
at **28,583,889 rows**, exactly the distinct-key count measured in 1.4, and
`assert_fact_transaction_grain_unique` passed on the full table (31.37 s).

Slowest tests on the real warehouse, for anyone budgeting the run:
grain uniqueness 31.37 s, customer relationships 14.24 s, article relationships
2.79 s. All referential-integrity tests pass, so every transaction's
`article_id` and `customer_id` really is present in the dimension files.

## Step 1.6 — Tests, and the second target

Seeds are generated by `src/marketrank/make_seeds.py` (committed, deterministic,
re-runnable via `make seeds`) rather than typed by hand, so they are real rows
with real edge cases:

```
seed_transactions.csv  355 rows
seed_customers.csv      50 rows
seed_articles.csv       47 rows
multi-quantity duplicate rows: 158
customers with null age: 1
```

Doc asked for ~200 / ~50 / ~50; 355 transactions is what falls out of taking 49
customers with 3–8 purchases each, and cutting to exactly 200 would have dropped
half the customers. **Read the seed's 44% multi-quantity rate as a fixture
property, not a dataset property** — it is a selection artifact of restricting to
the 50 most popular articles, and the real rate is 10.08% (step 1.4).

`tests/test_iceberg.py` builds its own five-line CSV in `tmp_path` and
monkeypatches `config.DATA_RAW`, so it drives the *real* write path
(declared-schema CSV read → `_ingested_at` stamp → `overwritePartitions`) with no
dependency on the 3.5 GB extract. That is what lets it run in CI. `ingest` gained
a `table=` parameter for the same reason — the test writes to a throwaway table
in a `test_tmp` namespace and drops it afterwards, rather than touching the
warehouse.

Three tests, not one:

1. `test_reload_is_idempotent` — the doc's test, driving two loads.
2. `test_whole_row_diff_still_sees_the_metadata_column` — asserts
   `assert_identical(..., ignore_cols=())` *raises*. Without it, the first test
   would keep passing if `_ingested_at` ever stopped varying, and would then be
   proving nothing. The step-1.3 checkpoint makes this pairing explicit, so it
   belongs in the suite rather than in a shell transcript.
3. `test_partition_overwrite_replaces_a_day_wholesale` — step 1.2's Think-first
   answer as an executable claim (load 3 rows, then load a 1-row "corrections"
   file for the same day, get 1 row). Week 9 has to change this behaviour, and
   this is the test that will go red when it does — deliberately.

**Checkpoint.**

```
make dbt-ci   ->  Done. PASS=19 WARN=0 ERROR=0 SKIP=0 TOTAL=19   in 4.5 s wall
pytest -m spark ->  3 passed in 11.05 s
```

Under a minute, with `MARKETRANK_DATA_RAW` never read. **PASS.**

## Step 1.7 — GitHub Actions

`.github/workflows/ci.yml`: `pull_request` → checkout → setup-python 3.11 →
setup-java 17 → `pip install -e ".[dev]"` → `dbt build --target ci` →
`pytest -v`. The Spark tests run (the doc's recommended option), because gate 1's
criterion is the PIT test and a job that skips it is not enforcing the gate.

**Deviation, and it is a real gap in this build: the workflow has never
executed.** The task brief forbids pushing to any remote, so no PR was opened,
no runner ran, and there is no green or red check anywhere. What was actually
done instead:

- The workflow's command sequence was run locally in the same order, from a
  shell where `MARKETRANK_DATA_RAW` was set but unused by either step.
- **"Watch it fail once"** was performed locally. `fact_transaction` was
  rewritten to grain (a) — one row per source row, the mistake step 1.4 rejects
  — and the CI target went red on the grain test:

  ```
  12 of 19 FAIL 98 assert_fact_transaction_grain_unique
           Got 98 results, configured to fail if != 0
  Done. PASS=18 WARN=0 ERROR=1 SKIP=0 TOTAL=19
  make exit status: 2
  ```

  Restoring the model returned it to `PASS=19 ERROR=0`. So the check *would* be
  red, and it is red for the right reason.
- **"The Spark tests actually ran rather than being collected and skipped"** was
  checked with `pytest -v`:

  ```
  tests/test_iceberg.py::test_reload_is_idempotent PASSED
  tests/test_iceberg.py::test_whole_row_diff_still_sees_the_metadata_column PASSED
  tests/test_iceberg.py::test_partition_overwrite_replaces_a_day_wholesale PASSED
  3 passed
  ```

  No SKIPPED lines. Note this only proves it on macOS with a Temurin 17 already
  installed; the `setup-java` step is unverified.

**Not run: anything involving GitHub.** No push, no PR, no badge. If you are
following these notes, this is the one step you have to do yourself, and the
first push is where `setup-java` and the runner's `pip install -e ".[dev]"` get
their first real test.

**PR discipline** is simulated with local branches: this week's steps 1.5–1.7
were committed on `feature/week1-dbt-ci` and merged into `build/implementation`
with `--no-ff`, so the merge structure exists in the history even though no PR
does.

---

## Step 2.0 — The dataset fact that shapes everything

Nothing to build. Both consequences are written into the README's limitations
list: the `[d − w, d − 1]` rule (date-only `t_dat`, so same-day events cannot be
ordered and are excluded entirely) and the current-state-snapshot dimensions.

## Step 2.1 — Write the leakage test first, and watch it fail

**Checkpoint: both tests fail with a clear message, committed failing.**

```
E  ImportError: cannot import name 'features' from 'marketrank'
   tests/test_pit.py:69: ImportError
E  ImportError: cannot import name 'features' from 'marketrank'
   tests/test_pit.py:90: ImportError
FAILED tests/test_pit.py::test_window_excludes_same_day
FAILED tests/test_pit.py::test_future_events_do_not_change_past_features
2 failed in 4.03s
```

**PASS** (in the sense the step means: they fail, for the right reason, and the
commit that adds them precedes the commit that adds `features.py`).

Implementation note: `from marketrank import features` is inside each test body,
not at module scope. At module scope the import error is a *collection* error —
pytest reports the file as broken rather than the two tests as failing, which is
a worse artifact for the history the step is trying to create.

**Deviation, and it is forced by this very step.** The doc specifies
`daily_customer_agg(spark)` in step 2.2 — a function that takes a session and
reads the warehouse. A function shaped like that cannot be driven by "a tiny
hand-built DataFrame", which is what step 2.1 requires. The two steps contradict
each other. Resolved in favour of 2.1: the aggregate and window functions take
**DataFrames**, and a separate `build_features(spark, ...)` is the only thing
that touches tables. That is also the shape that makes the backfill parameter of
step 2.4 natural, so the doc's own later step wants it too.
