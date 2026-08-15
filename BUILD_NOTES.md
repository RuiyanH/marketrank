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
