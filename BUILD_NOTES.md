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
