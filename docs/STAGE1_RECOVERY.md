# Stage-1 recovery plan — making retrieval earn its place

*Written 2026-08-15, against the measured results of the reference build
(`build/implementation`, `BUILD_NOTES.md`). This is the continuation of
[`IMPLEMENTATION.md`](IMPLEMENTATION.md) after step 3.2's checkpoint failed; it
replaces nothing in that document — it inserts between week 3 and week 5.
Same five-part step format, same rules: checkpoints are observable facts, and
anything marked **VERIFY** was not run.*

**The situation.** The two-tower lost to the baseline union — recall@100 of
**5.531% vs 6.967%** on the identical cohort (20,000 `val_tune` customers,
70,715 true pairs), 21% worse, losing at every cutoff. The candidate-set recall
ceiling is **7.475%**, so week 5 is blocked: a ranker trained on this set would
be ordering candidates that usually contain nothing worth ordering.

**The verdict available from the data: none yet.** The measurement cannot
distinguish "two-towers don't work here" from three cheaper explanations, each
pointed at by the build's own numbers:

1. **A known bug sits upstream of the eval.** The features were built without a
   spine; step 4.2 measured **85.6%** of candidate rows joining to NULL customer
   features. The tower's customer side consumes those same features, scored on a
   day most customers have no feature row for. Whether 3.2's eval was
   contaminated is *unknown*, which is the problem.
2. **The tower was judged against information it wasn't given.** The popularity
   baseline sees recent transaction volume. The article tower is an id plus five
   *static* categorical embeddings — structurally blind to "trending now," on a
   fast-fashion dataset. The build measured the same story twice more: exact-article
   repurchase ceiling **3.36%** vs **64.34%** at `product_type_no` (short-horizon
   signal is trend, not identity), and global popularity beating personalized
   dominant-category popularity at equal depth (**3.168% vs 2.162%**).
3. **The one lever measured moved the result 17×, and it was the negative
   sampler** (logQ ablation: 11.22% vs 0.67% recall@500 — with the *worse* model
   holding the *better* training loss). When one knob on the negative
   distribution is worth 17×, the prior belongs on the rest of that knob.

Note what is *not* on the list: raw scale. The per-epoch curve plateaus at epoch
4, a 20-epoch run on less data was worse (overfitting), and the reduced run
already trained on the most recent — most relevant — six months. Scale is worth
doing (step R.4), but it is the weakest of the four hypotheses and goes last.

**Budget: the tower gets one time-boxed week** (R.0–R.4). R.5 runs in parallel
and de-risks the schedule no matter how the tower story ends. R.6 is the
decision, and both of its outcomes are shippable.

---

## Step R.0 — Tests before compute

**Why now.** Week 3 produced two silent bugs (logQ applied in the wrong units;
an off-by-one between `article_idx` and array position that sent recall to ~0
without raising). Two found means the prior on a third is not small — and week 3
has no tests where week 2 had its leakage pair. Every experiment below is
worthless if a third silent bug is still in the eval path.

**Write** `tests/test_retrieval.py`, three assertions:

1. **Index alignment** — a synthetic catalog of 5 articles; assert that
   `topk()` positions map back to the exact `article_idx` values the ground
   truth is keyed on, including the row-0 padding slot.
2. **Feature coverage at eval** — after hydrating the eval cohort's customer
   features, assert non-null coverage ≈ 100%. This is the audit that would have
   caught the spine bug in week 3 instead of week 4.
3. **Recall sanity** — a hand-built 3-customer case where the correct
   recall@k is computable on paper; assert it exactly.

**Must know.** These are the week-3 equivalent of week 2's mutation check: a
passing test proves nothing until you've seen it fail for the right reason. Both
known bugs are one line to re-introduce; re-introduce them.

**Checkpoint.** Reverting the logQ-units fix fails test 3; reverting the
alignment fix fails test 1. Restored, all three pass. That table goes in the
notes, same as gate 1's.

## Step R.1 — Kill the confound: spine rebuild, then the identical eval

**Why now.** Cheapest information in the plan: zero new ideas, and every later
number is uninterpretable without it.

**Write.** Nothing new — run the already-plumbed spine fix at full scale
(`build_features` with the spine; the reduced-scale proof showed coverage
14.09% → 100% with zero value changes), then re-run 3.2's evaluation **bit-for-bit
unchanged**: same checkpointed model, same 20,000-customer cohort, same
denominator. Then retrain once on the clean features, same reduced config.

**Must know.** Two numbers come out, and they answer different questions.
Re-eval of the *old model* on clean features isolates eval-time contamination;
the *retrain* isolates training-time contamination. Record both next to 5.531%.
Resist the urge to change anything else in the same run — this step is a
measurement, not an improvement.

**Checkpoint.** Feature coverage on the eval cohort is ~100% (R.0's test 2
green at full scale), and both numbers are written down. Whatever they are, the
remaining gap to 6.967% is now attributable to the model rather than the data.

## Step R.2 — Information parity: give the tower what the baseline sees

**Why now.** Highest-prior fix. The baseline's entire advantage is knowing
recent volume; the week-2 article aggregates (`art_n_txn_7d`, `art_n_txn_30d`,
`art_n_customers_7d`, …) already compute exactly that and were never wired into
the article tower. The features exist; this is plumbing, not research.

**Write.**

1. Concatenate the article's rolling aggregates (log-scaled) into the article
   tower input, PIT-stamped as of `d−1` like everything else.
2. **Recency-weight the training positives** — sample weight decaying with age,
   half-life a hyperparameter (start ~30 days). A 2019 purchase should not pull
   embeddings as hard as last week's.

**Must know.** This changes what the article vector *is*: it stops being a
static description and becomes time-varying, which means the article index must
be re-exported as of the eval date rather than once ever. That's the honest cost
of trend-awareness, and it's the same cost the serving path (week 6) inherits —
note it now. The PIT rule carries: the aggregates entering the tower for a day-`d`
event are the `[d−w, d−1]` ones. Week 2's leakage test does not cover the tower's
inputs; **VERIFY** by truncation (train-time feature export at day `D` vs full)
if in doubt.

**Checkpoint.** Recall@100 on the identical cohort, one change at a time:
R.1's clean baseline → +article-volume features → +recency weighting. Three
numbers, additive story.

## Step R.3 — The rest of the negatives lever

**Why now.** The 17× logQ result is the largest effect size in the build. In-batch
negatives — even logQ-corrected — draw only from the *positive* distribution:
the model never sees the long tail as negatives at all. Mixed negative sampling
(uniform-random negatives alongside in-batch, both logQ-corrected with their own
sampling probabilities) is the standard fix.

**Write.** Add `n_uniform` random-article negatives per batch row to the
in-batch set; the logQ term for a uniform negative is `log(1/105542)`, for an
in-batch negative the empirical frequency, as now. `n_uniform` ∈ {16, 64, 256}.

**Must know.** Why this specifically: with in-batch-only, an obscure article is
almost never a negative, so the model can inflate scores on the whole tail
without penalty — and retrieval at k=100 is precisely where tail inflation
bites. Uniform negatives price the tail. The ablation table (with/without, per
`n_uniform`) is this step's artifact, the way the logQ table was week 3's.

**Checkpoint.** Same cohort, same cutoffs, one row per configuration. Keep the
best; record all.

### RESULT (2026-08-18) — null, and R.4 carries 16 anyway

| n_uniform | r@100 | Δ vs 0 |
|---|---|---|
| 0 (`r2_recency`) | 6.659% | — |
| **16** | **6.710%** | **+0.051** |
| 64 | 6.451% | −0.208 |
| 256 | 6.608% | −0.051 |

**Noise floor (R.1b, 3 seeds, one config): spread 0.051.** The best rung clears
baseline by exactly that, and the response is not monotone. `r@500` — the
sharper test, since pricing the tail should surface at deep k first — agrees:
+0.191, −0.048, +0.013 across a 16× range. Mixed negatives do not help this
model on this dataset.

**"Keep the best; record all" is applied literally**: R.4 carries
`n_uniform=16`. It is the nominal best, directionally positive everywhere, and
free at GPU scale, and the tail-inflation mechanism has more room on the full
catalog than on this cohort. **Its delta is inside the noise floor, and R.4's
writeup must say so** — any R.4 gain belongs to scale until measured otherwise.
A second full-scale run to ablate it is explicitly not worth its cost.

**The logQ specification above was right.** This build raised a pre-registered
objection that a mixed proposal needs `log(p·freq(a) + (1−p)/N)` rather than
each component's own probability, and blocked recording the null until it was
tested. That objection is **retracted**: the implementation appends uniform
columns rather than merging the pools, so each column is an independent draw
corrected with its own distribution — precisely what this section specified.
The mixture form applies to a design where one column could come from either
pool. Detail in `BUILD_NOTES.md` R.3.

**Provenance note.** `train_towers.py` wrote no sha, so all three R.3 rungs
record nothing about the code that produced them and cannot be bound to it
retroactively. Fixed forward — metrics now carry `code.sha` and `code.dirty` —
but the three rungs stay unbound, recorded rather than repaired.

## Step R.4 — Scale, last and honestly

**Why now (last).** Because the plateau and the overfitting result say scale is
the weakest hypothesis. It goes after R.1–R.3 so its contribution is measured on
top of the fixes, not confounded with them.

**Write.** `jobs/train_towers.sbatch` on misha (account confirmed through
~2027-08): full `train`-slice positives with recency weighting, full 1.37M
customer cohort, d=128, early stopping on `val_tune`. Same eval cohort as ever
for comparability, plus the full-cohort number.

**Plumbing.** The sbatch follows `jobs/train_ranker.sbatch`'s conventions:
`--time` explicit, `mkdir -p logs` before submit, `OMP_NUM_THREADS` from the
allocation, git commit echoed into the log. CPU-node training at this scale is
**VERIFY** — measure one epoch's wall clock before committing to the full run.

**Checkpoint.** The final ablation ladder, R.1 → R.4, one line each, on the
identical cohort. This table *is* the week's deliverable regardless of R.6's
outcome.

## Step R.5 — Ceiling raisers that don't need the tower (parallel track)

**Why now.** These de-risk the schedule: they raise the stage-1 ceiling whether
or not the tower recovers, and they're cheap.

**Write.**

1. **Global recent popularity as a fourth candidate source** — already measured:
   ceiling 7.475% → **9.083%** for 27 extra candidates. Apply it (the reference
   build left it as an experiment; `candidates.py` still implements three
   sources).
2. **Item-item co-visitation** — `covisit.py`: for article pairs bought by the
   same customer within a short window (same day, then ≤7 days), a
   time-decayed co-occurrence count; candidates = top co-visited articles of the
   customer's recent purchases. This is the workhorse of the actual
   competition's top solutions, it's a few Spark joins, and it captures the
   sequential/trend structure this dataset measurably rewards.
3. Re-measure the union ceiling with each source's **marginal contribution at
   fixed slot budget** (~130 candidates/customer), not just solo coverage.

**Must know.** The sources are nearly disjoint (mean sources per candidate
1.033), so marginal ≈ solo *today* — but that stops being true as sources are
added, which is why the marginal-at-fixed-budget number is the one to track.
Slots are the budget; a source pays rent in marginal ceiling per slot.

**Checkpoint.** A source-contribution table at the fixed budget, and a new
ceiling. Target: **comfortably above 12%** before week 5 restarts —
2.5–3× the ranker's room to work versus today's 7.5%. **VERIFY** that
co-visitation delivers at H&M scale; its solo coverage is the number to find out.

### RESULT (2026-08-18) — 7.475% → **11.930%**, gate not cleared

Co-visitation delivers: it is the largest single contributor at 4.805% solo and
the second-most efficient source at 0.0779%/slot. The **VERIFY** above is
answered yes.

| config | covisit reach | covisit solo | union ceiling |
|---|---|---|---|
| committed 30/20, depth 20/40 | 45.9% | 2.803% | 10.777% |
| depth probe 40/60 (R.5c) | 45.9% | 3.501% | 11.176% |
| misha 60/50 (R.5d) | 66.7% | 4.470% | 11.761% |
| misha 90/50 (R.5d) | **74.4%** | **4.805%** | **11.930%** |

Detail and reasoning in `BUILD_NOTES.md` R.5c/R.5d. Three things this changes:

1. **The target is missed by 0.070 points**, and the lever that got here is
   nearly spent — per *added* slot, the last step returned 0.0469%, below `ann`
   and level with `global_pop`. A 120/50 run projects to ~12.0%: touching the
   gate, not comfortably above it, at the worst efficiency yet measured.
2. **These are deterministic measurements.** R.1b's noise floor applies to
   two-tower training, not to a fixed-cohort ceiling with no sampling. No seed
   replication is owed here.
3. **The binding constraint has moved.** R.5 was written to de-risk the schedule
   "whether or not the tower recovers". It has: +1.153 points over the committed
   configuration. The largest remaining headroom is `ann` — 4.156% solo from the
   tower that lost to popularity, with its repair specified in R.3 and R.4 and
   not yet run. Continuing to squeeze covisit is worse value than fixing the
   source with a known defect.

**Recommendation carried into R.6:** treat 11.930% as R.5's final number, do not
spend further runs on covisit bounds, and re-measure the ceiling once R.3/R.4
have produced a new ANN parquet. `max_basket`'s independent contribution is
confounded inside the 60/50 step and remains unmeasured; a 60/20 run would
separate it, and is not on the critical path.

## Step R.6 — The decision gate

**Think first.** The original checkpoint — "the tower beats the baseline union
solo" — conflates two roles. Which role is stage 1 actually hiring for?

<details>
<summary>Answer</summary>

A *source in an ensemble*, not a sole retriever. The union is the product;
sources are judged by **marginal ceiling per candidate slot at the fixed
budget**. On that metric the tower was already near parity with global
popularity before any fixes (3.329%/50 slots vs 3.168%/40). "Beat the union
solo" was the right bar for shipping the tower *as the retrieval story*; it is
the wrong bar for keeping it *as one of four sources*.
</details>

**The rule, set now, before R.1–R.4's numbers exist** (that's what makes it a
rule): after the time-boxed week,

- **Keep the tower as a source** if its marginal-ceiling-per-slot ≥ the weakest
  heuristic source's. Report the ablation ladder as the retrieval story.
- **Demote or drop it** if not — co-visitation and popularity carry stage 1, and
  the writeup reports the three-way diagnosis and the measured decision. The
  logQ table stays in the README either way; it's the best single artifact
  week 3 produced.
- **Not permitted:** a bigger tower, more architecture, or continued tuning past
  the box. Nothing in the data says capacity is the constraint, and the project's
  thesis — the decision layer and its evaluation — is waiting on a working
  candidate set, not on a retrieval trophy.

**Checkpoint.** The decision, written in the README with its numbers, and week
5 unblocked with a ceiling the ranker can live under.

---

## Re-entry into IMPLEMENTATION.md

Week 5 resumes as written, with three amendments carried from the reference
build's findings:

1. Candidates are regenerated from the post-R.5 source set; step 4.2's audits
   rerun, including the leakage re-test **plus the row-set assertion** (PIT row
   *existence* is a separate property from PIT *values* — the build caught a
   changed row set with zero value mismatches).
2. Money measures are **DECIMAL end to end** (the build proved DOUBLE summation
   is partition-order-dependent; 23,758 rows differed between a build and a
   backfill of the same window, which makes step 2.4's bit-identical checkpoint
   impossible by construction).
3. The ranker's feature list picks up the candidate-source tags from the
   *expanded* source set — with four-plus sources the tag is a stronger feature
   than it was with three.
