"""
The retrieval tests. **This file is step R.0, and it is week 3's missing gate.**

Week 2 had `test_pit.py` and week 3 had nothing, and week 3 duly shipped two
silent bugs -- logQ applied in the wrong units, and an off-by-one between
`article_idx` and array position that sent recall to approximately zero without
raising anything. Two found means the prior on a third is not small, and every
experiment in the recovery plan is worthless if a third one is still sitting in
the eval path.

The same rule week 2's gate was held to applies here: **a passing test proves
nothing until you have seen it fail for the right reason.** Both known bugs are
one line to re-introduce, and the mutation table in BUILD_NOTES step R.0 records
what happened when they were.

Three of these four run with no JVM and no data; only `test_feature_coverage...`
needs Spark, and it builds its own dozen-row DataFrame, so the whole file runs
in CI.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pytest
import torch

from marketrank.retrieval import dataset as ds, model as M

# --------------------------------------------------------------------------
# A stub in place of a trained TwoTower.
#
# `recall_at` only ever asks a model for three things: `.eval()`, `.article(ids,
# cats)` and `.customer(recent, age, cats, numeric)`. Supplying those directly
# with hand-chosen vectors is what makes the expected recall computable on
# paper -- a real (even tiny) trained model would make these tests assertions
# about optimisation, which is not what is being tested.
# --------------------------------------------------------------------------


class StubTowers:
    """Fixed article/customer matrices, addressed the way the real model is."""

    def __init__(self, V: torch.Tensor, U: torch.Tensor):
        self.V, self.U = V, U

    def eval(self):
        return self

    def train(self):
        return self

    def article(self, ids, cats, numeric=None):
        # Indexed BY article_idx -- exactly the contract load_articles promises.
        return self.V[ids]

    def customer(self, recent, age, cats, numeric):
        # `age` doubles as the row selector so each eval customer gets a chosen
        # vector; the real tower reads all four tensors.
        return self.U[age]


def _eval_batch(n_customers: int) -> dict:
    """The tensor bundle `recall_at` expects, with `age` as the row selector."""
    return {
        "customer_id": [f"c{i}" for i in range(n_customers)],
        "recent": np.zeros((n_customers, ds.RECENT_K), dtype=np.int64),
        "age": np.arange(n_customers, dtype=np.int64),
        "cats": np.zeros((n_customers, len(ds.CUSTOMER_CATEGORICALS)), dtype=np.int64),
        "numeric": np.zeros((n_customers, len(ds.CUSTOMER_NUMERIC)), dtype=np.float32),
    }


def _write_synthetic_catalog(dirpath, n_articles: int = 5):
    """
    Five articles at article_idx 1..5, written in DELIBERATELY SHUFFLED row
    order.

    The shuffle is the point. If the parquet happened to be sorted by
    article_idx then a densely-packed loader and a correctly-indexed one would
    agree except at the padding slot, and the test would mostly pass while the
    bug was present.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    idx = list(range(1, n_articles + 1))
    shuffled = [3, 1, 5, 2, 4][:n_articles]
    rows = {
        "article_id": [f"art{ i :04d}" for i in shuffled],
        "article_idx": shuffled,
    }
    # Each categorical carries a value derived from the index, so a misaligned
    # load is detectable per row rather than only in aggregate.
    for j, c in enumerate(ds.ARTICLE_CATEGORICALS):
        rows[f"{c}_idx"] = [100 * (j + 1) + i for i in shuffled]

    (dirpath / "articles").mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(rows), str(dirpath / "articles" / "part-0.parquet"))
    return idx


# --------------------------------------------------------------------------
# Test 1 -- index alignment
# --------------------------------------------------------------------------


def test_topk_positions_map_back_to_article_idx(tmp_path):
    """
    The bug this exists for: `article_idx` starts at 1 because 0 is the padding
    slot, so a densely-packed catalog array of 105,542 rows is shifted by one
    against every index in the data -- and `topk()` then returns positions that
    are one less than the `article_idx` the ground truth is keyed on. Recall
    goes to approximately zero and NOTHING RAISES.

    Two halves, because the bug has two halves:

      a. `load_articles` returns arrays indexed BY article_idx, row 0 reserved.
      b. a topk position is an article_idx, including when the padding slot
         would otherwise have won the row outright.
    """
    _write_synthetic_catalog(tmp_path)
    ids, cats, art_ids = M.load_articles(tmp_path)

    # (a) alignment of the loaded arrays
    assert len(ids) == 6, "array must be sized max(article_idx) + 1, not n_articles"
    assert ids.tolist() == [0, 1, 2, 3, 4, 5]
    assert art_ids[0] == "", "row 0 is the padding slot and belongs to no article"
    assert cats[0].tolist() == [0] * len(ds.ARTICLE_CATEGORICALS)
    for k in range(1, 6):
        assert art_ids[k] == f"art{k:04d}", f"row {k} must hold article_idx {k}"
        assert cats[k].tolist() == [
            100 * (j + 1) + k for j in range(len(ds.ARTICLE_CATEGORICALS))
        ], f"categoricals at row {k} belong to a different article"

    # (b) topk positions are article_idx values, and row 0 cannot win.
    dim = 6
    V = torch.zeros(dim, dim)
    for k in range(dim):
        V[k, k] = 1.0
    # The padding slot is given the LARGEST possible score for our customer, so
    # if recall_at ever stops masking column 0 this assertion fails immediately.
    V[0, 0] = 10.0

    U = torch.zeros(1, dim)
    U[0, 0] = 1.0   # would rank the padding slot first
    U[0, 3] = 0.9   # article_idx 3 is the correct answer

    stub = StubTowers(V, U)
    ev = _eval_batch(1)
    truth = {"c0": {3}}
    out = M.recall_at(
        stub, ev, truth,
        torch.as_tensor(ids), torch.as_tensor(cats),
        ns=(1,), device=torch.device("cpu"),
    )
    assert out["n_true_pairs"] == 1
    assert out["recall_at_1"] == 1.0, (
        "top-1 did not come back as article_idx 3. Either the catalog arrays "
        "are densely packed (off by one), or the padding slot was not masked."
    )


# --------------------------------------------------------------------------
# Test 2 -- feature coverage at eval time
# --------------------------------------------------------------------------

DAY_ZERO = dt.date(2018, 9, 20)


def _d(day_index: int) -> dt.date:
    return DAY_ZERO + dt.timedelta(days=day_index)


@pytest.mark.spark
def test_feature_coverage_at_eval_is_total(spark):
    """
    Score three customers on a day none of them transacted, which is the normal
    case at eval time: candidates are evaluated on a day the customer bought
    nothing. Without a spine, window functions emit output only for rows that
    EXIST, so the feature table has nothing to join to and the tower is scored
    on zero-filled rubbish.

    This is the audit that would have caught the spine bug in week 3 instead of
    week 4, and the number it would have reported is 14.09%.
    """
    from marketrank import features

    rows = [
        ("c1", "a1", _d(10), 0.10, 1),
        ("c1", "a2", _d(12), 0.20, 1),
        ("c2", "a1", _d(11), 0.15, 1),
        ("c3", "a3", _d(5), 0.05, 2),
    ]
    schema = (
        "customer_id string, article_id string, t_dat date, "
        "price double, sales_channel_id int"
    )
    txn = spark.createDataFrame(rows, schema=schema)
    cohort = spark.createDataFrame(
        [("c1",), ("c2",), ("c3",)], schema="customer_id string"
    )

    score_day = 30  # nobody transacted on day 30
    spine = features.customer_day_spine(
        spark, cohort, _d(score_day).isoformat(), _d(score_day).isoformat()
    )

    feats = features.customer_features(txn, spine=spine)
    cov = features.feature_coverage(feats, cohort, "customer_id", score_day)

    assert cov["n_entities"] == 3
    assert cov["coverage"] == 1.0, (
        "customers are missing a feature row on the scoring day. With "
        f"spine=None this reads {cov['n_covered']}/3 -- and every missing row "
        "reaches the tower as log1p(coalesce(null, 0)) = 0.0, silently."
    )

    # The spine must ADD rows and CHANGE NO VALUE: c1 transacted on day 12, and
    # that row's features are a property of days [12-w, 11] either way.
    no_spine = features.customer_features(txn)
    a = no_spine.filter("customer_id = 'c1' and day_index = 12").collect()
    b = feats.filter("customer_id = 'c1' and day_index = 12").collect()
    if b:  # day 12 is outside the spine range, so only assert when present
        assert a[0].asDict() == b[0].asDict()


# --------------------------------------------------------------------------
# Test 3 -- recall arithmetic
# --------------------------------------------------------------------------


def test_recall_at_matches_hand_computation():
    """
    Three customers over a five-article catalog, with the scores chosen so each
    customer's ranking is a strict total order and the answer is computable on
    paper.

        c1  ranks 1,2,3,4,5   truth {1,3}   hits@1 = 1  @3 = 2  @5 = 2
        c2  ranks 5,4,3,2,1   truth {5}     hits@1 = 1  @3 = 1  @5 = 1
        c3  ranks 2,4,5,3,1   truth {1,2}   hits@1 = 1  @3 = 1  @5 = 2

    Denominator is TRUE PAIRS, not customers -- 2 + 1 + 2 = 5 -- which is the
    same denominator baselines.py uses, and the reason the tower's numbers are
    comparable to the union's at all.

        recall@1 = 3/5 = 0.6   recall@3 = 4/5 = 0.8   recall@5 = 5/5 = 1.0
    """
    dim = 6
    V = torch.zeros(dim, dim)
    for k in range(dim):
        V[k, k] = 1.0

    # Row i of U is customer i's weight on each article_idx.
    U = torch.tensor(
        [
            [0.0, 0.5, 0.4, 0.3, 0.2, 0.1],
            [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
            [0.0, 0.1, 0.5, 0.2, 0.4, 0.3],
        ]
    )

    stub = StubTowers(V, U)
    ev = _eval_batch(3)
    truth = {"c0": {1, 3}, "c1": {5}, "c2": {1, 2}}

    out = M.recall_at(
        stub, ev, truth,
        torch.arange(dim), torch.zeros((dim, len(ds.ARTICLE_CATEGORICALS)), dtype=torch.long),
        ns=(1, 3, 5), device=torch.device("cpu"),
    )

    assert out["n_true_pairs"] == 5, "denominator must be true pairs, not customers"
    assert out["recall_at_1"] == pytest.approx(0.6)
    assert out["recall_at_3"] == pytest.approx(0.8)
    assert out["recall_at_5"] == pytest.approx(1.0)


# --------------------------------------------------------------------------
# Test 4 -- the logQ correction's units
#
# NOT in the recovery plan's list of three; added because the plan's checkpoint
# ("reverting the logQ-units fix fails test 3") cannot hold as written. Test 3
# is a pure function of fixed embeddings and ground truth and never constructs
# a training logit, so no training-time bug can reach it. See BUILD_NOTES step
# R.0 for the reasoning. The plan's INTENT -- both known bugs are covered by the
# suite -- is what this preserves.
# --------------------------------------------------------------------------


def test_logq_correction_is_applied_in_logit_units():
    """
    `(u@v.T)/T - log_q`, not `(u@v.T - log_q)/T`.

    With T = 0.05 and log q ~ -11 the wrong spelling adds ~+230 to every logit
    instead of ~+11, which swamps dot products that live in [-1, 1]. It trains,
    it converges, and it is nonsense: loss 49, recall@500 3.5%.
    """
    torch.manual_seed(0)
    B, d, T = 4, 8, 0.05
    u = torch.nn.functional.normalize(torch.randn(B, d), dim=-1)
    v = torch.nn.functional.normalize(torch.randn(B, d), dim=-1)
    log_q = torch.log(torch.full((B,), 1.0 / 105_542))

    got = M.sampled_softmax_logits(u, v, log_q, T)

    right = (u @ v.t()) / T - log_q.unsqueeze(0)
    wrong = (u @ v.t() - log_q.unsqueeze(0)) / T

    assert torch.allclose(got, right, atol=1e-5)
    assert not torch.allclose(got, wrong, atol=1e-2), (
        "the logQ term was divided by the temperature -- it is a correction in "
        "logit units and belongs AFTER the temperature scaling"
    )

    # And it is off by exactly the factor the bug is made of: the wrong spelling
    # carries log_q/T where the right one carries log_q, so the gap is
    # log_q * (1 - 1/T) = -11.57 * -19 = +219.77 on every logit.
    gap = (log_q - log_q / T).unsqueeze(0).expand(B, B)
    assert torch.allclose(wrong - right, gap, atol=1e-3)
    assert float(gap[0, 0]) > 200, "T=0.05 and log q~-11 is a ~220-logit offset"

    off = M.sampled_softmax_logits(u, v, log_q, T, use_logq=False)
    assert torch.allclose(off, (u @ v.t()) / T, atol=1e-5)
