"""
The two towers.

Article tower: an id embedding plus embeddings for product type, colour group,
department, index group and garment group; concatenated, MLP'd to d, L2
normalised.

Customer tower: age bucket, club member status and newsletter frequency (all
current-state snapshot attributes, not PIT -- see step 2.0), plus the week-2
rolling features, plus **the mean of the embeddings of the customer's recent
articles**, taken from the PIT feature window.

Score is a dot product.

WHY THE ITEM-AVERAGE AND NOT A 1.37M-ROW CUSTOMER ID EMBEDDING -- three reasons,
and all three matter:

1. 1.37M x 128 is 175M parameters for a table where the median customer has 30
   distinct prior articles. It memorises and does not generalise.
2. It cannot serve a customer who was not in training, so cold start is
   unsolvable by construction.
3. The item-average updates the moment a customer buys something, with no
   retraining -- which is what makes the week-6 serving path honest.

SAMPLED SOFTMAX AND THE logQ CORRECTION. In-batch negatives means every other
positive in the batch acts as a negative. That is efficient and standard, and it
is **biased toward popularity**: a popular article appears in more batches, so it
is penalised as a negative more often and the model learns to under-rank it. The
fix is one term -- subtract log(sampling_prob(article)) from the logits before
the softmax. Three lines, and the single most interviewable detail in this week.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

from marketrank.retrieval import dataset as ds

EMB_DIM = 64
HIDDEN = 128


def sampled_softmax_logits(
    u: torch.Tensor,
    v: torch.Tensor,
    log_q: torch.Tensor,
    temperature: float,
    use_logq: bool = True,
) -> torch.Tensor:
    """
    Logits for one sampled-softmax batch: `(u @ v.T) / T - log_q`.

    THE ORDER OF THESE TWO OPERATIONS IS THE WHOLE FUNCTION. The logQ term is a
    correction in *logit* units, so it is subtracted AFTER temperature scaling.
    Writing `(u @ v.T - log_q) / T` divides the correction by T as well: with
    T = 0.05 and `log q ~ -11` that is a +230 offset which swamps the dot
    products entirely. It still trains and it still converges -- measured
    symptom was a loss of 49 and recall@500 of 3.5% (BUILD_NOTES step 3.2).

    `log_q` is per COLUMN (one entry per candidate article in the batch), so it
    broadcasts along the row axis. Extracted from `train()` so the placement is
    an executable claim rather than a comment -- see
    `tests/test_retrieval.py::test_logq_correction_is_applied_in_logit_units`.
    """
    logits = (u @ v.t()) / temperature
    if use_logq:
        logits = logits - log_q.unsqueeze(0)
    return logits


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class ArticleTower(nn.Module):
    """
    Id embedding + five static categorical embeddings, and from R.2 optionally
    the article's rolling volume features.

    `n_numeric > 0` is what makes the article vector TIME-VARYING, which is the
    whole point of R.2 and also its cost: the index has to be re-exported as of
    the scoring day rather than once ever.
    """

    def __init__(
        self, vocab_sizes: dict[str, int], dim: int = EMB_DIM, n_numeric: int = 0
    ):
        super().__init__()
        self.n_numeric = n_numeric
        self.id_emb = nn.Embedding(vocab_sizes["article_id"], dim, padding_idx=0)
        self.cat_embs = nn.ModuleList(
            [
                nn.Embedding(vocab_sizes[c], 16, padding_idx=0)
                for c in ds.ARTICLE_CATEGORICALS
            ]
        )
        width = dim + 16 * len(ds.ARTICLE_CATEGORICALS) + n_numeric
        self.mlp = nn.Sequential(
            nn.Linear(width, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, dim)
        )

    def forward(
        self,
        art_idx: torch.Tensor,
        cats: torch.Tensor,
        numeric: torch.Tensor | None = None,
    ) -> torch.Tensor:
        parts = [self.id_emb(art_idx)]
        for i, emb in enumerate(self.cat_embs):
            parts.append(emb(cats[:, i]))
        if self.n_numeric:
            if numeric is None:
                raise ValueError(
                    "article tower was built with volume features but none were "
                    "passed -- a silently zero-filled article vector is exactly "
                    "the failure R.2 exists to fix"
                )
            parts.append(numeric)
        return Fn.normalize(self.mlp(torch.cat(parts, dim=-1)), dim=-1)


class CustomerTower(nn.Module):
    def __init__(
        self,
        vocab_sizes: dict[str, int],
        n_numeric: int,
        article_id_emb: nn.Embedding,
        dim: int = EMB_DIM,
    ):
        super().__init__()
        # Shared with the article tower on purpose: the customer's recent-item
        # average has to live in the same space as the items it averages.
        self.article_id_emb = article_id_emb
        self.age_emb = nn.Embedding(len(ds.AGE_BUCKETS) + 2, 16, padding_idx=0)
        self.cat_embs = nn.ModuleList(
            [
                nn.Embedding(vocab_sizes[c], 16, padding_idx=0)
                for c in ds.CUSTOMER_CATEGORICALS
            ]
        )
        width = dim + 16 * (1 + len(ds.CUSTOMER_CATEGORICALS)) + n_numeric
        self.mlp = nn.Sequential(
            nn.Linear(width, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, dim)
        )

    def forward(
        self,
        recent: torch.Tensor,      # (B, K) padded with 0
        age: torch.Tensor,         # (B,)
        cats: torch.Tensor,        # (B, n_cat)
        numeric: torch.Tensor,     # (B, n_numeric)
    ) -> torch.Tensor:
        mask = (recent != 0).float().unsqueeze(-1)          # (B, K, 1)
        emb = self.article_id_emb(recent) * mask            # (B, K, d)
        denom = mask.sum(dim=1).clamp(min=1.0)
        item_avg = emb.sum(dim=1) / denom                   # (B, d)

        parts = [item_avg, self.age_emb(age)]
        for i, e in enumerate(self.cat_embs):
            parts.append(e(cats[:, i]))
        parts.append(numeric)
        return Fn.normalize(self.mlp(torch.cat(parts, dim=-1)), dim=-1)


class TwoTower(nn.Module):
    def __init__(
        self,
        vocab_sizes: dict[str, int],
        n_numeric: int,
        dim: int = EMB_DIM,
        n_article_numeric: int = 0,
    ):
        super().__init__()
        self.article = ArticleTower(vocab_sizes, dim, n_article_numeric)
        self.customer = CustomerTower(vocab_sizes, n_numeric, self.article.id_emb, dim)


# --------------------------------------------------------------------------
# Data loading -- parquet -> numpy, no pandas
# --------------------------------------------------------------------------


def _read_parquet(path: Path):
    import pyarrow.parquet as pq

    return pq.read_table(str(path))


def load_train(dirpath: Path = ds.DATASET_DIR):
    tbl = _read_parquet(dirpath / "train")
    n = tbl.num_rows
    art = tbl.column("article_idx").to_numpy(zero_copy_only=False).astype(np.int64)
    age = tbl.column("age_bucket").to_numpy(zero_copy_only=False).astype(np.int64)
    cats = np.stack(
        [
            tbl.column(f"{c}_idx").to_numpy(zero_copy_only=False).astype(np.int64)
            for c in ds.CUSTOMER_CATEGORICALS
        ],
        axis=1,
    )
    numeric = np.stack(
        [
            tbl.column(c).to_numpy(zero_copy_only=False).astype(np.float32)
            for c in ds.CUSTOMER_NUMERIC
        ],
        axis=1,
    )
    recent = np.zeros((n, ds.RECENT_K), dtype=np.int64)
    col = tbl.column("recent_articles").to_pylist()
    for i, lst in enumerate(col):
        if lst:
            k = min(len(lst), ds.RECENT_K)
            recent[i, :k] = lst[:k]
    out = dict(article=art, recent=recent, age=age, cats=cats, numeric=numeric)
    # day_index drives R.2's recency weighting; article volume is R.2's article
    # tower input. Both are optional so an older export still loads.
    if "day_index" in tbl.column_names:
        out["day_index"] = tbl.column("day_index").to_numpy(
            zero_copy_only=False
        ).astype(np.int64)
    if all(c in tbl.column_names for c in ds.ARTICLE_NUMERIC):
        out["art_numeric"] = np.stack(
            [
                tbl.column(c).to_numpy(zero_copy_only=False).astype(np.float32)
                for c in ds.ARTICLE_NUMERIC
            ],
            axis=1,
        )
    return out


def _dense_by_idx(idx: np.ndarray, values: np.ndarray, size: int) -> np.ndarray:
    """
    Scatter `values` into an array addressed BY `idx`, row 0 left zeroed.

    One function so the alignment rule has one home. `article_idx` starts at 1
    because 0 is the padding slot; a densely-packed array is off by one against
    every index in the data and makes recall approximately zero without raising.
    """
    dense = np.zeros((size, values.shape[1]), dtype=values.dtype)
    dense[idx] = values
    return dense


def load_articles(dirpath: Path = ds.DATASET_DIR):
    """
    Return catalog arrays **indexed by article_idx**, with row 0 reserved.

    This alignment is load-bearing and it is easy to get wrong: article_idx
    starts at 1 (0 is the padding/unknown slot), so a densely packed array of
    105,542 rows would be off by one against every index in the training data,
    and -- worse -- topk() would return positions that are silently one less
    than the article_idx the ground truth is keyed on. The bug does not raise;
    it just makes recall approximately zero.
    """
    tbl = _read_parquet(dirpath / "articles")
    idx = tbl.column("article_idx").to_numpy(zero_copy_only=False).astype(np.int64)
    cats = np.stack(
        [
            tbl.column(f"{c}_idx").to_numpy(zero_copy_only=False).astype(np.int64)
            for c in ds.ARTICLE_CATEGORICALS
        ],
        axis=1,
    )
    ids = tbl.column("article_id").to_pylist()
    size = int(idx.max()) + 1
    dense_cats = _dense_by_idx(idx, cats, size)
    dense_ids = [""] * size
    for i, a in zip(idx, ids):
        dense_ids[i] = a
    return np.arange(size, dtype=np.int64), dense_cats, dense_ids


def load_article_volume(dirpath: Path = ds.DATASET_DIR) -> np.ndarray | None:
    """
    The catalog's rolling volume features, **indexed by article_idx**, or None
    if this export predates R.2.

    Same alignment contract as `load_articles`, via the same helper: row 0 is
    the padding slot and holds zeros.
    """
    tbl = _read_parquet(dirpath / "articles")
    if not all(c in tbl.column_names for c in ds.ARTICLE_NUMERIC):
        return None
    idx = tbl.column("article_idx").to_numpy(zero_copy_only=False).astype(np.int64)
    vals = np.stack(
        [
            tbl.column(c).to_numpy(zero_copy_only=False).astype(np.float32)
            for c in ds.ARTICLE_NUMERIC
        ],
        axis=1,
    )
    return _dense_by_idx(idx, vals, int(idx.max()) + 1)


def load_eval(dirpath: Path = ds.DATASET_DIR):
    tbl = _read_parquet(dirpath / "eval_customers")
    cust = tbl.column("customer_id").to_pylist()
    n = tbl.num_rows
    age = tbl.column("age_bucket").to_numpy(zero_copy_only=False).astype(np.int64)
    cats = np.stack(
        [
            tbl.column(f"{c}_idx").to_numpy(zero_copy_only=False).astype(np.int64)
            for c in ds.CUSTOMER_CATEGORICALS
        ],
        axis=1,
    )
    numeric = np.stack(
        [
            tbl.column(c).to_numpy(zero_copy_only=False).astype(np.float32)
            for c in ds.CUSTOMER_NUMERIC
        ],
        axis=1,
    )
    recent = np.zeros((n, ds.RECENT_K), dtype=np.int64)
    for i, lst in enumerate(tbl.column("recent_articles").to_pylist()):
        if lst:
            k = min(len(lst), ds.RECENT_K)
            recent[i, :k] = lst[:k]

    truth = _read_parquet(dirpath / "eval_truth")
    pairs = {}
    for c, a in zip(
        truth.column("customer_id").to_pylist(),
        truth.column("article_idx").to_numpy(zero_copy_only=False),
    ):
        pairs.setdefault(c, set()).add(int(a))
    return dict(
        customer_id=cust, recent=recent, age=age, cats=cats, numeric=numeric
    ), pairs


def vocab_sizes(dirpath: Path = ds.DATASET_DIR) -> dict[str, int]:
    v = json.loads((dirpath / "vocabs.json").read_text())
    return {k: len(d) + 1 for k, d in v.items()}


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------


def train(
    dirpath: Path = ds.DATASET_DIR,
    epochs: int = 3,
    batch_size: int = 1024,
    lr: float = 3e-3,
    dim: int = EMB_DIM,
    use_logq: bool = True,
    temperature: float = 0.07,
    seed: int = 0,
    device: torch.device | None = None,
    verbose: bool = True,
    eval_each_epoch: tuple | None = None,
    checkpoint_path: Path | None = None,
    select_metric: str = "recall_at_100",
    n_uniform: int = 0,
    recency_half_life: float = 0.0,
    # Filled in by train() with things the caller must be able to record but
    # that do not belong in `history`. Out-param rather than a changed return
    # type, so existing callers keep working.
    stats_out: dict | None = None,
    article_volume: bool = False,
):
    """
    Train the two towers with in-batch sampled softmax.

    `checkpoint_path` saves the state dict of the BEST epoch by `select_metric`,
    not the last. Week 3 did neither -- it returned the final-epoch model and
    reported the best epoch's number, so `artifacts/twotower/model.pt` was
    epoch 7 (recall@100 5.481%) while the headline was epoch 4's 5.531%. The
    two are only 0.05 points apart here, which is exactly why it went unnoticed
    for a month. See BUILD_NOTES, recovery preamble.
    """
    # Accepted here so the CLI and the sbatch are stable across R.1-R.4, but a
    # parameter that is plumbed and inert is precisely the class of silent bug
    # this build keeps finding. Each one raises until its step implements it.
    if n_uniform and article_volume:
        # The open question R.2 recorded and never had to answer: a uniform
        # negative needs a volume vector, and all three options are wrong
        # (zero-fill is trivially separable; the catalog matrix is as of the
        # scoring day and leaks; its own event day can be LATER than the row's).
        # Article volume was dropped as a measured leak, so articles are static
        # and the question does not arise -- but it returns the moment anything
        # time-varying re-enters the article side, so refuse rather than pick
        # silently.
        raise NotImplementedError(
            "uniform negatives + article_volume is an unresolved design "
            "question -- see BUILD_NOTES step R.2/R.5 on the day-fingerprint leak"
        )

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    device = device or pick_device()

    tr = load_train(dirpath)
    art_idx_all, art_cats_all, _ = load_articles(dirpath)
    vs = vocab_sizes(dirpath)
    n_numeric = tr["numeric"].shape[1]

    # R.2 -- the article tower's volume features. The catalog-wide matrix is what
    # scoring uses; the per-row training values come from the train table, which
    # carries each positive's article volume as of ITS OWN day.
    art_vol_all = load_article_volume(dirpath) if article_volume else None
    if article_volume:
        if art_vol_all is None or "art_numeric" not in tr:
            raise ValueError(
                "article_volume=True but the export has no article volume "
                "columns -- re-run the export with --article-volume"
            )
    art_vol_t = (
        torch.tensor(art_vol_all, dtype=torch.float32, device=device)
        if art_vol_all is not None
        else None
    )
    n_art_numeric = art_vol_t.shape[1] if art_vol_t is not None else 0

    model = TwoTower(vs, n_numeric, dim, n_article_numeric=n_art_numeric).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    # R.2 -- recency weighting. A 2019 purchase should not pull the embeddings
    # as hard as last week's: weight = 0.5 ** (age_in_days / half_life),
    # normalised to mean 1 so the effective learning rate does not move with the
    # half-life and confound the ablation.
    if recency_half_life:
        if "day_index" not in tr:
            raise ValueError(
                "recency weighting needs day_index in the train export; re-export"
            )
        age_days = tr["day_index"].max() - tr["day_index"]
        w = np.power(0.5, age_days / float(recency_half_life)).astype(np.float32)
        w = w / w.mean()
        sample_w = torch.tensor(w, device=device)
        # EFFECTIVE SAMPLE SIZE (Kish): (sum w)^2 / sum(w^2). With w normalised
        # to mean 1 this is n / mean(w^2), and it is the number that decides
        # whether a wider training window actually added data. It can collapse
        # far below n without anything failing: at half_life=30 a row 700 days
        # old carries 2^-23, so widening the window to 23 months appends rows
        # that are numerically absent from the gradient. Reported so a scale
        # rung cannot claim n rows of evidence while training on a fraction.
        _p = np.percentile(w, [5, 50, 95, 99]).tolist()
        _ess = float(len(w) / np.mean(w.astype(np.float64) ** 2))
        recency_stats = {
            "half_life_days": float(recency_half_life),
            "n_rows": int(len(w)),
            "weight_p05": _p[0], "weight_p50": _p[1],
            "weight_p95": _p[2], "weight_p99": _p[3],
            "max_age_days": int(age_days.max()),
            "effective_sample_size": _ess,
            "ess_fraction": _ess / len(w),
        }
        if verbose:
            print(
                "RECENCY half_life=%.1f  weight p05/p50/p95 = %.4f / %.4f / %.4f"
                % (recency_half_life, *_p[:3])
            )
            print(
                "RECENCY n_rows %d  ESS %.0f  (%.1f%% of rows)"
                % (len(w), _ess, 100.0 * _ess / len(w))
            )
    else:
        sample_w = None
        recency_stats = {
            "half_life_days": 0.0,
            "n_rows": int(len(tr["article"])),
            # Unweighted: every row counts once, so ESS is n by definition.
            "effective_sample_size": float(len(tr["article"])),
            "ess_fraction": 1.0,
        }
    if stats_out is not None:
        stats_out["recency"] = recency_stats

    # logQ correction. sampling_prob(a) is the article's empirical frequency
    # among the positives, which is exactly its probability of turning up as an
    # in-batch negative.
    counts = np.bincount(tr["article"], minlength=vs["article_id"]).astype(np.float64)
    probs = np.clip(counts / max(counts.sum(), 1), 1e-12, None)
    log_q = torch.tensor(np.log(probs), dtype=torch.float32, device=device)
    # A uniform draw over the real catalog (slot 0 is padding, never sampled).
    log_q_uniform = torch.tensor(
        np.log(1.0 / max(vs["article_id"] - 1, 1)), dtype=torch.float32, device=device
    )

    art_cats_t = torch.tensor(art_cats_all, device=device)
    art_ids_t = torch.tensor(art_idx_all, device=device)

    n = len(tr["article"])
    history = []
    best = {"epoch": None, select_metric: float("-inf")}
    for ep in range(epochs):
        perm = rng.permutation(n)
        total, nb, t0 = 0.0, 0, time.time()
        for s in range(0, n - batch_size + 1, batch_size):
            b = perm[s : s + batch_size]
            recent = torch.tensor(tr["recent"][b], device=device)
            age = torch.tensor(tr["age"][b], device=device)
            cats = torch.tensor(tr["cats"][b], device=device)
            numeric = torch.tensor(tr["numeric"][b], device=device)
            arts = torch.tensor(tr["article"][b], device=device)

            u = model.customer(recent, age, cats, numeric)          # (B, d)
            # The positive's article volume is as of the EVENT's day, taken from
            # the train row -- not from the catalog matrix, which is as of the
            # scoring day and would leak the future into training.
            art_num = (
                torch.tensor(tr["art_numeric"][b], device=device)
                if article_volume
                else None
            )
            v = model.article(arts, art_cats_t[arts], art_num)      # (B, d)

            # Subtract the log sampling probability of the COLUMN article, after
            # temperature scaling. Without it the popular items in the batch are
            # over-penalised as negatives and the model systematically
            # under-ranks them (measured: 17x on recall@500). The placement of
            # the correction relative to T lives in sampled_softmax_logits.
            # R.3 -- MIXED NEGATIVES. In-batch negatives are drawn from the
            # POSITIVE distribution, so an article nobody buys is almost never a
            # negative and the model can inflate the whole tail without penalty.
            # Retrieval at k=100 is exactly where that bites. Uniform negatives
            # price the tail; their sampling probability is 1/|catalog|, which
            # is a different logQ term from the in-batch columns' empirical
            # frequency -- so the correction is per-column, not per-batch.
            v_cols, q_cols = v, log_q[arts]
            if n_uniform:
                neg = torch.as_tensor(
                    rng.integers(1, vs["article_id"], size=n_uniform),  # 0 is padding
                    device=device,
                )
                v_cols = torch.cat([v, model.article(neg, art_cats_t[neg], None)], 0)
                q_cols = torch.cat([q_cols, log_q_uniform.expand(n_uniform)], 0)

            logits = sampled_softmax_logits(          # (B, B + n_uniform)
                u, v_cols, q_cols, temperature, use_logq=use_logq
            )

            # The positive is still column i: uniform negatives are APPENDED, so
            # the diagonal is untouched and the target does not move.
            target = torch.arange(len(b), device=device)
            if sample_w is None:
                loss = Fn.cross_entropy(logits, target)
            else:
                per_row = Fn.cross_entropy(logits, target, reduction="none")
                bw = sample_w[torch.tensor(b, device=device)]
                loss = (per_row * bw).sum() / bw.sum()

            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.item())
            nb += 1
        rec = {"epoch": ep, "loss": total / max(nb, 1), "seconds": time.time() - t0}
        if eval_each_epoch is not None:
            # val_tune is the slice allocated for early stopping (splits.py), so
            # selecting the epoch on it is the intended use, not a leak.
            ev, truth = eval_each_epoch
            rec.update(
                recall_at(
                    model, ev, truth, art_ids_t, art_cats_t,
                    device=device, art_numeric=art_vol_t,
                )
            )
            model.train()
        history.append(rec)
        if verbose:
            print("EPOCH", json.dumps(rec))
        # Keep the BEST epoch, not the last one. val_tune is the slice splits.py
        # allocates for early stopping, so selecting on it is the intended use.
        if select_metric in rec and rec[select_metric] > best[select_metric]:
            best = {"epoch": ep, select_metric: rec[select_metric]}
            if checkpoint_path is not None:
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), checkpoint_path)
    if verbose and best["epoch"] is not None:
        print("BEST", json.dumps(best))
    return model, history, (art_ids_t, art_cats_t)


@torch.no_grad()
def article_matrix(model: TwoTower, art_ids_t, art_cats_t, chunk: int = 8192, art_numeric=None):
    model.eval()
    out = []
    for s in range(0, len(art_ids_t), chunk):
        sl = slice(s, s + chunk)
        num = art_numeric[sl] if art_numeric is not None else None
        out.append(model.article(art_ids_t[sl], art_cats_t[sl], num))
    return torch.cat(out, dim=0)


@torch.no_grad()
def customer_matrix(model: TwoTower, ev: dict, device, chunk: int = 4096):
    model.eval()
    out = []
    for s in range(0, len(ev["age"]), chunk):
        sl = slice(s, s + chunk)
        out.append(
            model.customer(
                torch.tensor(ev["recent"][sl], device=device),
                torch.tensor(ev["age"][sl], device=device),
                torch.tensor(ev["cats"][sl], device=device),
                torch.tensor(ev["numeric"][sl], device=device),
            )
        )
    return torch.cat(out, dim=0)


@torch.no_grad()
def recall_at(
    model: TwoTower,
    ev: dict,
    truth: dict,
    art_ids_t,
    art_cats_t,
    ns: tuple[int, ...] = (12, 100, 500),
    device: torch.device | None = None,
    chunk: int = 512,
    art_numeric=None,
) -> dict:
    """
    Exact top-N by inner product over the whole catalog, chunked over customers.

    Denominator is (customer, article) TRUE PAIRS, identical to the definition
    baselines.py uses, so the two numbers are comparable.
    """
    device = device or pick_device()
    V = article_matrix(model, art_ids_t, art_cats_t, art_numeric=art_numeric)  # (A, d)
    U = customer_matrix(model, ev, device)                       # (C, d)
    maxn = max(ns)
    hits = {n: 0 for n in ns}
    total = 0
    for s in range(0, U.shape[0], chunk):
        scores = U[s : s + chunk] @ V.t()
        scores[:, 0] = -1e9  # row 0 is the padding slot, not an article
        top = torch.topk(scores, maxn, dim=1).indices.cpu().numpy()
        for i, row in enumerate(top):
            cid = ev["customer_id"][s + i]
            want = truth.get(cid)
            if not want:
                continue
            total += len(want)
            for n in ns:
                hits[n] += len(want & set(row[:n].tolist()))
    return {"n_true_pairs": total, **{f"recall_at_{n}": hits[n] / total for n in ns}}
