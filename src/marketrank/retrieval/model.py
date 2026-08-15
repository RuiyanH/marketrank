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


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class ArticleTower(nn.Module):
    def __init__(self, vocab_sizes: dict[str, int], dim: int = EMB_DIM):
        super().__init__()
        self.id_emb = nn.Embedding(vocab_sizes["article_id"], dim, padding_idx=0)
        self.cat_embs = nn.ModuleList(
            [
                nn.Embedding(vocab_sizes[c], 16, padding_idx=0)
                for c in ds.ARTICLE_CATEGORICALS
            ]
        )
        width = dim + 16 * len(ds.ARTICLE_CATEGORICALS)
        self.mlp = nn.Sequential(
            nn.Linear(width, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, dim)
        )

    def forward(self, art_idx: torch.Tensor, cats: torch.Tensor) -> torch.Tensor:
        parts = [self.id_emb(art_idx)]
        for i, emb in enumerate(self.cat_embs):
            parts.append(emb(cats[:, i]))
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
    def __init__(self, vocab_sizes: dict[str, int], n_numeric: int, dim: int = EMB_DIM):
        super().__init__()
        self.article = ArticleTower(vocab_sizes, dim)
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
    return dict(article=art, recent=recent, age=age, cats=cats, numeric=numeric)


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
    dense_cats = np.zeros((size, cats.shape[1]), dtype=np.int64)
    dense_cats[idx] = cats
    dense_ids = [""] * size
    for i, a in zip(idx, ids):
        dense_ids[i] = a
    return np.arange(size, dtype=np.int64), dense_cats, dense_ids


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
):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    device = device or pick_device()

    tr = load_train(dirpath)
    art_idx_all, art_cats_all, _ = load_articles(dirpath)
    vs = vocab_sizes(dirpath)
    n_numeric = tr["numeric"].shape[1]

    model = TwoTower(vs, n_numeric, dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    # logQ correction. sampling_prob(a) is the article's empirical frequency
    # among the positives, which is exactly its probability of turning up as an
    # in-batch negative.
    counts = np.bincount(tr["article"], minlength=vs["article_id"]).astype(np.float64)
    probs = np.clip(counts / max(counts.sum(), 1), 1e-12, None)
    log_q = torch.tensor(np.log(probs), dtype=torch.float32, device=device)

    art_cats_t = torch.tensor(art_cats_all, device=device)
    art_ids_t = torch.tensor(art_idx_all, device=device)

    n = len(tr["article"])
    history = []
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
            v = model.article(arts, art_cats_t[arts])               # (B, d)

            logits = (u @ v.t()) / temperature                      # (B, B)
            if use_logq:
                # Subtract the log sampling probability of the COLUMN article,
                # AFTER temperature scaling -- the correction is in logit units,
                # not score units, and dividing it by T as well swamps the dot
                # products entirely (measured: loss 49 and recall 3.5%).
                # Without it the popular items in the batch are over-penalised
                # as negatives and the model systematically under-ranks them.
                logits = logits - log_q[arts].unsqueeze(0)

            target = torch.arange(len(b), device=device)
            loss = Fn.cross_entropy(logits, target)

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
            rec.update(recall_at(model, ev, truth, art_ids_t, art_cats_t, device=device))
            model.train()
        history.append(rec)
        if verbose:
            print("EPOCH", json.dumps(rec))
    return model, history, (art_ids_t, art_cats_t)


@torch.no_grad()
def article_matrix(model: TwoTower, art_ids_t, art_cats_t, chunk: int = 8192):
    model.eval()
    out = []
    for s in range(0, len(art_ids_t), chunk):
        out.append(model.article(art_ids_t[s : s + chunk], art_cats_t[s : s + chunk]))
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
) -> dict:
    """
    Exact top-N by inner product over the whole catalog, chunked over customers.

    Denominator is (customer, article) TRUE PAIRS, identical to the definition
    baselines.py uses, so the two numbers are comparable.
    """
    device = device or pick_device()
    V = article_matrix(model, art_ids_t, art_cats_t)             # (A, d)
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
