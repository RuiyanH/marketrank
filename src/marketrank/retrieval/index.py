"""
The ANN index -- and an honest account of whether it is needed.

105k vectors at d=64 is 27 MB (at d=128, 54 MB). Exact inner-product search over
that is a single matrix multiply and takes single-digit milliseconds.
**ANN is not needed here for speed**, and claiming otherwise is the
manufactured-scale trap wearing a different hat -- the same move as claiming 32M
rows need Spark.

So this module builds both, measures both, and reports the recall-vs-latency
tradeoff at this catalog size. The defensible sentence is of the form "at 105k
items exact search is X ms and HNSW is Y ms at Z% recall -- the index earns its
place in the serving path's tail latency, not in its median", with X, Y and Z
measured. See BUILD_NOTES step 3.3 for the numbers this build got.
"""

from __future__ import annotations

import time

import numpy as np
import torch


def build_hnsw(vectors: np.ndarray, ef_construction: int = 200, m: int = 32):
    import hnswlib

    n, dim = vectors.shape
    idx = hnswlib.Index(space="ip", dim=dim)
    idx.init_index(max_elements=n, ef_construction=ef_construction, M=m)
    idx.add_items(vectors, np.arange(n))
    return idx


def exact_topk(vectors: torch.Tensor, queries: torch.Tensor, k: int) -> np.ndarray:
    scores = queries @ vectors.t()
    scores[:, 0] = -1e9  # row 0 is the padding slot, not an article
    return torch.topk(scores, k, dim=1).indices.cpu().numpy()


def hnsw_topk(idx, queries: np.ndarray, k: int, ef: int = 200) -> np.ndarray:
    idx.set_ef(max(ef, k))
    labels, _ = idx.knn_query(queries, k=k)
    return labels


def compare(
    vectors: torch.Tensor,
    queries: torch.Tensor,
    k: int = 100,
    efs: tuple[int, ...] = (100, 200, 400),
    repeats: int = 3,
) -> dict:
    """
    Exact vs HNSW: per-query latency and HNSW's recall against exact as truth.

    'Recall' here is index recall -- agreement with exact search -- not
    recommendation recall. They are different quantities and conflating them is
    how an ANN section stops meaning anything.
    """
    v_np = vectors.detach().cpu().numpy().astype(np.float32)
    q_np = queries.detach().cpu().numpy().astype(np.float32)
    nq = q_np.shape[0]

    lat = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        exact = exact_topk(vectors, queries, k)
        lat.append((time.perf_counter() - t0) / nq * 1000)
    out = {
        "n_vectors": int(v_np.shape[0]),
        "dim": int(v_np.shape[1]),
        "n_queries": int(nq),
        "k": k,
        "exact_ms_per_query": float(np.median(lat)),
        "vectors_mb": float(v_np.nbytes / 1e6),
    }

    t0 = time.perf_counter()
    idx = build_hnsw(v_np)
    out["hnsw_build_seconds"] = time.perf_counter() - t0

    exact_sets = [set(r.tolist()) for r in exact]
    for ef in efs:
        lat = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            approx = hnsw_topk(idx, q_np, k, ef=ef)
            lat.append((time.perf_counter() - t0) / nq * 1000)
        agree = sum(
            len(exact_sets[i] & set(approx[i].tolist())) for i in range(nq)
        ) / (nq * k)
        out[f"hnsw_ef{ef}_ms_per_query"] = float(np.median(lat))
        out[f"hnsw_ef{ef}_index_recall"] = float(agree)
    return out
