"""
Export a trained tower's top-N articles per evaluation customer as a candidate
source.

    python -m marketrank.jobs.ann_candidates --run r4_scale --top-n 50

Writes `(customer_id, article_id, source, source_rank)` parquet, which
`candidates.union_candidates` consumes exactly like the heuristic sources. This
is the bridge between the tower and R.5's ceiling table, and it is what lets
R.6's decision be made on the tower's **marginal ceiling per candidate slot**
rather than on whether it beats the baseline union solo.

Exact inner-product search over the whole catalog, not HNSW. Step 3.3 measured
the index at 105k items: exact is 0.147 ms/query batched and HNSW is 0.0323 ms
at 98.6% index recall, so the index buys tail latency in a serving path and
nothing at all in an offline export -- where using it would only inject a 1.4%
disagreement into a recall measurement for no benefit.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from marketrank.candidates import SOURCE_ANN
from marketrank.retrieval import dataset as ds, model as M


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True, help="run label under artifacts/twotower/runs")
    p.add_argument("--data", type=Path, default=ds.DATASET_DIR)
    p.add_argument("--top-n", type=int, default=50)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--chunk", type=int, default=512)
    return p.parse_args(argv)


def main(argv=None) -> Path:
    import pyarrow as pa
    import pyarrow.parquet as pq

    args = parse_args(argv)
    run_dir = args.data / "runs" / args.run
    metrics = json.loads((run_dir / "metrics.json").read_text())
    cfg = metrics["args"]

    device = M.pick_device()
    ev, _ = M.load_eval(args.data)
    vs = M.vocab_sizes(args.data)
    art_ids, art_cats, art_id_strings = M.load_articles(args.data)
    art_vol = M.load_article_volume(args.data) if cfg.get("article_volume") else None

    n_art_numeric = art_vol.shape[1] if art_vol is not None else 0
    net = M.TwoTower(
        vs, ev["numeric"].shape[1], cfg["dim"], n_article_numeric=n_art_numeric
    ).to(device)
    net.load_state_dict(torch.load(run_dir / "model.pt", map_location=device))
    net.eval()

    art_ids_t = torch.as_tensor(art_ids, device=device)
    art_cats_t = torch.as_tensor(art_cats, device=device)
    art_vol_t = (
        torch.tensor(art_vol, dtype=torch.float32, device=device)
        if art_vol is not None
        else None
    )

    V = M.article_matrix(net, art_ids_t, art_cats_t, art_numeric=art_vol_t)
    U = M.customer_matrix(net, ev, device)

    cust_out: list[str] = []
    art_out: list[str] = []
    rank_out: list[int] = []
    with torch.no_grad():
        for s in range(0, U.shape[0], args.chunk):
            scores = U[s : s + args.chunk] @ V.t()
            scores[:, 0] = -1e9  # padding slot is not an article
            top = torch.topk(scores, args.top_n, dim=1).indices.cpu().numpy()
            for i, row in enumerate(top):
                cid = ev["customer_id"][s + i]
                for r, aidx in enumerate(row.tolist(), start=1):
                    cust_out.append(cid)
                    art_out.append(art_id_strings[aidx])
                    rank_out.append(r)

    out = args.out or (run_dir / "ann_candidates.parquet")
    pq.write_table(
        pa.table(
            {
                "customer_id": cust_out,
                "article_id": art_out,
                "source": [SOURCE_ANN] * len(cust_out),
                "source_rank": np.array(rank_out, dtype=np.int32),
            }
        ),
        str(out),
    )
    print(f"ANN_CANDIDATES rows {len(cust_out)} customers {U.shape[0]} -> {out}")
    return out


if __name__ == "__main__":
    main()
