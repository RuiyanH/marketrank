"""
Train the two towers and evaluate on the fixed `val_tune` cohort.

    python -m marketrank.jobs.train_towers --label r1_clean --epochs 8

The entrypoint `jobs/train_towers.sbatch` calls. Every run writes
`artifacts/twotower/runs/<label>/` containing `model.pt` (the BEST epoch, not
the last), `metrics.json` (the full per-epoch history plus the selected epoch)
and the exact argument vector it was run with, so a number in the ablation
ladder can always be traced back to the configuration that produced it.

**THE COHORT IS FIXED AND MUST STAY FIXED.** Every recall number in the recovery
ladder is on the identical 20,000 hash-sampled `val_tune` customers and the
identical 70,715 true (customer, article) pairs the reference build used. Change
the cohort or the denominator and the ladder stops being a comparison; the
`--expect-pairs` guard below fails the run rather than letting that happen
quietly.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from marketrank.retrieval import dataset as ds, model as M

RUNS_DIR = ds.DATASET_DIR / "runs"


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--label", required=True, help="run name; names the output dir")
    p.add_argument("--data", type=Path, default=ds.DATASET_DIR)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--dim", type=int, default=M.EMB_DIM)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--temperature", type=float, default=0.07)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-logq", action="store_true", help="ablation: drop the logQ term")
    p.add_argument(
        "--n-uniform",
        type=int,
        default=0,
        help="R.3: uniform-random negatives per batch, on top of in-batch",
    )
    p.add_argument(
        "--recency-half-life",
        type=float,
        default=0.0,
        help="R.2: positive sample-weight half-life in days; 0 disables weighting",
    )
    p.add_argument(
        "--article-volume",
        action="store_true",
        help="R.2: feed the article's rolling volume features into the article tower",
    )
    p.add_argument("--threads", type=int, default=0, help="torch intra-op threads")
    p.add_argument(
        "--expect-pairs",
        type=int,
        default=70_715,
        help="guard: fail unless the eval denominator is exactly this",
    )
    return p.parse_args(argv)


def _git_provenance() -> dict:
    """Commit and cleanliness of the tree that produced this run."""
    import subprocess

    def _run(*cmd):
        try:
            return subprocess.run(
                cmd, cwd=Path(__file__).resolve().parents[3],
                capture_output=True, text=True, timeout=10, check=True,
            ).stdout.strip()
        except Exception:
            return None

    sha = _run("git", "rev-parse", "HEAD")
    status = _run("git", "status", "--porcelain")
    return {
        "sha": sha,
        # None means the git call failed, which is NOT the same as clean.
        "dirty": None if status is None else bool(status),
    }


def main(argv=None) -> dict:
    args = parse_args(argv)
    if args.threads:
        torch.set_num_threads(args.threads)

    out = args.out or (RUNS_DIR / args.label)
    out.mkdir(parents=True, exist_ok=True)

    ev, truth = M.load_eval(args.data)
    n_pairs = sum(len(v) for v in truth.values())
    if args.expect_pairs and n_pairs != args.expect_pairs:
        raise SystemExit(
            f"eval denominator is {n_pairs}, expected {args.expect_pairs}. "
            "The cohort moved -- every number in the ablation ladder is measured "
            "against the same 20,000 customers / 70,715 pairs, so this run would "
            "not be comparable. Re-export, or pass --expect-pairs 0 deliberately."
        )

    t0 = time.time()
    train_stats: dict = {}
    model, history, arts = M.train(
        args.data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        dim=args.dim,
        use_logq=not args.no_logq,
        temperature=args.temperature,
        seed=args.seed,
        eval_each_epoch=(ev, truth),
        checkpoint_path=out / "model.pt",
        n_uniform=args.n_uniform,
        stats_out=train_stats,
        recency_half_life=args.recency_half_life,
        article_volume=args.article_volume,
    )
    seconds = time.time() - t0

    scored = [h for h in history if "recall_at_100" in h]
    best = max(scored, key=lambda h: h["recall_at_100"]) if scored else {}

    metrics = {
        "label": args.label,
        # PROVENANCE. candidate_ceiling.py has written a `run` block since R.5;
        # this job wrote none, so r3_uniform16/64/256 record no sha at all and
        # cannot be bound to the code that produced them after the fact. `dirty`
        # is the load-bearing half: a clean sha is a reproduction recipe, a
        # dirty one is a warning that the sha does not describe what ran.
        "code": _git_provenance(),
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "n_eval_customers": len(ev["customer_id"]),
        "n_true_pairs": n_pairs,
        "train_seconds": seconds,
        # `recency`: weight percentiles and Kish ESS. A scale rung must be
        # able to show how much of n actually reached the gradient.
        **train_stats,
        "history": history,
        "best": best,
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))

    print("TRAIN_SECONDS %.1f" % seconds)
    print("RESULT", json.dumps({"label": args.label, **best}))
    return metrics


if __name__ == "__main__":
    main()
