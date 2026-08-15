"""
The time-based split. Six disjoint slices, not four.

Three different things downstream each need data the others have not touched,
and a slice used twice tells you less than you think it does:

* `val_tune` vs `val_calib` -- fitting isotonic regression on the slice the
  ranker was early-stopped against makes measured ECE flatter itself. Cheap to
  avoid now, awkward to caveat later.
* `ope_env` -- week 8's reward model serves as ground truth, and if it is fit on
  the slice the ranker trained on, DM flatters itself and DR inherits the
  flattery through the residual term. Allocating the slice costs one line now;
  re-cutting data in week 8 means retraining the ranker to keep the boundaries
  honest.

`holdout` is the last 7 days so it mirrors the Kaggle test week's shape.

Nothing in weeks 3-7 may read `ope_env`. That is a rule, not a preference.
"""

SPLITS: dict[str, tuple[str, str]] = {
    "train":     ("2018-09-20", "2020-08-11"),
    "val_tune":  ("2020-08-12", "2020-08-25"),
    "val_calib": ("2020-08-26", "2020-09-01"),
    "ope_env":   ("2020-09-02", "2020-09-08"),
    "test":      ("2020-09-09", "2020-09-15"),
    "holdout":   ("2020-09-16", "2020-09-22"),
}

# Consumers, so a later week cannot quietly borrow a slice.
CONSUMED_BY = {
    "train":     "two-tower (wk 3), ranker (wk 5)",
    "val_tune":  "ranker early stopping / hyperparameters (wk 5); retrieval recall (wk 3)",
    "val_calib": "isotonic calibration fit (wk 5.3) -- nothing else",
    "ope_env":   "week 8's reward model / environment -- nothing else",
    "test":      "reported NDCG, AUC, revenue numbers",
    "holdout":   "local MAP@12 sanity check (wks 3-5)",
}

DATA_START = SPLITS["train"][0]
DATA_END = SPLITS["holdout"][1]


def bounds(name: str) -> tuple[str, str]:
    return SPLITS[name]


def sql_filter(name: str, col: str = "feature_date") -> str:
    lo, hi = SPLITS[name]
    return f"{col} between date'{lo}' and date'{hi}'"
