"""Walk-forward (expanding-window) validation.

The hidden evaluation set is Nov-Dec 2025 -- strictly AFTER all labeled data (Jan-Oct).
A random K-fold would leak the market regime of neighbouring days and be over-optimistic,
so every fold trains on the past only and tests on the next 2 months, mimicking the real task.

Folds (train -> test):
  Jan-Apr -> May-Jun | Jan-May -> Jun-Jul | Jan-Jun -> Jul-Aug | Jan-Jul -> Aug-Sep | Jan-Aug -> Sep-Oct

Metrics are reported on ALL rows and on "clean" rows (label not a spike). ~1.4% of labels are random
x3.4 / x0.28 spikes that no model can predict; they dominate RMSE but say nothing about model quality.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from features import build_features, daily_signals, load_csv
from models import HuberLinear, Hybrid, HybridRE, LGBM, RPMBaseline

warnings.filterwarnings("ignore")
DATA = Path(__file__).resolve().parents[1] / "data"
SPIKE_THRESHOLD = 0.5  # |log(rate / baseline)| above this => corrupted label

FOLDS = [("2025-05-01", "2025-06-30"), ("2025-06-01", "2025-07-31"), ("2025-07-01", "2025-08-31"),
         ("2025-08-01", "2025-09-30"), ("2025-09-01", "2025-10-31")]


def metrics(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    err = np.abs(p - y)
    return {"MAE": err.mean(), "MedAE": np.median(err), "MAPE%": 100 * np.mean(err / y),
            "MedAPE%": 100 * np.median(err / y), "WAPE%": 100 * err.sum() / y.sum(),
            "RMSE": np.sqrt(np.mean((p - y) ** 2))}


def load_all():
    tr = load_csv(DATA / "train_test.csv")
    va = load_csv(DATA / "validation.csv")
    daily = daily_signals(tr, va)
    X = build_features(tr, daily)
    return tr, X, daily


def walk_forward(models, tr, X, trim_outliers=True):
    y = np.log(tr["posted_rate"].values)
    rows = []
    for (t0, t1) in FOLDS:
        te = (tr["date"] >= t0) & (tr["date"] <= t1)
        trn = tr["date"] < t0
        # robust training set: drop labels the baseline flags as corrupt (>~5 sigma in log-space)
        base = RPMBaseline().fit(None, None, tr[trn])
        res = np.log(tr.loc[trn, "posted_rate"]) - np.log(base.predict(None, tr[trn]))
        keep = trn.copy()
        if trim_outliers:
            keep.loc[trn] = (res.abs() < SPIKE_THRESHOLD).values
        for m in models:
            m.fit(X[keep], y[keep], tr[keep])
            p = m.predict(X[te], tr[te])
            r = metrics(tr.loc[te, "posted_rate"], p)
            # same metrics on rows whose label is not a spike (|log-resid vs baseline| < 0.5) -> isolates model quality
            rb = np.log(tr.loc[te, "posted_rate"]) - np.log(base.predict(None, tr[te]))
            c = (rb.abs() < SPIKE_THRESHOLD).values
            rc = metrics(tr.loc[te, "posted_rate"][c], np.asarray(p)[c])
            r.update({"cMAE": rc["MAE"], "cMAPE%": rc["MAPE%"], "cMedAPE%": rc["MedAPE%"]})
            r.update(model=m.name, fold=f"{t0[:7]}..{t1[:7]}")
            rows.append(r)
    return pd.DataFrame(rows)


# feature groups --------------------------------------------------------------
BASE = ["log_distance", "distance", "equip_code", "weight", "weight_missing"]
MKT = ["market_index", "market_index_missing", "mi_day", "mi_day_c3", "mi_day_c7", "mi_day_chg7"]
QS = ["quote_signal", "qs_dev", "qs_day", "qs_day_c7"]
GEO = ["pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon", "hv", "dist_over_hv"]
GBM_FEATURES = BASE + MKT + QS + GEO                      # no day_index: trees cannot extrapolate a trend
LINEAR_COLS = ["mi_day", "qs_day", "day_index", "market_index", "quote_signal", "qs_dev"]


def final_model():
    # n_estimators=200: best in the hyper-parameter grid (src/compare.py); validation loss is flat after ~300 trees
    return HybridRE(LINEAR_COLS, GBM_FEATURES, "FINAL hybrid: robust-linear(trend)+lane RE+LightGBM", re_alpha=10, n_estimators=200)


if __name__ == "__main__":
    tr, X, daily = load_all()
    models = [
        RPMBaseline(),
        HuberLinear(["mi_day", "qs_day"], "huber_linear (no trend)"),
        HuberLinear(LINEAR_COLS, "huber_linear + trend"),
        LGBM(GBM_FEATURES, "lgbm (no trend)"),
        LGBM(GBM_FEATURES + ["day_index"], "lgbm + day_index"),
        Hybrid(LINEAR_COLS, GBM_FEATURES, "hybrid: linear(trend)+lgbm"),
        final_model(),
    ]
    res = walk_forward(models, tr, X)
    cols = ["MAE", "MedAE", "MAPE%", "MedAPE%", "WAPE%", "RMSE", "cMAE", "cMAPE%", "cMedAPE%"]
    summary = res.groupby("model")[cols].mean().round(3).sort_values("cMAPE%")
    per_fold = res.pivot(index="model", columns="fold", values="cMAPE%").round(2).loc[summary.index]
    out = Path(__file__).resolve().parents[1] / "outputs"
    out.mkdir(exist_ok=True)
    summary.to_csv(out / "cv_summary.csv")
    per_fold.to_csv(out / "cv_per_fold_cmape.csv")
    res.to_csv(out / "cv_all_folds.csv", index=False)
    pd.set_option("display.width", 220)
    print(summary.to_string())
    print("\nclean-MAPE% per fold")
    print(per_fold.to_string())
