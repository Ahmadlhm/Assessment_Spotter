"""Model + hyper-parameter comparison, evaluation plots and tables  ->  outputs/comparison/

    python src/compare.py

Evaluation uses:
  error metrics     -> MAE, MedAE, MAPE, MedAPE, WAPE, RMSE (all rows and "clean" rows without label spikes)
  hit rate          -> share of predictions within +-2% / +-5% / +-10% of the true rate
  loss curves       -> Huber loss / MAE (log scale) per boosting iteration, train vs validation
  residual analysis -> predicted vs actual, % error distribution, bias over time
"""
from __future__ import annotations

import copy
import warnings
from pathlib import Path

import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from models import HuberLinear, Hybrid, HybridRE, LGBM, RPMBaseline
from validate import (FOLDS, GBM_FEATURES, LINEAR_COLS, SPIKE_THRESHOLD, load_all, metrics)

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "comparison"
OUT.mkdir(parents=True, exist_ok=True)
TEAL, ORANGE, GREEN, GREY = "#064A56", "#E0813B", "#5B9E4D", "#9AA7AB"
FINAL = "FINAL (RE a=10, 31 leaves, 200 trees)"
DEFAULT = "Hybrid-RE default (500 trees)"


# ---------------------------------------------------------------------------------- model grid
def build_grid():
    """(group, name, model, parameter description)"""
    g = []
    g.append(("benchmark", "Baseline: median $/mile", RPMBaseline(), "equipment x distance bucket"))
    g.append(("linear", "Huber linear (no trend)", HuberLinear(["mi_day", "qs_day"], "a"), "Huber loss, no time trend"))
    g.append(("linear", "Huber linear + trend", HuberLinear(LINEAR_COLS, "b"), "Huber loss + day_index"))
    for obj in ("regression", "l1", "huber"):
        lab = {"regression": "L2 (MSE)", "l1": "L1 (MAE)", "huber": "Huber"}[obj]
        g.append(("lgbm loss", f"LightGBM loss={lab}", LGBM(GBM_FEATURES, "c", objective=obj), f"objective={obj}, no trend"))
    g.append(("lgbm", "LightGBM + day_index", LGBM(GBM_FEATURES + ["day_index"], "d"), "trees given day_index"))
    g.append(("hybrid", "Hybrid: linear+LGBM (no lane RE)", Hybrid(LINEAR_COLS, GBM_FEATURES, "e"), "no random effects"))
    # --- final family, one parameter changed at a time
    for a in (1, 3, 30, 100):
        g.append(("RE alpha", f"Hybrid-RE alpha={a}", HybridRE(LINEAR_COLS, GBM_FEATURES, "f", re_alpha=a), f"re_alpha={a}, num_leaves=31, n_estimators=500"))
    for nl in (7, 15, 63):
        g.append(("num_leaves", f"Hybrid-RE leaves={nl}", HybridRE(LINEAR_COLS, GBM_FEATURES, "g", re_alpha=10, num_leaves=nl), f"re_alpha=10, num_leaves={nl}, n_estimators=500"))
    g.append(("n_estimators", "Hybrid-RE trees=1000", HybridRE(LINEAR_COLS, GBM_FEATURES, "h", re_alpha=10, n_estimators=1000), "re_alpha=10, num_leaves=31, n_estimators=1000"))
    g.append(("default", DEFAULT, HybridRE(LINEAR_COLS, GBM_FEATURES, "i", re_alpha=10), "re_alpha=10, num_leaves=31, n_estimators=500"))
    g.append(("final", FINAL, HybridRE(LINEAR_COLS, GBM_FEATURES, "k", re_alpha=10, n_estimators=200), "re_alpha=10, num_leaves=31, n_estimators=200"))
    nt = HybridRE(LINEAR_COLS, GBM_FEATURES, "j", re_alpha=10, n_estimators=200); nt.trim = False
    g.append(("ablation", "Final WITHOUT spike removal", nt, "labels not cleaned"))
    return g


# ---------------------------------------------------------------------------------- walk-forward with stored predictions
def run_walk_forward(grid, tr, X):
    y = np.log(tr["posted_rate"].values)
    rows, preds = [], {}
    for (t0, t1) in FOLDS:
        te = ((tr["date"] >= t0) & (tr["date"] <= t1)).values
        trn = (tr["date"] < t0).values
        base = RPMBaseline().fit(None, None, tr[trn])
        res_tr = y[trn] - np.log(base.predict(None, tr[trn]))
        keep_trim = trn.copy(); keep_trim[trn] = np.abs(res_tr) < SPIKE_THRESHOLD
        res_te = y[te] - np.log(base.predict(None, tr[te]))
        clean = np.abs(res_te) < SPIKE_THRESHOLD
        for group, name, m, desc in grid:
            keep = keep_trim if getattr(m, "trim", True) else trn
            m.fit(X[keep], y[keep], tr[keep])
            p = np.asarray(m.predict(X[te], tr[te]))
            actual = tr.loc[te, "posted_rate"].values
            r = metrics(actual, p)
            rc = metrics(actual[clean], p[clean])
            r.update({"cMAE": rc["MAE"], "cMAPE%": rc["MAPE%"], "cMedAPE%": rc["MedAPE%"], "cRMSE": rc["RMSE"],
                      "model": name, "group": group, "params": desc, "fold": f"{t0[:7]}..{t1[:7]}"})
            for tol in (2, 5, 10):
                r[f"within{tol}%"] = 100 * np.mean(np.abs(p - actual) / actual <= tol / 100)
                r[f"c_within{tol}%"] = 100 * np.mean(np.abs(p[clean] - actual[clean]) / actual[clean] <= tol / 100)
            rows.append(r)
            preds.setdefault(name, []).append(pd.DataFrame({"idx": np.where(te)[0], "pred": p, "actual": actual,
                                                            "clean": clean, "fold": f"{t0[:7]}..{t1[:7]}"}))
        print("fold done", t0, flush=True)
    return pd.DataFrame(rows), {k: pd.concat(v) for k, v in preds.items()}


# ---------------------------------------------------------------------------------- loss curves
def loss_curves(tr, X):
    """Stage-3 LightGBM loss per iteration (train Jan-Aug / validate Sep-Oct), for 3 tree sizes."""
    y = np.log(tr["posted_rate"].values)
    trn = (tr["date"] < "2025-09-01").values
    te = ~trn
    base = RPMBaseline().fit(None, None, tr[trn])
    keep = trn.copy(); keep[trn] = np.abs(y[trn] - np.log(base.predict(None, tr[trn]))) < SPIKE_THRESHOLD
    clean_te = np.abs(y[te] - np.log(base.predict(None, tr[te]))) < SPIKE_THRESHOLD
    stage12 = HybridRE(LINEAR_COLS, GBM_FEATURES, "lc", re_alpha=10, use_gbm=False).fit(X[keep], y[keep], tr[keep])
    r_tr = y[keep] - np.log(stage12.predict(X[keep], tr[keep]))
    r_te = (y[te] - np.log(stage12.predict(X[te], tr[te])))[clean_te]
    Xtr, Xte = X.loc[keep, GBM_FEATURES], X.loc[te, GBM_FEATURES][clean_te]
    curves = {}
    for nl in (7, 31, 63):
        m = lgb.LGBMRegressor(objective="huber", alpha=0.08, n_estimators=1500, learning_rate=0.03, num_leaves=nl,
                              min_child_samples=40, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                              verbose=-1, random_state=7)
        m.fit(Xtr, r_tr, eval_set=[(Xtr, r_tr), (Xte, r_te)], eval_metric=["huber", "l1"])
        ev = m.evals_result_
        curves[nl] = {"train_huber": ev["training"]["huber"], "val_huber": ev["valid_1"]["huber"],
                      "train_l1": ev["training"]["l1"], "val_l1": ev["valid_1"]["l1"]}
    return curves


# ---------------------------------------------------------------------------------- plotting
def plot_all(res, preds, curves, tr, X):
    summ = res.groupby(["model", "group", "params"])[["MAE", "MedAE", "MAPE%", "MedAPE%", "WAPE%", "RMSE", "cMAE", "cMAPE%", "cMedAPE%",
                                                      "within2%", "within5%", "within10%", "c_within2%", "c_within5%", "c_within10%"]].mean().reset_index()
    summ = summ.sort_values("cMAPE%").reset_index(drop=True)
    summ.round(3).to_csv(OUT / "model_comparison.csv", index=False)
    res.round(4).to_csv(OUT / "model_comparison_by_fold.csv", index=False)

    # 1. overall comparison --------------------------------------------------------------
    s = summ.iloc[::-1]
    colors = [ORANGE if m == FINAL else (GREY if g in ("benchmark",) else TEAL) for m, g in zip(s.model, s.group)]
    fig, ax = plt.subplots(1, 2, figsize=(15, 8), sharey=True)
    ax[0].barh(s.model, s["cMAPE%"], color=colors); ax[0].set_title("MAPE % on clean rows (lower = better)")
    ax[1].barh(s.model, s["MAPE%"], color=colors); ax[1].set_title("MAPE % on ALL rows (incl. unpredictable label spikes)")
    for a, col in zip(ax, ("cMAPE%", "MAPE%")):
        for i, v in enumerate(s[col]):
            a.text(v + .03, i, f"{v:.2f}", va="center", fontsize=8)
    fig.suptitle("Model & hyper-parameter comparison - mean of 5 walk-forward folds (orange = chosen model)", fontweight="bold")
    fig.tight_layout(); fig.savefig(OUT / "01_model_comparison.png", dpi=140); plt.close(fig)

    # 2. hyper-parameter sensitivity --------------------------------------------------------
    import re
    def sub(group, key_fn):
        extra = [DEFAULT] + ([FINAL] if group == "n_estimators" else [])
        rows = summ[(summ.group == group) | summ.model.isin(extra)]
        return sorted([(key_fn(r.params), r["cMAPE%"], r["MedAPE%"]) for _, r in rows.iterrows()])
    num = lambda key: (lambda p: float(re.search(key + r"=(\d+)", p).group(1)))
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    panels = [("RE alpha", num("re_alpha"), "random-effects ridge alpha (log scale)", True),
              ("num_leaves", num("num_leaves"), "LightGBM num_leaves", False),
              ("n_estimators", num("n_estimators"), "LightGBM n_estimators", False)]
    for a, (grp, fn, xl, lg) in zip(ax, panels):
        pts = sub(grp, fn)
        a.plot([p[0] for p in pts], [p[1] for p in pts], "o-", color=TEAL, lw=2)
        best = min(pts, key=lambda t: t[1]); a.scatter([best[0]], [best[1]], s=140, color=ORANGE, zorder=5, label="best")
        if lg: a.set_xscale("log")
        a.set_xlabel(xl); a.set_ylabel("clean MAPE %"); a.grid(alpha=.3); a.legend()
    fig.suptitle("Hyper-parameter sensitivity (one parameter changed at a time)", fontweight="bold")
    fig.tight_layout(); fig.savefig(OUT / "02_hyperparameter_sensitivity.png", dpi=140); plt.close(fig)

    # 3. per-fold stability ----------------------------------------------------------------------
    key = ["Baseline: median $/mile", "Huber linear + trend", "LightGBM loss=Huber", "Hybrid: linear+LGBM (no lane RE)", FINAL]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for k, c in zip(key, [GREY, GREEN, "#7B6FB5", "#4A90C2", ORANGE]):
        d = res[res.model == k]; ax.plot(d.fold, d["cMAPE%"], "o-", label=k, color=c, lw=3 if k == FINAL else 1.8)
    ax.set_ylabel("clean MAPE %"); ax.set_title("Stability across walk-forward folds (train on past, test on next 2 months)", fontweight="bold")
    ax.grid(alpha=.3); ax.legend(fontsize=8); plt.xticks(rotation=20)
    fig.tight_layout(); fig.savefig(OUT / "03_per_fold_stability.png", dpi=140); plt.close(fig)

    # 4. loss curves ------------------------------------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.5))
    for nl, c in zip((7, 31, 63), (GREEN, ORANGE, "#7B6FB5")):
        cv = curves[nl]
        ax[0].plot(cv["train_huber"], c=c, ls="--", lw=1.3); ax[0].plot(cv["val_huber"], c=c, lw=2, label=f"{nl} leaves")
        ax[1].plot(cv["train_l1"], c=c, ls="--", lw=1.3); ax[1].plot(cv["val_l1"], c=c, lw=2, label=f"{nl} leaves")
        bi = int(np.argmin(cv["val_huber"])); ax[0].scatter([bi], [cv["val_huber"][bi]], c=c, s=60, zorder=5)
    ax[0].axvline(200, c="k", ls=":", lw=1); ax[1].axvline(200, c="k", ls=":", lw=1)
    ax[0].set_title("Huber loss per iteration (dashed = train, solid = validation)"); ax[1].set_title("MAE of log-rate per iteration")
    for a in ax: a.set_xlabel("boosting iteration"); a.grid(alpha=.3); a.legend(); a.set_yscale("log")
    fig.suptitle("Loss curves - train Jan-Aug, validate Sep-Oct (dot = best iteration, dotted line = 200 trees used)", fontweight="bold")
    fig.tight_layout(); fig.savefig(OUT / "04_loss_curves.png", dpi=140); plt.close(fig)

    # 5. hit rate within tolerance ---------------------------------------------------------------------------
    tk = ["Baseline: median $/mile", "Huber linear + trend", "LightGBM loss=Huber", "Hybrid: linear+LGBM (no lane RE)", FINAL]
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.5), sharey=True)
    w = .16
    for panel, pre in zip(ax, ("c_", "")):
        for i, (tol, c) in enumerate(zip((2, 5, 10), (TEAL, GREEN, ORANGE))):
            vals = [summ.loc[summ.model == k, f"{pre}within{tol}%"].values[0] for k in tk]
            panel.bar(np.arange(len(tk)) + (i - 1) * w * 1.3, vals, w * 1.25, color=c, label=f"within +-{tol}%")
            for x, v in zip(np.arange(len(tk)) + (i - 1) * w * 1.3, vals):
                panel.text(x, v + 1, f"{v:.0f}", ha="center", fontsize=7)
        panel.set_xticks(range(len(tk))); panel.set_xticklabels([k.replace(" (", "\n(").replace(": ", ":\n") for k in tk], fontsize=7)
        panel.set_title("clean rows" if pre else "ALL rows (incl. spikes)"); panel.set_ylim(0, 105); panel.grid(axis="y", alpha=.3)
    ax[0].set_ylabel("% of predictions"); ax[0].legend(fontsize=8)
    fig.suptitle("Hit rate: % of predictions within +-2/5/10% of the true rate (mean of folds)", fontweight="bold")
    fig.tight_layout(); fig.savefig(OUT / "05_hit_rate_within_tolerance.png", dpi=140); plt.close(fig)

    # last walk-forward fold = pseudo test window (train Jan-Aug -> test Sep-Oct)
    last = FOLDS[-1]; lf = f"{last[0][:7]}..{last[1][:7]}"

    # 6. predicted vs actual + residuals -----------------------------------------------------------------------
    d = preds[FINAL]; d = d[d.fold == lf]
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
    c = d[d.clean]; s_ = d[~d.clean]
    ax[0].scatter(c.actual, c.pred, s=3, alpha=.35, c=TEAL, label="regular loads")
    ax[0].scatter(s_.actual, s_.pred, s=14, c="red", label="label spikes (unpredictable)")
    lim = [d.actual.min(), d.actual.max()]; ax[0].plot(lim, lim, "k--", lw=1)
    ax[0].set_xscale("log"); ax[0].set_yscale("log"); ax[0].set_xlabel("actual rate $"); ax[0].set_ylabel("predicted rate $"); ax[0].legend(markerscale=3, fontsize=8)
    ax[0].set_title("Predicted vs actual (log-log)")
    pe = 100 * (c.pred - c.actual) / c.actual
    ax[1].hist(pe.clip(-15, 15), bins=80, color=TEAL); ax[1].axvline(0, c="k", lw=1)
    ax[1].set_title(f"% error, regular loads\nmedian {pe.median():+.2f}% | 90% within +-{np.percentile(np.abs(pe), 90):.1f}%"); ax[1].set_xlabel("% error")
    ax[2].hist(np.log(d.pred / d.actual).clip(-1.8, 1.8), bins=120, color=ORANGE); ax[2].set_yscale("log"); ax[2].set_title("log(pred/actual), ALL rows (log y-axis):\ncentral peak + symmetric spikes (x3.4 / x0.28)")
    fig.suptitle("Final model on the Sep-Oct test window", fontweight="bold")
    fig.tight_layout(); fig.savefig(OUT / "06_predicted_vs_actual.png", dpi=140); plt.close(fig)

    # 7. why the time-trend matters: bias over time -------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 4.3))
    for k, col, lab in (("LightGBM loss=Huber", "#7B6FB5", "LightGBM only (no trend)"), (FINAL, ORANGE, "Final (linear trend + RE + LGBM)")):
        d = preds[k]; d = d[(d.fold == lf) & d.clean].copy()
        d["date"] = tr["date"].values[d["idx"].values]
        wk = (np.log(d.pred / d.actual)).groupby(d.date.dt.to_period("W")).median()
        ax.plot(wk.index.to_timestamp(), 100 * wk.values, "o-", color=col, lw=2, label=lab)
    ax.axhline(0, c="k", lw=1); ax.set_ylabel("median prediction error % (weekly)"); ax.grid(alpha=.3); ax.legend()
    ax.set_title("Bias over time, Sep-Oct: trees alone drift (cannot extrapolate the upward trend)", fontweight="bold")
    fig.tight_layout(); fig.savefig(OUT / "07_bias_over_time.png", dpi=140); plt.close(fig)
    return summ


if __name__ == "__main__":
    tr, X, _ = load_all()
    grid = build_grid()
    print(f"{len(grid)} models x {len(FOLDS)} folds", flush=True)
    res, preds = run_walk_forward(grid, tr, X)
    curves = loss_curves(tr, X)
    summ = plot_all(res, preds, curves, tr, X)
    pd.set_option("display.width", 250)
    print(summ[["model", "params", "MAE", "MAPE%", "MedAPE%", "cMAPE%", "within5%", "c_within5%"]].round(3).to_string())
    print("saved to", OUT)
