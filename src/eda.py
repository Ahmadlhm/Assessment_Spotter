from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from features import daily_signals, haversine_miles, load_csv
from models import RPMBaseline

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "eda"
OUT.mkdir(parents=True, exist_ok=True)
C = "#064A56"


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=140)
    plt.close(fig)


def main() -> None:
    tr = load_csv(ROOT / "data" / "train_test.csv")
    va = load_csv(ROOT / "data" / "validation.csv")
    tr["rpm"] = tr["posted_rate"] / tr["distance"]
    tr["month"] = tr["date"].dt.to_period("M")
    lines = []

    # ---- 1. shape, ranges, missing values, duplicates ------------------------------------
    lines += [f"train rows={len(tr):,} ({tr.date.min().date()} .. {tr.date.max().date()})",
              f"validation rows={len(va):,} ({va.date.min().date()} .. {va.date.max().date()})  -> hidden set is AFTER training period",
              f"duplicate load_id train={tr.load_id.duplicated().sum()} val={va.load_id.duplicated().sum()}"]
    shared = [c for c in va.columns if c in tr.columns]          # label is hidden in validation by design
    miss = pd.DataFrame({"train_missing": tr[shared].isna().sum(), "train_%": (tr[shared].isna().mean() * 100).round(2),
                         "val_missing": va[shared].isna().sum(), "val_%": (va[shared].isna().mean() * 100).round(2)})
    miss = miss[(miss.train_missing > 0) | (miss.val_missing > 0)]
    miss.to_csv(OUT / "missing_values.csv")
    lines.append("missing values:\n" + miss.to_string())
    tr.describe().T.round(3).to_csv(OUT / "describe_train.csv")

    # ---- 2. data-quality checks ------------------------------------------------------------
    lines += [f"negative weight: train={(tr.weight < 0).sum()} val={(va.weight < 0).sum()} (abs distribution matches positives -> sign flip)",
              f"weight == +-47,500 (cap): {(tr.weight.abs() == 47500).sum()} rows",
              f"cities in validation never seen in train: {sorted((set(va.pickup) | set(va.delivery)) - (set(tr.pickup) | set(tr.delivery)))}"]
    base = RPMBaseline().fit(None, None, tr)
    res = np.log(tr["posted_rate"]) - np.log(base.predict(None, tr))
    up, dn = (res > 0.5).sum(), (res < -0.5).sum()
    lines.append(f"label spikes |log-resid|>0.5: up={up} (x{np.exp(res[res > 0.5]).median():.2f}) "
                 f"down={dn} (x{np.exp(res[res < -0.5]).median():.2f}) = {(up + dn) / len(tr) * 100:.2f}% of rows")
    hv = haversine_miles(tr.pickup_lat, tr.pickup_lon, tr.delivery_lat, tr.delivery_lon)
    lines.append(f"distance / haversine median={np.median(tr.distance / hv):.3f} (distance is clean; short hauls naturally have larger ratio)")

    # ---- 3. figures ------------------------------------------------------------------------
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].hist(tr.posted_rate, bins=80, color=C); ax[0].set_title("posted_rate (skewed)")
    ax[1].hist(np.log(tr.posted_rate), bins=80, color=C); ax[1].set_title("log(posted_rate) -> modelling target")
    ax[2].hist(res.clip(-2, 2), bins=100, color=C); ax[2].axvline(.5, c="r", ls="--"); ax[2].axvline(-.5, c="r", ls="--")
    ax[2].set_title("log-residual vs baseline: ~1.4% label spikes (red)")
    save(fig, "01_target_and_spikes.png")

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    s = tr.sample(6000, random_state=0)
    for e, col in zip(["Dry Van", "Flatbed", "Reefer"], ["#064A56", "#E0813B", "#5B9E4D"]):
        q = s[s.equipment == e]
        ax[0].scatter(q.distance, q.posted_rate, s=3, alpha=.4, c=col, label=e)
    ax[0].set_xscale("log"); ax[0].set_yscale("log"); ax[0].legend(); ax[0].set_title("rate vs distance (log-log)")
    tr.boxplot(column="rpm", by="equipment", ax=ax[1], showfliers=False); ax[1].set_title("$ / mile by equipment"); plt.suptitle("")
    save(fig, "02_rate_vs_distance_equipment.png")

    d = daily_signals(tr, va).set_index("date")
    daily_rate = tr.groupby("date").apply(lambda g: np.median(np.log(g.rpm))).rename("log_rpm_median")
    fig, ax = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    ax[0].plot(d.index, d.mi_day, c=C); ax[0].axvline(pd.Timestamp("2025-11-01"), c="r", ls="--"); ax[0].set_title("market_index (daily mean) - red line = start of hidden set")
    ax[1].plot(d.index, d.qs_day, c="#E0813B"); ax[1].set_title("quote_signal (daily mean)")
    ax[2].plot(daily_rate.index, daily_rate.values, c="#5B9E4D"); ax[2].set_title("median log($/mile) per day - level moves with market_index + upward trend")
    save(fig, "03_daily_signals.png")

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].hist(tr.weight.dropna(), bins=60, color=C); ax[0].set_title("weight (neg. values + cap at 47,500)")
    ax[1].hist(tr.market_index.dropna(), bins=60, alpha=.6, label="train", color=C, density=True)
    ax[1].hist(va.market_index.dropna(), bins=60, alpha=.6, label="validation", color="#E0813B", density=True); ax[1].legend(); ax[1].set_title("market_index drift: train vs hidden set")
    ax[2].hist(tr.quote_signal, bins=60, color=C); ax[2].set_title("quote_signal")
    save(fig, "04_feature_distributions.png")

    num = tr[["posted_rate", "distance", "weight", "market_index", "quote_signal", "rpm"]].corr().round(3)
    num.to_csv(OUT / "correlations.csv")
    lines.append("correlations:\n" + num.to_string())
    lines.append("monthly stats:\n" + tr.groupby("month").agg(rows=("load_id", "size"), median_rate=("posted_rate", "median"),
                                                              median_rpm=("rpm", "median"), market_index=("market_index", "mean"),
                                                              quote_signal=("quote_signal", "mean")).round(3).to_string())
    (OUT / "eda_summary.txt").write_text("\n\n".join(lines))
    print("\n\n".join(lines))
    print(f"\nfigures + tables saved to {OUT}")


if __name__ == "__main__":
    main()
