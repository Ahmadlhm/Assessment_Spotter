from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from features import build_features, city_coordinates, daily_signals, load_csv
from models import RPMBaseline
from validate import SPIKE_THRESHOLD, final_model

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def build_december_frame(dec: pd.DataFrame, coords: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    """december_chart_inputs.csv only carries 6 columns. Fill the rest the same way the model saw them:
       * lat/lon from the city lookup learned on train+validation,
       * market_index / quote_signal = the real daily levels observed in validation.csv for that date
         (these are FEATURES of Dec loads, available at prediction time -- no label is used)."""
    d = dec.copy()
    d["date"] = pd.to_datetime(d["date"])
    for side in ("pickup", "delivery"):
        d[f"{side}_lat"] = d[side].map(coords["lat"])
        d[f"{side}_lon"] = d[side].map(coords["lon"])
    sig = daily.set_index("date")
    d["market_index"] = d["date"].map(sig["mi_day"])
    d["quote_signal"] = d["date"].map(sig["qs_day"])
    assert d[["pickup_lat", "delivery_lat", "market_index", "quote_signal"]].notna().all().all()
    return d


def main() -> None:
    tr = load_csv(DATA / "train_test.csv")
    va = load_csv(DATA / "validation.csv")
    tmpl = pd.read_csv(DATA / "validation_predictions_template.csv")
    dec_raw = pd.read_csv(DATA / "december_chart_inputs.csv")

    daily = daily_signals(tr, va)          # features only -> no label leakage
    coords = city_coordinates(tr, va)

    X_tr, X_va = build_features(tr, daily), build_features(va, daily)
    y = np.log(tr["posted_rate"].values)

    # drop corrupted labels (random x3.4 / x0.28 spikes) from the TRAINING set only
    base = RPMBaseline().fit(None, None, tr)
    resid = y - np.log(base.predict(None, tr))
    keep = np.abs(resid) < SPIKE_THRESHOLD
    print(f"training rows: {keep.sum():,} / {len(tr):,}  (dropped {int((~keep).sum())} spike labels)")

    model = final_model().fit(X_tr[keep], y[keep], tr[keep])

    # ---- validation predictions (exact template order / ids)
    pred = model.predict(X_va, va)
    out = pd.DataFrame({"load_id": va["load_id"], "predicted_rate": np.round(pred, 2)})
    out = tmpl[["load_id"]].merge(out, on="load_id", how="left", validate="one_to_one")
    assert len(out) == 12_000 and out["predicted_rate"].notna().all() and (out["predicted_rate"] > 0).all()
    out.to_csv(ROOT / "validation_predictions.csv", index=False)

    # ---- December chart predictions
    dec = build_december_frame(dec_raw, coords, daily)
    dec_pred = model.predict(build_features(dec, daily), dec)
    dec_out = dec_raw.copy()
    dec_out["predicted_rate"] = np.round(dec_pred, 2)
    dec_out.to_csv(DATA / "december_chart_inputs.csv", index=False)

    (ROOT / "models").mkdir(exist_ok=True)
    joblib.dump(model, ROOT / "models" / "final_model.joblib")

    print("validation_predictions.csv written:", out.shape)
    print(out["predicted_rate"].describe().round(2).to_string())
    print("\nDecember predictions:")
    print(dec_out[["date", "predicted_rate"]].to_string(index=False))


if __name__ == "__main__":
    main()
