from __future__ import annotations

import numpy as np
import pandas as pd

WEIGHT_CAP = 47_500.0
EQUIPMENT = ["Dry Van", "Flatbed", "Reefer"]


def haversine_miles(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 3958.8 * 2 * np.arcsin(np.sqrt(a))


def load_csv(path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Row-level cleaning. Never touches the label."""
    df = df.copy()
    df["weight_was_negative"] = (df["weight"] < 0).astype(int)
    df["weight"] = df["weight"].abs().clip(upper=WEIGHT_CAP)
    df["weight_missing"] = df["weight"].isna().astype(int)
    df["market_index_missing"] = df["market_index"].isna().astype(int) if "market_index" in df else 0
    return df


def city_coordinates(*frames: pd.DataFrame) -> pd.DataFrame:
    """city -> (lat, lon) lookup built from every frame that carries coordinates."""
    parts = []
    for f in frames:
        parts.append(f[["pickup", "pickup_lat", "pickup_lon"]].set_axis(["city", "lat", "lon"], axis=1))
        parts.append(f[["delivery", "delivery_lat", "delivery_lon"]].set_axis(["city", "lat", "lon"], axis=1))
    return pd.concat(parts).groupby("city")[["lat", "lon"]].first()


def daily_signals(*frames: pd.DataFrame) -> pd.DataFrame:
    """Date-level aggregates of market_index / quote_signal (features only, no labels)."""
    all_rows = pd.concat([f[["date", "market_index", "quote_signal"]] for f in frames])
    d = all_rows.groupby("date").agg(mi_day=("market_index", "mean"), qs_day=("quote_signal", "mean"))
    d = d.asfreq("D").interpolate(limit_direction="both")
    d["mi_day_c3"] = d["mi_day"].rolling(3, center=True, min_periods=1).mean()
    d["mi_day_c7"] = d["mi_day"].rolling(7, center=True, min_periods=1).mean()
    d["mi_day_chg7"] = d["mi_day"] - d["mi_day"].shift(7)
    d["qs_day_c7"] = d["qs_day"].rolling(7, center=True, min_periods=1).mean()
    return d.reset_index()


def build_features(df: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    """Return the model matrix (numeric + equipment dummies kept as category codes)."""
    df = clean(df).merge(daily, on="date", how="left")
    out = pd.DataFrame(index=df.index)
    out["log_distance"] = np.log(df["distance"])
    out["distance"] = df["distance"]
    out["equip_code"] = df["equipment"].map({e: i for i, e in enumerate(EQUIPMENT)}).astype(int)
    out["weight"] = df["weight"]
    out["weight_missing"] = df["weight_missing"]
    # row-level market_index, falling back to the daily level when missing
    out["market_index"] = df["market_index"].fillna(df["mi_day"])
    out["market_index_missing"] = df["market_index_missing"]
    out["quote_signal"] = df["quote_signal"]
    out["qs_dev"] = (df["quote_signal"] - 2.05).abs()
    for c in ["mi_day", "mi_day_c3", "mi_day_c7", "mi_day_chg7", "qs_day", "qs_day_c7"]:
        out[c] = df[c]
    hv = haversine_miles(df["pickup_lat"], df["pickup_lon"], df["delivery_lat"], df["delivery_lon"])
    out["hv"] = hv
    out["dist_over_hv"] = df["distance"] / hv.clip(lower=1)
    for c in ["pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"]:
        out[c] = df[c]
    out["dow"] = df["date"].dt.dayofweek
    out["day_index"] = (df["date"] - pd.Timestamp("2025-01-01")).dt.days  # used only by linear trend variants
    return out
