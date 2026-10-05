from __future__ import annotations

import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import HuberRegressor
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

class RPMBaseline:
    """Median $/mile by equipment x distance bucket (the 'dumb but honest' benchmark)."""
    name = "baseline_median_rpm"
    bins = [0, 150, 300, 600, 900, 1200, 1800, 2500, 10_000]

    def fit(self, X, y, raw):
        b = pd.cut(raw["distance"], self.bins)
        self.t = (raw["posted_rate"] / raw["distance"]).groupby([raw["equipment"], b]).median()
        return self

    def predict(self, X, raw):
        b = pd.cut(raw["distance"], self.bins)
        idx = pd.MultiIndex.from_arrays([raw["equipment"], b])
        return (self.t.reindex(idx).values * raw["distance"].values)


class HuberLinear:
    """Robust linear model on log(rate)."""
    def __init__(self, cols, name="huber_linear"):
        self.cols, self.name = cols, name

    def _design(self, X):
        D = pd.DataFrame(index=X.index)
        D["ld"], D["ld2"] = X["log_distance"], X["log_distance"] ** 2
        for k in (1, 2):
            D[f"eq{k}"] = (X["equip_code"] == k).astype(float)
            D[f"eq{k}_ld"] = D[f"eq{k}"] * X["log_distance"]
        D["w"] = X["weight"].fillna(31_000) / 10_000
        D["w2"] = D["w"] ** 2
        D["w_miss"] = X["weight_missing"]
        for c in self.cols:
            D[c] = X[c]
        return D

    def fit(self, X, y, raw):
        D = self._design(X)
        self.sc = StandardScaler().fit(D)
        self.m = HuberRegressor(epsilon=1.35, alpha=1e-4, max_iter=500).fit(self.sc.transform(D), y)
        return self

    def predict(self, X, raw):
        return np.exp(self.m.predict(self.sc.transform(self._design(X))))


class LGBM:
    def __init__(self, features, name, objective="huber", **params):
        self.features, self.name, self.objective = features, name, objective
        self.params = dict(n_estimators=700, learning_rate=0.03, num_leaves=31, min_child_samples=40,
                           subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                           verbose=-1, random_state=7)
        self.params.update(params)

    def fit(self, X, y, raw):
        kw = {"objective": self.objective}
        if self.objective == "huber":
            kw["alpha"] = 0.15  # in log space ~ 15% -> spikes (>2x) get linear loss
        self.m = lgb.LGBMRegressor(**kw, **self.params).fit(X[self.features], y)
        return self

    def predict(self, X, raw):
        return np.exp(self.m.predict(X[self.features]))


class Hybrid:
    """Stage 1: robust linear model (captures distance/equipment/weight + market level + TIME TREND, extrapolates).
       Stage 2: LightGBM on the stage-1 residual (geo/lane structure, interactions). Trees never see day_index."""
    def __init__(self, lin_cols, gbm_features, name, alpha=0.08, **gbm_params):
        self.lin = HuberLinear(lin_cols, name + "_lin")
        self.f, self.name, self.alpha = gbm_features, name, alpha
        self.params = dict(n_estimators=600, learning_rate=0.03, num_leaves=31, min_child_samples=40,
                           subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                           verbose=-1, random_state=7)
        self.params.update(gbm_params)

    def fit(self, X, y, raw):
        self.lin.fit(X, y, raw)
        r = y - np.log(self.lin.predict(X, raw))
        self.m = lgb.LGBMRegressor(objective="huber", alpha=self.alpha, **self.params).fit(X[self.f], r)
        return self

    def predict(self, X, raw):
        return self.lin.predict(X, raw) * np.exp(self.m.predict(X[self.f]))


class HybridRE:
    """Stage 1: robust linear (market level + TIME TREND + distance/equipment/weight).
       Stage 2: ridge random effects on pickup / delivery / directed lane / undirected pair (shrunk; unseen city -> 0).
       Stage 3: LightGBM on what is left (geo, interactions). All stages predict log-rate corrections."""
    def __init__(self, lin_cols, gbm_features, name, re_alpha=10.0, use_gbm=True, gbm_alpha=0.08, **gbm_params):
        self.lin = HuberLinear(lin_cols, name + "_lin")
        self.f, self.name, self.re_alpha, self.use_gbm, self.gbm_alpha = gbm_features, name, re_alpha, use_gbm, gbm_alpha
        self.params = dict(n_estimators=500, learning_rate=0.03, num_leaves=31, min_child_samples=40,
                           subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                           verbose=-1, random_state=7)
        self.params.update(gbm_params)

    @staticmethod
    def _cats(raw):
        p, d = raw["pickup"].astype(str), raw["delivery"].astype(str)
        pair = [a + "|" + b if a < b else b + "|" + a for a, b in zip(p, d)]
        return pd.DataFrame({"p": p.values, "d": d.values, "lane": (p + ">" + d).values, "pair": pair})

    def fit(self, X, y, raw):
        from sklearn.preprocessing import OneHotEncoder
        from sklearn.linear_model import Ridge
        self.lin.fit(X, y, raw)
        r1 = y - np.log(self.lin.predict(X, raw))
        self.enc = OneHotEncoder(handle_unknown="ignore").fit(self._cats(raw))
        self.re = Ridge(alpha=self.re_alpha).fit(self.enc.transform(self._cats(raw)), r1)
        r2 = r1 - self.re.predict(self.enc.transform(self._cats(raw)))
        if self.use_gbm:
            self.m = lgb.LGBMRegressor(objective="huber", alpha=self.gbm_alpha, **self.params).fit(X[self.f], r2)
        return self

    def predict(self, X, raw):
        out = np.log(self.lin.predict(X, raw)) + self.re.predict(self.enc.transform(self._cats(raw)))
        if self.use_gbm:
            out = out + self.m.predict(X[self.f])
        return np.exp(out)


