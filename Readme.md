# Freight Rate Prediction

Predict `posted_rate` for 12,000 loads (Nov-Dec 2025) from labeled loads of Jan-Oct 2025.

## Run
```bash
python -m pip install -r requirements.txt
python src/eda.py             # EDA -> outputs/eda/ (figures + tables)
python src/compare.py         # 19-config model/hyper-parameter comparison, loss curves, hit rates, residual plots -> outputs/comparison/ (~6 min)
python src/validate.py        # walk-forward validation -> outputs/cv_*.csv  (~1-2 min)
python src/train_predict.py   # trains final model, writes validation_predictions.csv + fills data/december_chart_inputs.csv
python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv
```
`score.py` validates both files and creates `scorer_results/candidate_december.png`.

## Layout
| Path | What |
|---|---|
| `src/eda.py` | EDA: missing values, quality checks, label spikes, drift, figures |
| `src/features.py` | cleaning + feature engineering (daily market signals, geo features) |
| `src/models.py` | baseline, robust linear, LightGBM, hybrid models |
| `src/compare.py` | model + hyper-parameter grid, loss curves, hit rate within tolerance, residual plots |
| `src/validate.py` | walk-forward (time-ordered) validation and model comparison |
| `src/train_predict.py` | final fit + submission files |
| `data/` | challenge inputs (`train_test.csv`, `validation.csv`, templates) |
| `outputs/` | CV result tables |

## Approach (short)
* **Split:** the hidden set is *after* the labeled data, so validation is walk-forward: 5 folds, each trains on the
  past and tests on the next 2 months (May-Jun ... Sep-Oct). No random K-fold (it leaks the market regime).
* **Data quality:** negative weights = sign flips (`abs`), weight capped at 47,500, ~0.7% missing weight/market_index
  (flagged + imputed from the daily level), ~1.4% of labels are random x3.4 / x0.28 spikes (dropped from training only,
  evaluated separately). `distance` is clean; coordinates are synthetic but consistent per city.
* **Key signals:** log-distance (rate ~ distance^0.87), equipment, weight, *daily* market_index and quote_signal
  (row values = daily level + noise), a **linear time trend** (+0.7%/month) and persistent lane/city effects.
* **Model:** robust linear (Huber, log-rate, with time trend so it can extrapolate to Nov-Dec) -> ridge random effects
  for pickup/delivery/lane (unseen cities get 0) -> LightGBM (Huber loss, 200 trees) on the remainder.
  Chosen from a 19-configuration comparison (`outputs/comparison/`).
* **December chart:** lat/lon from the city lookup; market_index / quote_signal = the real daily levels from
  `validation.csv` for each date (features, not labels).
