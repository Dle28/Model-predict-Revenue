# Forecast Pipeline Checkpoints

The pipeline is now split into the `forecast/` package. `forecast_pipeline.py` remains a thin backward-compatible entrypoint.

## File Layout

- `forecast/common.py`: configuration, constants, CSV/input validation, calendar helpers, retail/Tet holiday flags.
- `forecast/marts.py`: daily and weekly mart builders, including de-holidayed LV1 weekly targets plus traffic, order, basket, supply, return, review, customer, product-mix, promo, and target-history features.
- `forecast/lv1.py`: weekly baseline model. Target is trailing-smoothed de-holidayed weekly revenue/COGS with stronger target history, calendar regimes, and conservative business exogenous features.
- `forecast/lv2.py`: softmax daily allocation. Allocates weekly baseline into `*_lv2_base` daily values using adjusted historical scores plus a validation-tuned dynamic score blend.
- `forecast/lv3.py`: daily spike/residual multiplier model. Learns `actual_daily / lv2_base_daily` from promo, traffic, calendar, activity, base-rank, lag-ratio, and event-intensity features, with clipped spike-up/spike-down hybrid branches.
- `forecast/direct.py`: direct daily Ridge challenger with recursive target lags and moving averages, blended into the hierarchical forecast at 5%.
- `forecast/backtesting.py`: quarterly walk-forward COVID/recovery holdout plus weekday/month/revenue-bucket diagnostics.
- `forecast/final.py`: Base + Spikes orchestration, post-blend weekly reconciliation, final submission, bottom-up checks, interval construction.
- `forecast/plotting.py`: standalone HTML submission plot.
- `forecast/runner.py`: CLI orchestration and artifact saving.

## Current Architecture

The active model is now Base + Spikes:

- LV1 forecasts `revenue_w_base` and `cogs_w_base`, the smooth weekly baseline. It trains on complete weekly totals; holiday and event spikes are left to LV3 instead of being scaled into the weekly target. Target lags, same-ISO-week references, rolling means/volatility, demand, conversion, basket, supply, leakage, customer, promo, and calendar-regime signals enter as features.
- LV1 defaults to `auto`, which tries LightGBM first and falls back to Ridge/XGBoost alternatives if unavailable. A fitted bias factor corrects aggregate train-scale under/over forecast after the log-scale model.
- LV2 converts each weekly baseline into daily base values using a softmax over adjusted historical scores and a tuned model score by weekday, month-weekday, regime, day-of-month, payday, month boundary, holiday, active promo, near-promo, and weekly context.
- LV3 predicts event-capped daily multipliers and produces final daily forecasts. Holiday, promo, supply, stockout, and recovery interactions enter the spike model. A light classifier/regressor hybrid separately handles spike-up and spike-down candidates.
- A direct daily challenger is blended after LV3, before weekly bottom-up totals are recomputed. It now uses recursive target lag features so future days can use earlier forecasts as history.
- Final daily values are reconciled after direct blend with a light weekly anchor and capped by a weekly drift guardrail. Manual high-bucket multipliers are disabled to avoid double-counting event effects already learned by LV3. Weekly final totals are then computed from reconciled daily sums, so exported daily and weekly views have zero bottom-up drift.
- Final forecast training uses `2012-W28` through `2022-W51`.
- Controlled recovery total calibration is available but off by default; sample-submission anchoring is disabled.

## Current Artifacts

- `artifacts/daily_mart.csv`
- `artifacts/weekly_mart.csv`
- `artifacts/backtest_metrics.csv`
- `artifacts/weekly_forecast.csv`
- `outputs/submission.csv`
- `outputs/submission_intervals.csv`
- `outputs/submission_plot.html`

## Current Backtest

COVID/recovery holdout: train `2015-W01` to `2021-W52`, validate `2022-W01` to `2022-W51` using quarterly walk-forward refits (`13` weeks per chunk, `4` chunks total). The default weekly post-LV3 drift cap is `0.10`.

- Weekly baseline Revenue WAPE: `16.03%`, R2: `0.808`
- Weekly baseline COGS WAPE: `15.80%`, R2: `0.821`
- Tactical weekly final Revenue WAPE: `16.22%`, R2: `0.803`
- Tactical weekly final COGS WAPE: `16.60%`, R2: `0.801`
- Tactical daily Revenue WAPE: `22.63%`, R2: `0.656`
- Tactical daily COGS WAPE: `23.15%`, R2: `0.655`
- Revenue bias all/high bucket: `-0.15%` / `-14.06%`
- Revenue P10-P90 coverage: `89.64%`; COGS P10-P90 coverage: `90.20%`
- Max daily LV3 multiplier observed in tactical backtest: Revenue `1.24`, COGS `1.20`

## Run

```powershell
python forecast_pipeline.py
```

Use `--yearly-cv` for the full rolling-origin validation suite. Use `--no-artifacts` to run validation without saving intermediate mart artifacts.

Manual high-bucket correction and sample-submission anchoring are disabled. Set `FORECAST_CONTROLLED_RECOVERY_REVENUE_TOTAL` to a numeric target to enable final business total calibration for a controlled scenario.

LV3 event-denominator experiments are available but off by default because the cached rolling holdout worsened when forcing event days to train on `weekly_base / 7`. To test that hypothesis explicitly, set `FORECAST_LV3_EVENT_BASE_MODE=weekly_avg` and optionally `FORECAST_LV3_EVENT_FLOOR=on`.

Yearly rolling-origin CV is saved to `artifacts/yearly_rolling_origin_cv.csv`. Current result: 2020-2022 folds are stable around `8.5-15.4%` weekly Revenue WAPE, while 2019 remains a weak transition fold (`46.1%` weekly Revenue WAPE), so report conclusions should not rely on 2022 alone.

The default weekly/LV3 tree backend is `xgboost`. To force a backend:

```powershell
$env:FORECAST_MODEL_BACKEND = "xgboost"  # or "lightgbm", "auto", "ridge"
$env:FORECAST_LV3_MODEL_BACKEND = "xgboost"  # optional LV3-only override
python forecast_pipeline.py
```
