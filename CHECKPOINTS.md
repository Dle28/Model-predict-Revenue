# Forecast Pipeline Checkpoints

The pipeline is now split into the `forecast/` package. `forecast_pipeline.py` remains a thin backward-compatible entrypoint.

## File Layout

- `forecast/common.py`: configuration, constants, CSV/input validation, calendar helpers, retail/Tet holiday flags.
- `forecast/marts.py`: daily and weekly mart builders, including de-holidayed LV1 weekly targets plus traffic, order, basket, supply, return, review, customer, product-mix, promo, and target-history features.
- `forecast/lv1.py`: weekly baseline model. Target is trailing-smoothed de-holidayed weekly revenue/COGS with stronger target history, calendar regimes, and conservative business exogenous features.
- `forecast/lv2.py`: softmax daily allocation. Allocates weekly baseline into `*_lv2_base` daily values using adjusted historical scores plus a validation-tuned dynamic score blend.
- `forecast/lv3.py`: daily spike/residual multiplier model. Learns `actual_daily / lv2_base_daily` from promo, traffic, calendar, activity, base-rank, lag-ratio, and event-intensity features, with clipped spike-up/spike-down hybrid branches.
- `forecast/direct.py`: direct daily Ridge challenger with recursive target lags and moving averages, blended into the hierarchical forecast at 80%.
- `forecast/backtesting.py`: quarterly walk-forward COVID/recovery holdout plus weekday/month/revenue-bucket diagnostics.
- `forecast/final.py`: Base + Spikes orchestration, post-blend weekly reconciliation, final submission, bottom-up checks, interval construction.
- `forecast/plotting.py`: standalone HTML submission plot.
- `forecast/runner.py`: CLI orchestration and artifact saving.

## Current Architecture

The active model is now Base + Spikes:

- LV1 forecasts `revenue_w_base` and `cogs_w_base`, the smooth weekly baseline. It trains on `revenue_w_lv1_target` and `cogs_w_lv1_target`, which remove holiday days and scale the remaining days back to a 7-day weekly baseline. Target lags, same-ISO-week references, rolling means/volatility, demand, conversion, basket, supply, leakage, customer, promo, and calendar-regime signals enter as features.
- LV2 converts each weekly baseline into daily base values using a softmax over adjusted historical scores and a tuned model score by weekday, month-weekday, regime, day-of-month, payday, month boundary, holiday, active promo, near-promo, and weekly context.
- LV3 predicts clipped daily multipliers and produces final daily forecasts. Holiday effects enter here through `is_holiday` plus tactical event floors for Tet, double-day sale, 11/11, 12/12, Black Friday/Cyber Monday, new-year, and year-end windows. A light classifier/regressor hybrid separately handles spike-up and spike-down candidates.
- A direct daily challenger is blended after LV3, before weekly bottom-up totals are recomputed. It now uses recursive target lag features so future days can use earlier forecasts as history.
- Final daily values are reconciled after direct blend with a light weekly anchor. Weekly final totals are then computed from reconciled daily sums, so exported daily and weekly views have zero bottom-up drift.
- Final forecast training uses `2012-W28` through `2022-W52`.

## Current Artifacts

- `artifacts/daily_mart.csv`
- `artifacts/weekly_mart.csv`
- `artifacts/backtest_metrics.csv`
- `artifacts/weekly_forecast.csv`
- `outputs/submission.csv`
- `outputs/submission_intervals.csv`
- `outputs/submission_plot.html`

## Current Backtest

COVID/recovery holdout: start with train `2017-W01` to `2019-W52`, validate `2020-W01` to `2022-W51` using quarterly walk-forward refits (`13` weeks per chunk, `12` chunks total).

- Weekly baseline Revenue WAPE: `24.49%`, R2: `0.443`
- Weekly baseline COGS WAPE: `22.86%`, R2: `0.466`
- Tactical weekly final Revenue WAPE: `10.35%`, R2: `0.901`
- Tactical weekly final COGS WAPE: `10.69%`, R2: `0.903`
- Tactical daily Revenue WAPE: `14.05%`, R2: `0.867`
- Tactical daily COGS WAPE: `14.30%`, R2: `0.876`
- Revenue P10-P90 coverage: `75.64%`; COGS P10-P90 coverage: `83.52%`
- Max daily LV3 multiplier observed in tactical backtest: Revenue `2.23`, COGS `2.33`

## Run

```powershell
python forecast_pipeline.py
```

Use `--no-artifacts` to run validation without saving intermediate mart artifacts.

The default weekly backend is Ridge because it is more stable on the COVID/recovery holdout. To experiment with installed boosted libraries:

```powershell
$env:FORECAST_MODEL_BACKEND = "lightgbm"  # or "xgboost", "auto"
python forecast_pipeline.py
```
