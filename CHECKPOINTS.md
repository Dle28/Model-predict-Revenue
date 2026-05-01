# Forecast Pipeline Checkpoints

The pipeline is now split into the `forecast/` package. `forecast_pipeline.py` remains a thin backward-compatible entrypoint.

## File Layout

- `forecast/common.py`: configuration, constants, CSV/input validation, calendar helpers, retail/Tet holiday flags.
- `forecast/marts.py`: daily and weekly mart builders, intentionally limited to sales, sessions, calendar, planned promotions, and target-history features.
- `forecast/lv1.py`: weekly baseline model. Target is complete weekly revenue/COGS with stronger target history, calendar regimes, and conservative forecast-safe promo exogenous features.
- `forecast/structural.py`: optional structural weekly trend/seasonality challenger. It is direct, non-recursive, weekly-grain, and off by default after the mart cleanup benchmark.
- `forecast/lv2.py`: softmax daily allocation. Allocates weekly baseline into `*_lv2_base` daily values using adjusted historical scores plus a validation-tuned dynamic score blend.
- `forecast/lv3.py`: daily spike/residual multiplier model. Learns `actual_daily / lv2_base_daily` from forecast-safe promo, calendar, base-rank, lag-ratio, and event-intensity features, with clipped spike-up/spike-down hybrid branches.
- `forecast/direct.py`: direct daily Ridge challenger with recursive target lags and moving averages, blended at 5% after benchmark confirmation.
- `forecast/intervals.py`: split-conformal absolute residual quantile helper used by LV1/LV3/structural intervals.
- `forecast/backtesting.py`: quarterly walk-forward COVID/recovery holdout plus weekday/month/revenue-bucket diagnostics.
- `forecast/final.py`: Base + Spikes orchestration, post-blend weekly reconciliation, final submission, bottom-up checks, interval construction.
- `forecast/plotting.py`: standalone HTML submission plot.
- `forecast/runner.py`: CLI orchestration and artifact saving.

## Current Architecture

The active model is now Base + Spikes:

- LV1 forecasts `revenue_w_base` and `cogs_w_base`, the smooth weekly baseline. It trains on complete weekly totals; holiday and event spikes are left to LV3 instead of being scaled into the weekly target. Target lags, same-ISO-week references, rolling means, sessions, forecast-safe promo plans, and calendar-regime signals enter as features.
- The mart feature surface is deliberately forecast-safe. Operational outcomes such as order mix, customer tenure, returns, reviews, fulfillment, inventory, COD behavior, and promo/non-promo basket metrics are not merged into the modeling mart.
- A structural weekly trend layer remains available through `FORECAST_STRUCTURAL_WEEKLY_WEIGHT`, but the default is `0.00` because it worsened the cleaned 2022 holdout.
- LV1 defaults to XGBoost and falls back to Ridge/LightGBM alternatives if unavailable. A fitted bias factor corrects aggregate train-scale under/over forecast after the log-scale model.
- LV2 converts each weekly baseline into daily base values using a softmax over adjusted historical scores and a tuned model score by weekday, month-weekday, regime, day-of-month, payday, month boundary, holiday, active promo, near-promo, and weekly context.
- LV3 predicts event-capped daily multipliers and produces final daily forecasts. Holiday, promo, supply, stockout, and recovery interactions enter the spike model. A light classifier/regressor hybrid separately handles spike-up and spike-down candidates.
- A direct daily challenger blends after LV3 at 5% before weekly bottom-up totals are recomputed. It uses recursive target lag features, so keep the weight small.
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

- Primary 2022 holdout: train `2012-W28` to `2021-W52`, validate `2022-W01` to `2022-W51` using quarterly walk-forward refits (`13` weeks per chunk, `4` chunks total). The default weekly post-LV3 drift cap is `0.10`.

- Weekly baseline Revenue WAPE: `14.40%`, R2: `0.841`
- Weekly baseline COGS WAPE: `14.82%`, R2: `0.829`
- Tactical weekly final Revenue WAPE: `15.47%`, R2: `0.822`
- Tactical weekly final COGS WAPE: `16.15%`, R2: `0.803`
- Tactical daily Revenue WAPE: `22.63%`, R2: `0.667`
- Tactical daily COGS WAPE: `23.36%`, R2: `0.652`
- Revenue bias all/high bucket: `-1.25%` / `-16.55%`
- Revenue P10-P90 coverage: `92.16%`; COGS P10-P90 coverage: `92.16%`
- Max daily LV3 multiplier observed in tactical backtest: Revenue `1.24`, COGS `1.21`

## Run

```powershell
python forecast_pipeline.py
```

Use `--yearly-cv` for the full rolling-origin validation suite. Use `--no-artifacts` to run validation without saving intermediate mart artifacts.

Manual high-bucket correction and sample-submission anchoring are disabled. Set `FORECAST_CONTROLLED_RECOVERY_REVENUE_TOTAL` to a numeric target to enable final business total calibration for a controlled scenario.

The direct daily challenger defaults to a small `FORECAST_DIRECT_BLEND_WEIGHT=0.05`; larger values should be treated as benchmark experiments because recursive daily error accumulation can dominate.

The structural weekly anchor defaults to `FORECAST_STRUCTURAL_WEEKLY_WEIGHT=0.00`. After the mart cleanup, the primary 2022 score is better with structural off (`0.2084`) than with `0.10` (`0.2115`), so structural remains a challenger only.

LV3 event-denominator experiments are available but off by default because the cached rolling holdout worsened when forcing event days to train on `weekly_base / 7`. To test that hypothesis explicitly, set `FORECAST_LV3_EVENT_BASE_MODE=weekly_avg` and optionally `FORECAST_LV3_EVENT_FLOOR=on`.

Variant benchmark results are saved to `artifacts/separate_benchmark_metrics.csv`. The confirmed defaults are `FORECAST_DIRECT_BLEND_WEIGHT=0.05` and `FORECAST_STRUCTURAL_WEEKLY_WEIGHT=0.00`. On the cleaned 2022 fold, direct `0.05` improves selection score from `0.2104` to `0.2084`. Event-window tuning favored `FORECAST_LV2_HIST_DECAY_HALFLIFE_WEEKS=26`, but full-2022 validation worsened, so the default remains `52`.

Yearly rolling-origin CV is saved to `artifacts/backtest_metrics.csv` for the current default run. Current result: 2020-2022 folds range from `15.5-17.4%` weekly Revenue WAPE, while 2019 remains a weak transition fold (`51.2%` weekly Revenue WAPE), so report conclusions should not rely on 2022 alone.

The default weekly/LV3 tree backend is `xgboost`. To force a backend:

```powershell
$env:FORECAST_MODEL_BACKEND = "xgboost"  # or "lightgbm", "auto", "ridge"
$env:FORECAST_LV3_MODEL_BACKEND = "xgboost"  # optional LV3-only override
python forecast_pipeline.py
```
