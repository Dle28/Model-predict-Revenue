# Forecast Pipeline Checkpoints

The pipeline is now split into the `forecast/` package. `forecast_pipeline.py` remains a thin backward-compatible entrypoint.

## File Layout

- `forecast/common.py`: configuration, constants, CSV/input validation, calendar helpers.
- `forecast/marts.py`: daily and weekly mart builders.
- `forecast/models.py`: Ridge/optional LightGBM/XGBoost model layer, weekly feature generation, recursive weekly forecast, COGS ratio forecast, metrics.
- `forecast/allocation.py`: weekly-to-daily allocation, calendar/promo/payday weight adjustments, seasonal daily shim.
- `forecast/backtesting.py`: COVID holdout backtest.
- `forecast/final.py`: final submission, validation, coherence checks, interval construction.
- `forecast/plotting.py`: standalone HTML submission plot.
- `forecast/runner.py`: CLI orchestration and artifact saving.

## Current Artifacts

- `artifacts/daily_mart.csv`
- `artifacts/weekly_mart.csv`
- `artifacts/backtest_metrics.csv`
- `artifacts/weekly_forecast.csv`
- `outputs/submission.csv`
- `outputs/submission_intervals.csv`
- `outputs/submission_plot.html`

## Current Backtest

COVID holdout: train `2012-W28` to `2019-W52`, validate `2020-W01` to `2022-W51`.

- Weekly Revenue WAPE: `17.46%`, R2: `0.691`
- Weekly COGS WAPE: `16.92%`, R2: `0.673`
- Daily Revenue WAPE: `26.24%`, R2: `0.507`
- Daily COGS WAPE: `26.38%`, R2: `0.487`

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
