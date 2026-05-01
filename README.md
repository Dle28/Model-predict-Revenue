# DATATHON Forecast Pipeline

Repo nay chua pipeline du bao `Revenue` va `COGS` de tao file submission cho vong 1.

## Cau Truc Thu Muc

```text
DATATHON/
|-- Data/                  # Du lieu dau vao cua de thi
|-- forecast/              # Source code pipeline forecast
|   |-- common.py          # Config, validate input, calendar utilities
|   |-- marts.py           # Build daily/weekly mart
|   |-- lv1.py             # Forecast weekly baseline
|   |-- lv2.py             # Allocate weekly forecast xuong daily
|   |-- lv3.py             # Daily spike/multiplier model
|   |-- final.py           # Ghep final forecast va validate submission
|   `-- runner.py          # Orchestrate toan bo pipeline
|-- artifacts/             # Intermediate artifacts: marts, diagnostics, SHAP
|-- outputs/               # Submission va plot output
|-- Styles/                # Tai lieu/report/dashboard lien quan
|-- forecast_pipeline.py   # Entrypoint chinh de chay pipeline
|-- PIPELINE.md            # Mo ta pipeline chi tiet hon
|-- baseline.ipynb         # Notebook baseline/tham khao
`-- datathon_setup.sql     # SQL setup neu can nap data vao DB
```

## Du Lieu Dau Vao

Dat cac file csv cua de thi trong folder `Data/`. Pipeline se validate cac file chinh nhu:

- `sales.csv`
- `sample_submission.csv`
- `orders.csv`, `order_items.csv`
- `web_traffic.csv`
- `inventory.csv`
- `shipments.csv`, `returns.csv`, `reviews.csv`
- `customers.csv`, `promotions.csv`, `products.csv`

File submission dau ra se match ngay va format theo `Data/sample_submission.csv`.

## Cai Dat Moi Truong

Chay tu root repo:

```powershell
python -m pip install pandas numpy xgboost shap matplotlib
```

`lightgbm` la optional fallback, khong bat buoc neu dung mac dinh XGBoost.

## Chay Lay Submission

Lenh nhanh, dung cached mart co san va bo qua backtest/SHAP:

```powershell
python forecast_pipeline.py --use-cached-marts --skip-backtest --skip-shap
```

Ket qua can nop nam tai:

```text
outputs/submission.csv
```

Mot so output phu:

- `outputs/submission_model_raw.csv`: raw model output.
- `outputs/submission_intervals.csv`: forecast intervals.
- `outputs/submission_plot.html`: plot history va forecast.
- `artifacts/weekly_forecast.csv`: forecast o cap weekly.
- `artifacts/pipeline_model_summary.md`: tom tat run.

## Khi Can Build Lai Tu Dau

Neu chua co cached mart trong `artifacts/`, bo flag `--use-cached-marts`:

```powershell
python forecast_pipeline.py --skip-backtest --skip-shap
```

Lenh canonical co cleanup output phu va tao SHAP artifacts:

```powershell
python forecast_pipeline.py --use-cached-marts --skip-backtest --clean-extra-outputs --clean-extra-artifacts
```

Neu muon chay them backtest:

```powershell
python forecast_pipeline.py --with-backtest --yearly-cv --skip-shap
```

## Ghi Chu

- Nen chay command tu root repo `DATATHON/`.
- Neu chi can file nop bai, dung `outputs/submission.csv`.
- Neu command bao thieu cached mart, chay lai khong co `--use-cached-marts`.
- Chi tiet model va cac checkpoint nam trong `PIPELINE.md`.
