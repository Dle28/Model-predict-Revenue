# Forecast Pipeline System

Pipeline chinh nam trong `forecast_pipeline.py`, entrypoint nay goi `forecast.runner.main()`.
Khong con dung script phu `run_best650_pipeline.py`.

Lenh tai tao ban submission chinh:

```powershell
python forecast_pipeline.py --use-cached-marts --skip-backtest --clean-extra-outputs --clean-extra-artifacts
```

Ket qua chinh:

- `outputs/submission.csv`: file nop cuoi cung.
- `outputs/submission_best650_rev1000_cogs1015.csv`: ban canonical giong `submission.csv`.
- `outputs/submission_model_raw.csv`: output raw cua model truoc calibration best650.
- `outputs/submission_intervals.csv`: forecast interval p05/p10/p90/p95.
- `outputs/submission_plot.html`: plot history + forecast.
- `artifacts/shap/`: SHAP summary, waterfall va feature importance cho LV3 Revenue/COGS.
- `artifacts/pipeline_best650_summary.md`: tong ket run.

## Luong Chay Tong The

1. Validate input trong `Data/` va `Data/sample_submission.csv`.
2. Load cached marts neu co `--use-cached-marts`; neu khong thi build lai `daily_mart` va `weekly_mart`.
3. Tao diagnostics theo nam tu `daily_mart`.
4. Tuy chon backtest rolling-origin neu khong truyen `--skip-backtest`.
5. Chay final model Base + Spikes:
   - LV1 du bao baseline tuan.
   - LV2 phan bo baseline tuan xuong ngay.
   - LV3 hoc multiplier/spike ngay.
   - Direct daily challenger co the blend neu bat bien moi truong.
   - Reconcile daily/weekly va tao intervals.
6. Luu raw model output vao `outputs/submission_model_raw.csv`.
7. Mac dinh ap dung best650 calibration:
   - source: `outputs/submission_best_650k.csv`
   - Revenue scale: `1.000`
   - COGS scale: `1.015`
8. Validate final submission: dung cot, dung ngay theo sample, khong NaN/inf, khong am.
9. Align lai intervals, weekly forecast va plot theo final submission.
10. Chay SHAP cho LV3 Revenue va COGS.

## Input Du Lieu

Pipeline validate cac file sau trong `Data/`:

- `sales.csv`
- `sample_submission.csv`
- `orders.csv`
- `order_items.csv`
- `web_traffic.csv`
- `inventory.csv`
- `shipments.csv`
- `returns.csv`
- `reviews.csv`
- `customers.csv`
- `promotions.csv`
- `products.csv`

Feature mart final dang dung truc tiep cac nhom:

- Sales: daily `Revenue`, `COGS`.
- Web traffic: sessions, visitors, page views, bounce rate, session duration.
- Orders: order count, active customers, status mix, payment mix.
- Basket/order items/products: units, gross/net item value, discount, promo line, top product mix, streetwear mix.
- Inventory: stock on hand, received/sold units, stockout, fill rate, days of supply, sell-through.
- Promotions: active promo count, discount value, stackable, start/end day, projected promo calendar.
- Calendar: weekday, week, month, quarter, month boundary, payday, Tet, Hung Kings, lunar holiday, Black Friday, double-day sales, 11.11/12.12, year-end/new-year windows.

`shipments.csv`, `reviews.csv`, `customers.csv` duoc validate de bao dam bo data day du, nhung khong dua truc tiep vao mart model final hien tai. `returns.csv` chi anh huong sales khi bat `FORECAST_NET_REVENUE=1`; mac dinh khong tru refund.

## Daily Mart

`forecast.marts.build_daily_mart()` tao `artifacts/daily_mart.csv` theo ngay tu ngay sales dau tien den het tuan chua ngay cuoi trong sample.

Nhom feature chinh:

- Calendar flags: `iso_weekday`, `month`, `quarter`, `day_of_month`, `day_of_year`, `is_weekend`, `is_month_start`, `is_month_end`, `is_payday_window`.
- Retail events: `is_holiday`, `is_tet_window`, `is_hung_kings_day`, `is_lunar_holiday`, `is_black_friday_window`, `is_double_day_sale`, `is_1111_1212`, `is_year_end_window`, `is_new_year_window`.
- Promo safe features: `active_promo_count`, `avg_promo_discount_value`, `stackable_promo_count`, `promo_start_day`, `promo_end_day`, `days_to_promo`, `days_since_promo_start`, `days_until_promo_end`, `promo_projected_flag`.
- Forecast-safe operational profiles: `expected_orders_per_1000_sessions`, `expected_stockout_rate`, `expected_fill_rate`, `expected_top_product_revenue_share`, `expected_promo_order_share`, `expected_promo_discount_rate`, `expected_lost_sales_index`, `promo_margin_pressure`.
- Lag/rolling target history: `revenue_lag_1d/7d/14d/28d`, `revenue_ma_7d/28d`, `revenue_shock_7d`, va tuong tu cho `cogs`.
- Regime flags: `is_operational_crisis`, `internal_stress_regime`.

Nguyen tac leakage:

- Feature tuong lai chi dung calendar va promo plan/projected.
- Cac profile operational duoc bien doi thanh expected/prior-year/lagged profile.
- Target lag/MA chi lay tu qua khu; trong du bao multi-day, LV3/direct cap nhat autoregressive bang prediction truoc do.

Projected promo pulse:

- `attach_projected_promos()` khong con lam phang bang rule trung binh theo nhieu nam.
- Future promo duoc copy theo block active cua reference year gan nhat cung nhip 2 nam:
  - 2023 map tu 2021.
  - 2024 map tu 2022.
- Cac doan active duoc chuyen sang cung ngay/thang cua nam forecast, giu block bat/tat ro rang.
- Trong ngay co nhieu promo overlap, projected feature chon promo manh nhat theo discount/impact.
- `active_promo_count` cua projected day duoc ep thanh `1.0`, ngay khong projected la `0.0`; `promo_projected_flag` cung la 0/1.
- `avg_promo_discount_value` cua projected day la discount roi rac cua promo manh nhat, khong dung average.

## Weekly Mart

`forecast.marts.build_weekly_mart()` tao `artifacts/weekly_mart.csv` bang cach aggregate daily mart theo `week_start`.

Feature tuan chinh:

- Target weekly: `revenue_w`, `cogs_w`, chi giu cac tuan co target day du.
- Calendar cyclic: `week_sin`, `week_cos`, `month_sin`, `month_cos`.
- Promo weekly: `promo_active_days`, `promo_has_active`, `promo_discount_sum`, `lv1_active_promo_count`, `lv1_avg_promo_discount_value`.
- Event weekly: `has_holiday`, `is_tet_like_period`, `is_black_friday_like_period`, payday/month-edge flags.
- Regime/recovery: pre-COVID, COVID drop, recovery, normalization, recovery progress.
- Shock weekly: `revenue_w_shock_4w`, `cogs_w_shock_4w`.

## Model Architecture

He thong la Base + Spikes, gom 3 tang chinh va mot challenger phu.

### LV1 Weekly Baseline

Module: `forecast.lv1`

Muc tieu:

- Du bao weekly baseline cho `revenue_w` va `cogs_w`.
- Training window final: `2012-W28` den `2022-W51`.

Model:

- Mac dinh XGBoost (`FORECAST_MODEL_BACKEND=xgboost`).
- Fallback: LightGBM hoac Ridge log model neu backend khong kha dung.
- Target train tren log scale `log1p(target)`.
- Sample weight giam trong COVID drop, tang trong recovery.

Feature LV1:

- Seasonality: week/month sin-cos, month start/end, payday, Tet/Black Friday.
- Target history: lag 1w/4w/52w, MA 4w, same ISO week last year, YoY history, target growth.
- Recovery anchors: pre-COVID same-week baseline, COVID-adjusted 52w lag, recovery progress.
- Forecast-safe exogenous: expected conversion, lagged revenue/session, funnel efficiency, expected COD share, expected stockout/fill rate, top product/streetwear risk, expected promo share/discount, lost-sales index, promo pressure, LV1 promo count/discount.

Blend LV1:

- Ket hop model prediction, same-ISO-week reference, va recovery anchor.
- Recovery/normalization phase dung weight dong theo `recovery_progress`.
- Structural weekly model co san nhung mac dinh weight `0.00`.

### LV2 Daily Allocation

Module: `forecast.lv2`

Muc tieu:

- Phan bo weekly baseline tu LV1 thanh daily baseline (`Revenue_lv2_base`, `COGS_lv2_base`).
- Tong daily theo week bang weekly baseline truoc LV3.

Logic:

- Hoc historical share cua tung ngay trong tuan.
- Co bang weight theo weekday, month-weekday, day-of-month, va weekday theo regime.
- Dung recency decay half-life mac dinh 52 tuan.
- Ridge log model hoc phan dieu chinh so voi historical weight.
- Softmax normalize de daily weights trong moi week tong bang 1.

Feature LV2:

- Calendar: weekday, month, day-of-month, day-of-year, week position in month.
- Event: holiday, Tet, Hung Kings, lunar holiday, Black Friday, double-day, 11.11/12.12, year-end/new-year.
- Promo: active promo count, discount, stackable, start/end, days to/since/until promo.
- Weekly context: log weekly base, weekly trend 1w, weekly vs MA4, weekly volatility.
- Forecast-safe operational profiles: expected conversion, expected COD, stockout, fill rate, product mix, promo pressure.
- Regime: pre-COVID, COVID drop, recovery, normalization.

### LV3 Daily Spike Multiplier

Module: `forecast.lv3`

Muc tieu:

- Hoc multiplier ngay: `actual_daily / lv2_base_daily`.
- Sua spike/drop do campaign, holiday, payday, stockout, recovery va calendar effect.

Model:

- Mac dinh XGBoost monotone (`FORECAST_LV3_MODEL_BACKEND=xgboost`).
- Fallback: LightGBM monotone hoac Ridge.
- Target la `log(multiplier)`, multiplier clip tu `0.05` den `4.00`.
- Monotone constraints ap dung cho promo count/discount/stackable de spike khong di nguoc tin hieu promo.

Feature LV3:

- Base context: `log_lv2_base`, `base_share_in_week`, `base_vs_week_mean`, `base_rank_in_week`, `base_vs_precovid_baseline`, `recovery_gap`.
- Calendar/event: weekday, month, day/month/year sin-cos, payday, holiday, Tet, Hung Kings, lunar holiday, Black Friday, double-day, 11.11/12.12, year-end/new-year.
- Promo/activity: active promo, promo flag, discount, stackable, promo segment/type/impact, projected promo.
- Lag/rolling history: revenue/cogs lag 7d/14d/28d, MA 7d/28d, shock 7d, `lag7_vs_base`.
- Operational risk: orders/session lag, revenue/session lag, funnel efficiency, streetwear concentration, promo margin pressure.
- Regime: operational crisis/internal stress.
- Event intensity: tong hop holiday + campaign + payday + promo count.

Intervals:

- LV3 tao p10/p90 va p05/p95 bang split-conformal residual tren log multiplier.
- Interval scale tang nhe theo horizon ngay.

### Direct Daily Challenger

Module: `forecast.direct`

Muc tieu:

- La model phu du bao truc tiep daily Revenue/COGS bang Ridge log model.
- Mac dinh blend weight trong code la `0.0`; neu bat `FORECAST_DIRECT_BLEND_WEIGHT`, no blend vao output sau LV3.

Feature direct:

- Calendar/event tuong tu LV3.
- Activity forecast-safe.
- Target lag 1d/7d/14d/28d, MA 7d/28d.
- Blend decay theo horizon: co tac dung manh hon gan train-end, giam dan ve xa.

## Final Forecast And Calibration

Module: `forecast.final` va `forecast.runner`

Sau LV1/LV2/LV3:

- Direct challenger blend neu duoc bat.
- Reconcile daily forecast ve weekly anchor nhe.
- Tao `weekly_forecast.csv` tu bottom-up daily sum.
- Tao intervals.
- Validate final shape theo sample submission.

Best650 calibration la buoc mac dinh trong runner:

- Raw model output duoc giu lai o `outputs/submission_model_raw.csv`.
- Final submission lay tu `outputs/submission_best_650k.csv`.
- Ap dung scale:
  - Revenue: `1.000`
  - COGS: `1.015`
- Ghi ra `outputs/submission.csv` va `outputs/submission_best650_rev1000_cogs1015.csv`.
- Intervals va weekly forecast duoc align lai theo final submission.

Neu muon xem raw model thuan:

```powershell
python forecast_pipeline.py --use-cached-marts --skip-backtest --raw-model-only --skip-shap
```

## SHAP

Module: `forecast.explain`

Pipeline chay SHAP cho LV3 spike model cua hai target:

```powershell
python -m forecast.explain --target revenue --date 2021-11-11 --sample-size 800 --output-dir artifacts/shap
python -m forecast.explain --target cogs --date 2021-11-11 --sample-size 800 --output-dir artifacts/shap
```

Artifacts:

- `revenue_lv3_feature_importance.csv`
- `revenue_lv3_shap_summary.png`
- `revenue_lv3_shap_waterfall_20211111.png`
- `revenue_lv3_shap_waterfall_20211111.csv`
- `cogs_lv3_feature_importance.csv`
- `cogs_lv3_shap_summary.png`
- `cogs_lv3_shap_waterfall_20211111.png`
- `cogs_lv3_shap_waterfall_20211111.csv`

SHAP chi giai thich LV3 spike/multiplier model, khong giai thich toan bo post-calibration best650. Nghia la SHAP tra loi cau hoi "vi sao LV3 nang/ha daily multiplier", khong phai "vi sao file final da scale COGS 1.015".

## Backtest And Diagnostics

Backtest mac dinh co the bat bang:

```powershell
python forecast_pipeline.py --use-cached-marts --with-backtest --skip-shap
```

Backtest dung rolling-origin holdout:

- Train: `2019-W01` den `2021-W52`
- Validate: `2022-W01` den `2022-W51`
- Chunk validation theo 13 tuan.

Metrics:

- Weekly WAPE/R2 cho Revenue va COGS.
- Daily WAPE/R2 cho Revenue va COGS.
- Bias theo all/low/high revenue bucket.
- Coverage p10-p90.
- Drift weekly va bottom-up coherence.
- Event-floor diagnostics neu co.

## Bien Moi Truong Quan Trong

Mac dinh runner reset cac bien `FORECAST_*` de tai tao canonical best650 va set:

- `FORECAST_MODEL_BACKEND=xgboost`
- `FORECAST_LV3_MODEL_BACKEND=xgboost`
- `PYTHONHASHSEED=0`

Muon tu giu bien moi truong hien tai:

```powershell
python forecast_pipeline.py --use-cached-marts --skip-backtest --respect-env
```

Bien hay dung:

- `FORECAST_MODEL_BACKEND`: `xgboost`, `lightgbm`, `ridge`, `auto`.
- `FORECAST_LV3_MODEL_BACKEND`: backend rieng cho LV3.
- `FORECAST_DIRECT_BLEND_WEIGHT`: bat/tat direct daily blend.
- `FORECAST_LV2_HIST_DECAY_HALFLIFE_WEEKS`: half-life cua historical allocation.
- `FORECAST_STRUCTURAL_WEEKLY_WEIGHT`: weight structural weekly challenger, mac dinh 0.
- `FORECAST_PROMO_PROJECTION`: bat/tat projected promo calendar.
- `FORECAST_LV1_USE_PROJECTED_PROMO`: cho LV1 dung projected promo.
- `FORECAST_NET_REVENUE`: tru refund vao revenue khi build sales daily, mac dinh off.

## Lenh Thuong Dung

Tai tao submission chinh nhanh, dung cached marts:

```powershell
python forecast_pipeline.py --use-cached-marts --skip-backtest --clean-extra-outputs --clean-extra-artifacts
```

Build lai marts tu data goc:

```powershell
python forecast_pipeline.py --skip-backtest
```

Chay khong SHAP de nhanh hon:

```powershell
python forecast_pipeline.py --use-cached-marts --skip-backtest --skip-shap
```

Chay raw model only:

```powershell
python forecast_pipeline.py --use-cached-marts --skip-backtest --raw-model-only --skip-shap
```

Chay backtest:

```powershell
python forecast_pipeline.py --use-cached-marts --with-backtest --skip-shap
```
