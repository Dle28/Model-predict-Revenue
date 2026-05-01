# Diagnosis: 2019 Weakening and COVID/Recovery Regime

Generated from local data files:
`Data/sales.csv`, `Data/web_traffic.csv`, `Data/orders.csv`,
`Data/order_items.csv`, `Data/inventory.csv`, `Data/products.csv`,
`Data/returns.csv`, `Data/reviews.csv`.

Derived tables:
`artifacts/insights/yearly_business_diagnostics.csv`,
`artifacts/insights/period_business_diagnostics.csv`,
`artifacts/insights/category_top3_by_year.csv`.

## Executive Conclusion

The data supports a clear business weakening before COVID. The break is visible in 2019, mainly through conversion/order efficiency collapse, not through traffic loss.

Recommended regime interpretation for modeling:

- 2012-2018: healthy pre-COVID operating baseline.
- 2019: pre-COVID but internally weakened/stressed regime.
- 2020-2023: COVID/disrupted demand and operations regime.
- 2024 onward: unknown recovery path. Do not encode a known normalization date.

This means 2019 should not be treated as a clean "normal" anchor even though it is pre-COVID. It should be either downweighted in normal-history anchors or represented with a weakness/stress feature.

## Key Evidence

| Metric | 2018 | 2019 | Change |
|---|---:|---:|---:|
| Revenue | 1.850B | 1.137B | -38.6% |
| Sessions | 9.415M | 9.990M | +6.1% |
| Orders | 69.5K | 41.6K | -40.2% |
| Orders / 1000 sessions | 7.38 | 4.16 | -43.6% |
| Revenue / session | 196.5 | 113.8 | -42.1% |
| AOV | 26.6K | 27.3K | +2.7% |
| Stockout flag rate | 66.5% | 66.9% | +0.7% relative |
| Avg fill rate | 96.27% | 96.43% | +0.2% relative |
| Streetwear revenue share | 81.9% | 81.0% | -1.1% relative |

## What The Data Confirms

1. Traffic did not collapse. It increased in 2019 and kept rising through 2022.

2. Conversion/order efficiency collapsed. Orders per 1000 sessions fell from 7.38 in 2018 to 4.16 in 2019, then stayed around 3.1-3.3 during 2020-2022.

3. Revenue follows orders much more than traffic. Yearly correlation with revenue over 2013-2022:

- Orders: +0.967
- Revenue per session: +0.942
- Orders per 1000 sessions: +0.884
- Sessions: -0.764

4. AOV increased while volume fell. This looks like price/mix pressure or fewer but larger baskets. It did not offset the lost order volume.

5. Streetwear concentration is real. Streetwear stayed above 80% revenue share from 2018 onward. This is a concentration risk, but the 2019 share did not collapse, so the data supports "amplifier/risk" more than "standalone root cause".

## What The Data Does Not Fully Support

The claim "Stockout Rate = 97.72%" is not reproduced from `inventory.stockout_flag`.

Using available inventory definitions:

- Product-month stockout flag rate: about 66-68% across most years.
- Mean stockout-days / days-in-month: about 3.4-4.5%.
- Average fill rate: about 95.4-96.6%.
- Zero stock-on-hand rate: 0%.

Stockout is high as a product-month occurrence metric, but it is stable from 2012-2022 and even slightly improves by fill rate. So this dataset does not prove stockout caused the 2019 break. It may still explain lost conversion if `stockout_flag` means frequent SKU/size unavailability, but it should be modeled as a persistent operational friction, not as the main 2019 shock.

## Modeling Implications

1. Remove future normalization leakage. Recovery end date is unknown at forecast time, so no hard-coded "normal from 2024-W10" type flag.

2. Add/keep a 2019 internal weakness signal. 2019 is pre-COVID calendar time but not a healthy operating baseline.

3. Learn recovery from disrupted years, but do not assume full recovery. Treat 2020-2023 as disrupted/recovery regime and let trend/anchors learn from available data.

4. Conversion proxy should dominate traffic. Use orders/session, revenue/session, expected conversion, and lagged funnel efficiency where forecast-safe. Traffic alone is weak because more sessions coincided with lower revenue.

5. AOV pressure should be a separate feature. Rising AOV during falling conversion is a stress marker, not necessarily healthy growth.

6. Stockout should be included carefully. Use it as an operational friction/risk feature, but do not force it to explain the 2019 collapse unless a stronger stockout definition is available.

7. Streetwear dependency should remain a risk feature. It helps explain fragility because the business has limited category diversification.

