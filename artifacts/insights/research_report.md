# Ecommerce Insight Research

Phạm vi: dữ liệu train 2012-2022, không dùng biến vĩ mô. Các kết luận dưới đây là phân tích nội bộ từ `orders`, `order_items`, `payments`, `inventory`, `promotions`, `web_traffic`, `sales` và các bảng outlier đã sinh.

## 1. Revenue bị chi phối bởi conversion, không phải traffic

Tương quan với Revenue:

- `orders`: 0.936
- `rev_per_session`: 0.753
- `orders_per_session`: 0.627
- `sessions`: 0.283

Khi bóc outlier sau mùa vụ `month x weekday`, nhóm ngày high residual có:

- `orders_per_1000_sessions`: 19.83
- normal: 6.32
- `sessions`: 22.8k, thấp hơn normal 23.9k

Kết luận: ngày revenue tăng đột biến là ngày user mua dồn, không phải ngày traffic cao. Feature quan trọng nhất là conversion density/proxy, không phải raw traffic.

## 2. Payday là event nội bộ mạnh hơn promo

Nhóm high residual:

- 96.7% ngày nằm trong `is_payday_window`
- top raw 1% revenue: 94.9% nằm trong `is_payday_window`
- normal: chỉ 21.3%

Các ngày top revenue thường rơi vào 28-31 hoặc 1-2 đầu tháng. Đây giống hành vi mua sau lương tại Việt Nam hơn là event retail cố định.

Model implication:

- Giữ `is_payday_window`.
- Thêm interaction `payday x month_end`, `payday x month_start`.
- Cho LV3 học riêng payday spike, không gộp chung với promo.

## 3. Promo không đồng nghĩa với tăng revenue

Promo xuất hiện ở cả nhóm tốt và nhóm xấu:

- high residual: `promo_flag` 46.2%
- normal: 44.6%
- bottom raw 1%: 53.8%

Promo line economics:

- Non-promo margin rate: 19.96%
- Promo margin rate: -14.46%
- Promo discount rate: 13.79%

Kết luận: promo trong dataset có thể kéo đơn rẻ hoặc giảm margin, không nên dùng monotonic positive quá mạnh. Promo cần đi cùng context: payday, top SKU mix, conversion density.

Model implication:

- Dùng `promo_margin_pressure` như rủi ro giảm chất lượng doanh thu.
- Không ép mọi promo tăng Revenue.
- Dùng promo projection cho lịch tương lai, nhưng giảm trọng số nếu không đi cùng payday/top SKU.

## 4. Heavy-tail có thật nhưng không cực đoan toàn cục

Daily distribution:

- Top 1% days giữ 3.73% revenue
- Top 5% days giữ 13.99% revenue
- P50 daily revenue: 3.65M
- P99 daily revenue: 13.80M
- Max: 20.91M

Kết luận: có spike đáng kể, nhưng không phải toàn bộ revenue phụ thuộc vài ngày. Cần model spike riêng, nhưng không nên để vài ngày peak chi phối LV1.

Model implication:

- Winsorize hoặc downweight ngày payday spike cực đoan khi train base model.
- Để LV3 xử lý spike residual.

## 5. Product Pareto là driver mạnh

Product mix:

- 1,598 products
- Top 20% products đóng góp 81.86% revenue
- 294 products, tương đương 18.4%, tạo 80% revenue
- 735 products gần như không đóng góp đáng kể

Outlier top raw 1%:

- `top_product_revenue_share`: 87.3%
- normal: 80.6%

Kết luận: spike không chỉ do nhiều đơn, mà do đơn tập trung vào nhóm SKU chủ lực.

Model implication:

- Dùng `expected_top_product_revenue_share`.
- Nếu có dự báo inventory theo SKU, ưu tiên top SKU availability thay vì stock tổng.

## 6. COD là rủi ro vận hành đặc thù Việt Nam

Payment summary:

- COD orders: 96,681
- COD cancel rate: 16.0%
- COD returned rate: 8.9%
- Các payment khác cancel khoảng 8.0%, returned khoảng 5.0%

Kết luận: COD tạo rủi ro hủy/hoàn cao hơn rõ rệt. Đây là yếu tố Việt Nam nên đưa vào forecast COGS/margin và điều chỉnh revenue sau đơn.

Model implication:

- Dùng `expected_cod_order_share`.
- Tách ảnh hưởng COD cho Revenue và COGS vì return/cancel tác động khác nhau.

## 7. Stockout là proxy supply pressure, không phải causal đơn giản

Trong outlier:

- high residual `stockout_rate`: 13.4%
- normal: 1.3%
- top raw 1% `stockout_rate`: 15.6%

Điều này không có nghĩa stockout làm tăng revenue. Khả năng cao là hot SKU bán mạnh rồi hết hàng ở snapshot. Dữ liệu inventory là proxy áp lực cung, không phải causal daily state hoàn chỉnh.

Model implication:

- Dùng `expected_lost_sales_index` và `expected_stockout_rate` nhẹ tay.
- Không áp monotonic negative cứng nếu chưa có lost-sales estimate chuẩn.

## 8. Logistics delay không giải thích revenue trong dataset này

Delivery bucket:

- Return rate 0-2 ngày: 6.20%
- 3-4 ngày: 6.44%
- 5-7 ngày: 6.41%
- 8+ ngày: 6.27%

Kết luận: trong dataset này, delivery delay không tạo gradient return rõ. Không nên đưa trực tiếp vào forecast daily revenue nếu không có feature tương lai chắc chắn.

## 9. Rating không đủ dùng để kết luận conversion

Review là post-purchase/selected sample. Không thấy đủ bằng chứng rating gây conversion trong dữ liệu hiện tại.

Model implication:

- Không đưa rating vào forecast chính.
- Chỉ dùng cho diagnostic chất lượng sản phẩm nếu cần.

## 10. Cấu trúc outlier theo thời kỳ

High residual top 80 tập trung trước 2019:

- 2014-2018 chiếm phần lớn
- 2016 có nhiều nhất

Low residual top 80 tập trung 2019-2022:

- 2020: 21 ngày
- 2021: 23 ngày
- 2022: 11 ngày

Kết luận: ngoài feature vận hành, còn có regime shift lớn sau 2019/Covid/recovery. Base model phải xử lý regime trước khi học spike.

## Khuyến nghị ưu tiên

1. Base weekly model: giữ regime/recovery, thêm expected conversion và expected top SKU mix.
2. LV2 allocation: tăng vai trò payday, month-start/month-end, expected conversion.
3. LV3 spike: học residual cho payday spike riêng, không coi promo là spike positive mặc định.
4. Guardrail: downweight hoặc winsorize top payday spikes khi train base, để spike model xử lý.
5. Không đưa rating/logistics vào model chính hiện tại vì bằng chứng yếu hoặc không forecast-safe.

