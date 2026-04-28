
/* =================================================================================================
   DATATHON 2026 - FULL MYSQL SETUP SCRIPT
   Mục tiêu:
   1) Tạo lại schema datathon.
   2) Tạo toàn bộ bảng theo các file CSV.
   3) LOAD DATA LOCAL INFILE đúng thứ tự, xử lý NULL/blank ở các cột nullable.
   4) Gắn primary key, unique key, foreign key, index.
   5) Tạo view/function/procedure phục vụ EDA, kiểm tra dữ liệu và forecasting.

   Yêu cầu môi trường:
   - MySQL Server 8.0+.
   - MySQL Workbench bật Allow LOAD DATA LOCAL INFILE.
   - Server/client cho phép local_infile.
   - Đổi biến @DATA_DIR bên dưới đúng thư mục chứa file CSV của bạn.
   - Nên dùng dấu "/" trong path Windows để tránh lỗi mất ký tự "\": 
     đúng:  C:/Users/DungLe/Documents/project/Data/
     sai dễ lỗi: C:\Users\DungLe\Documents\project\Data\
================================================================================================= */

-- Nếu SHOW bên dưới trả về OFF, chạy riêng bằng user root/admin:
-- SET GLOBAL local_infile = 1;
-- Ngoài ra MySQL Workbench phải bật: Edit > Preferences > SQL Editor > Enable LOAD DATA LOCAL INFILE.
SHOW VARIABLES LIKE 'local_infile';

-- Đường dẫn thư mục chứa 13 file CSV.
-- Lưu ý: phải có dấu "/" cuối cùng.
SET @DATA_DIR = 'C:/Users/DungLe/Documents/project/Data/';

DROP DATABASE IF EXISTS datathon;
CREATE DATABASE datathon
  CHARACTER SET utf8mb4
  COLLATE utf8mb4_unicode_ci;

USE datathon;

-- Chế độ strict giúp phát hiện lỗi định dạng dữ liệu sớm.
SET SESSION sql_mode = 'STRICT_TRANS_TABLES,NO_ENGINE_SUBSTITUTION';

-- Tắt FK khi drop/create để tránh lỗi thứ tự phụ thuộc.
SET FOREIGN_KEY_CHECKS = 0;

/* ================================================================================================
   1. DROP OBJECTS CŨ
================================================================================================ */

DROP VIEW IF EXISTS v_forecast_feature_daily;
DROP VIEW IF EXISTS v_customer_cohort_monthly;
DROP VIEW IF EXISTS v_reviews_daily;
DROP VIEW IF EXISTS v_returns_daily;
DROP VIEW IF EXISTS v_fulfillment_daily;
DROP VIEW IF EXISTS v_inventory_monthly;
DROP VIEW IF EXISTS v_conversion_daily;
DROP VIEW IF EXISTS v_web_traffic_daily;
DROP VIEW IF EXISTS v_revenue_reconciliation;
DROP VIEW IF EXISTS v_orders_daily_kpi;
DROP VIEW IF EXISTS v_sales_weekly;
DROP VIEW IF EXISTS v_sales_daily;
DROP VIEW IF EXISTS v_order_items_enriched;

DROP PROCEDURE IF EXISTS sp_validate_row_counts;
DROP PROCEDURE IF EXISTS sp_validate_fk_orphans;
DROP PROCEDURE IF EXISTS sp_get_forecast_features;
DROP PROCEDURE IF EXISTS sp_refresh_daily_feature_store;

DROP FUNCTION IF EXISTS fn_week_start;
DROP FUNCTION IF EXISTS fn_gross_margin_rate;
DROP FUNCTION IF EXISTS fn_discount_amount;
DROP FUNCTION IF EXISTS fn_order_net_value;

DROP TABLE IF EXISTS daily_feature_store;
DROP TABLE IF EXISTS web_traffic;
DROP TABLE IF EXISTS inventory;
DROP TABLE IF EXISTS sales;
DROP TABLE IF EXISTS reviews;
DROP TABLE IF EXISTS returns;
DROP TABLE IF EXISTS shipments;
DROP TABLE IF EXISTS payments;
DROP TABLE IF EXISTS order_items;
DROP TABLE IF EXISTS orders;
DROP TABLE IF EXISTS promotions;
DROP TABLE IF EXISTS customers;
DROP TABLE IF EXISTS products;
DROP TABLE IF EXISTS geography;

SET FOREIGN_KEY_CHECKS = 1;

/* ================================================================================================
   2. CREATE TABLES
   Ghi chú thiết kế:
   - Master tables: geography, products, customers, promotions.
   - Transaction tables: orders, order_items, payments, shipments, returns, reviews.
   - Analytical: sales.
   - Operational: inventory, web_traffic.
   - order_items không có khóa dòng trong CSV, nên thêm order_item_id AUTO_INCREMENT làm surrogate key.
   - web_traffic cũng thêm traffic_id để schema vẫn an toàn nếu sau này có nhiều nguồn traffic cùng ngày.
================================================================================================ */

CREATE TABLE geography (
    zip INT NOT NULL COMMENT 'Mã bưu chính, khóa chính',
    city VARCHAR(64) NOT NULL COMMENT 'Thành phố',
    region VARCHAR(32) NOT NULL COMMENT 'Vùng địa lý: East/West/Central',
    district VARCHAR(64) NOT NULL COMMENT 'Quận/huyện hoặc cụm địa lý',
    PRIMARY KEY (zip),
    KEY idx_geography_region_city (region, city),
    CONSTRAINT chk_geography_region CHECK (region IN ('East','West','Central'))
) ENGINE=InnoDB COMMENT='Master: danh sách mã bưu chính và vùng địa lý';

CREATE TABLE products (
    product_id INT NOT NULL COMMENT 'Khóa chính sản phẩm',
    product_name VARCHAR(128) NOT NULL COMMENT 'Tên sản phẩm',
    category VARCHAR(32) NOT NULL COMMENT 'Danh mục sản phẩm',
    segment VARCHAR(32) NOT NULL COMMENT 'Phân khúc sản phẩm',
    size VARCHAR(8) NOT NULL COMMENT 'Kích cỡ: S/M/L/XL',
    color VARCHAR(32) NOT NULL COMMENT 'Màu sắc',
    price DECIMAL(18,6) NOT NULL COMMENT 'Giá bán lẻ',
    cogs DECIMAL(18,6) NOT NULL COMMENT 'Giá vốn hàng bán',
    PRIMARY KEY (product_id),
    KEY idx_products_category_segment (category, segment),
    KEY idx_products_size_color (size, color),
    CONSTRAINT chk_products_price_positive CHECK (price > 0),
    CONSTRAINT chk_products_cogs_nonnegative CHECK (cogs >= 0),
    CONSTRAINT chk_products_cogs_lt_price CHECK (cogs < price),
    CONSTRAINT chk_products_size CHECK (size IN ('S','M','L','XL'))
) ENGINE=InnoDB COMMENT='Master: danh mục sản phẩm';

CREATE TABLE customers (
    customer_id INT NOT NULL COMMENT 'Khóa chính khách hàng',
    zip INT NOT NULL COMMENT 'Mã bưu chính khách hàng, FK sang geography.zip',
    city VARCHAR(64) NOT NULL COMMENT 'Thành phố khách hàng',
    signup_date DATE NOT NULL COMMENT 'Ngày đăng ký tài khoản',
    gender VARCHAR(16) NULL COMMENT 'Giới tính, nullable',
    age_group VARCHAR(16) NULL COMMENT 'Nhóm tuổi, nullable',
    acquisition_channel VARCHAR(32) NULL COMMENT 'Kênh thu hút khách hàng',
    PRIMARY KEY (customer_id),
    KEY idx_customers_zip (zip),
    KEY idx_customers_signup_date (signup_date),
    KEY idx_customers_segment (age_group, gender, acquisition_channel),
    CONSTRAINT chk_customers_gender CHECK (gender IS NULL OR gender IN ('Female','Male','Non-binary')),
    CONSTRAINT chk_customers_age_group CHECK (age_group IS NULL OR age_group IN ('18-24','25-34','35-44','45-54','55+'))
) ENGINE=InnoDB COMMENT='Master: thông tin khách hàng';

CREATE TABLE promotions (
    promo_id VARCHAR(16) NOT NULL COMMENT 'Khóa chính chương trình khuyến mãi',
    promo_name VARCHAR(128) NOT NULL COMMENT 'Tên chương trình',
    promo_type VARCHAR(16) NOT NULL COMMENT 'percentage hoặc fixed',
    discount_value DECIMAL(18,6) NOT NULL COMMENT 'Giá trị giảm theo % hoặc số tiền',
    start_date DATE NOT NULL COMMENT 'Ngày bắt đầu',
    end_date DATE NOT NULL COMMENT 'Ngày kết thúc',
    applicable_category VARCHAR(32) NULL COMMENT 'Danh mục áp dụng, NULL nếu toàn bộ',
    promo_channel VARCHAR(32) NULL COMMENT 'Kênh phân phối khuyến mãi',
    stackable_flag TINYINT NOT NULL DEFAULT 0 COMMENT '1 nếu cho phép cộng dồn',
    min_order_value DECIMAL(18,6) NULL COMMENT 'Giá trị đơn tối thiểu',
    PRIMARY KEY (promo_id),
    KEY idx_promotions_period (start_date, end_date),
    KEY idx_promotions_category_channel (applicable_category, promo_channel),
    CONSTRAINT chk_promotions_type CHECK (promo_type IN ('percentage','fixed')),
    CONSTRAINT chk_promotions_discount_nonnegative CHECK (discount_value >= 0),
    CONSTRAINT chk_promotions_period CHECK (end_date >= start_date),
    CONSTRAINT chk_promotions_stackable CHECK (stackable_flag IN (0,1))
) ENGINE=InnoDB COMMENT='Master: chương trình khuyến mãi';

CREATE TABLE orders (
    order_id INT NOT NULL COMMENT 'Khóa chính đơn hàng',
    order_date DATE NOT NULL COMMENT 'Ngày đặt hàng',
    customer_id INT NOT NULL COMMENT 'FK sang customers.customer_id',
    zip INT NOT NULL COMMENT 'Mã bưu chính giao hàng, FK sang geography.zip',
    order_status VARCHAR(16) NOT NULL COMMENT 'created/paid/shipped/delivered/returned/cancelled',
    payment_method VARCHAR(32) NOT NULL COMMENT 'Phương thức thanh toán ghi nhận tại đơn',
    device_type VARCHAR(16) NOT NULL COMMENT 'Thiết bị đặt hàng',
    order_source VARCHAR(32) NOT NULL COMMENT 'Nguồn/kênh tạo đơn',
    PRIMARY KEY (order_id),
    KEY idx_orders_date (order_date),
    KEY idx_orders_customer_date (customer_id, order_date),
    KEY idx_orders_zip (zip),
    KEY idx_orders_status_date (order_status, order_date),
    KEY idx_orders_source_device (order_source, device_type),
    CONSTRAINT chk_orders_status CHECK (order_status IN ('created','paid','shipped','delivered','returned','cancelled')),
    CONSTRAINT chk_orders_payment_method CHECK (payment_method IN ('apple_pay','bank_transfer','cod','credit_card','paypal')),
    CONSTRAINT chk_orders_device_type CHECK (device_type IN ('desktop','mobile','tablet'))
) ENGINE=InnoDB COMMENT='Transaction: thông tin đơn hàng';

CREATE TABLE order_items (
    order_item_id BIGINT NOT NULL AUTO_INCREMENT COMMENT 'Surrogate key vì CSV không có khóa dòng chi tiết',
    order_id INT NOT NULL COMMENT 'FK sang orders.order_id',
    product_id INT NOT NULL COMMENT 'FK sang products.product_id',
    quantity INT NOT NULL COMMENT 'Số lượng sản phẩm trong dòng đơn',
    unit_price DECIMAL(18,6) NOT NULL COMMENT 'Đơn giá tại thời điểm mua',
    discount_amount DECIMAL(18,6) NOT NULL DEFAULT 0 COMMENT 'Tổng giảm giá của dòng hàng',
    promo_id VARCHAR(16) NULL COMMENT 'Khuyến mãi thứ nhất, nullable',
    promo_id_2 VARCHAR(16) NULL COMMENT 'Khuyến mãi thứ hai, nullable',
    PRIMARY KEY (order_item_id),
    KEY idx_order_items_order (order_id),
    KEY idx_order_items_product (product_id),
    KEY idx_order_items_promo_1 (promo_id),
    KEY idx_order_items_promo_2 (promo_id_2),
    KEY idx_order_items_order_product (order_id, product_id),
    CONSTRAINT chk_order_items_quantity_positive CHECK (quantity > 0),
    CONSTRAINT chk_order_items_price_nonnegative CHECK (unit_price >= 0),
    CONSTRAINT chk_order_items_discount_nonnegative CHECK (discount_amount >= 0)
) ENGINE=InnoDB COMMENT='Transaction: chi tiết từng dòng sản phẩm trong đơn hàng';

CREATE TABLE payments (
    order_id INT NOT NULL COMMENT 'PK đồng thời là FK sang orders.order_id, quan hệ 1:1',
    payment_method VARCHAR(32) NOT NULL COMMENT 'Phương thức thanh toán',
    payment_value DECIMAL(18,6) NOT NULL COMMENT 'Giá trị thanh toán của đơn',
    installments INT NOT NULL COMMENT 'Số kỳ trả góp',
    PRIMARY KEY (order_id),
    KEY idx_payments_method_installments (payment_method, installments),
    CONSTRAINT chk_payments_value_nonnegative CHECK (payment_value >= 0),
    CONSTRAINT chk_payments_installments_positive CHECK (installments > 0),
    CONSTRAINT chk_payments_method CHECK (payment_method IN ('apple_pay','bank_transfer','cod','credit_card','paypal'))
) ENGINE=InnoDB COMMENT='Transaction: thanh toán, 1 đơn có 1 thanh toán';

CREATE TABLE shipments (
    order_id INT NOT NULL COMMENT 'PK đồng thời là FK sang orders.order_id, quan hệ 1:0/1',
    ship_date DATE NOT NULL COMMENT 'Ngày gửi hàng',
    delivery_date DATE NOT NULL COMMENT 'Ngày giao hàng',
    shipping_fee DECIMAL(18,6) NOT NULL DEFAULT 0 COMMENT 'Phí vận chuyển',
    PRIMARY KEY (order_id),
    KEY idx_shipments_ship_date (ship_date),
    KEY idx_shipments_delivery_date (delivery_date),
    CONSTRAINT chk_shipments_fee_nonnegative CHECK (shipping_fee >= 0),
    CONSTRAINT chk_shipments_delivery_after_ship CHECK (delivery_date >= ship_date)
) ENGINE=InnoDB COMMENT='Transaction: thông tin vận chuyển';

CREATE TABLE returns (
    return_id VARCHAR(16) NOT NULL COMMENT 'Khóa chính phiếu trả hàng',
    order_id INT NOT NULL COMMENT 'FK sang orders.order_id',
    product_id INT NOT NULL COMMENT 'FK sang products.product_id',
    return_date DATE NOT NULL COMMENT 'Ngày trả hàng',
    return_reason VARCHAR(32) NOT NULL COMMENT 'Lý do trả hàng',
    return_quantity INT NOT NULL COMMENT 'Số lượng trả',
    refund_amount DECIMAL(18,6) NOT NULL COMMENT 'Số tiền hoàn',
    PRIMARY KEY (return_id),
    KEY idx_returns_order_product (order_id, product_id),
    KEY idx_returns_date (return_date),
    KEY idx_returns_reason (return_reason),
    CONSTRAINT chk_returns_quantity_positive CHECK (return_quantity > 0),
    CONSTRAINT chk_returns_refund_nonnegative CHECK (refund_amount >= 0),
    CONSTRAINT chk_returns_reason CHECK (return_reason IN ('changed_mind','defective','late_delivery','not_as_described','wrong_size'))
) ENGINE=InnoDB COMMENT='Transaction: trả hàng và hoàn tiền';

CREATE TABLE reviews (
    review_id VARCHAR(16) NOT NULL COMMENT 'Khóa chính đánh giá',
    order_id INT NOT NULL COMMENT 'FK sang orders.order_id',
    product_id INT NOT NULL COMMENT 'FK sang products.product_id',
    customer_id INT NOT NULL COMMENT 'FK sang customers.customer_id',
    review_date DATE NOT NULL COMMENT 'Ngày đánh giá',
    rating INT NOT NULL COMMENT 'Điểm đánh giá 1-5',
    review_title VARCHAR(128) NOT NULL COMMENT 'Tiêu đề đánh giá',
    PRIMARY KEY (review_id),
    KEY idx_reviews_order_product (order_id, product_id),
    KEY idx_reviews_customer_date (customer_id, review_date),
    KEY idx_reviews_product_rating (product_id, rating),
    KEY idx_reviews_date (review_date),
    CONSTRAINT chk_reviews_rating CHECK (rating BETWEEN 1 AND 5)
) ENGINE=InnoDB COMMENT='Transaction: đánh giá sản phẩm';

CREATE TABLE sales (
    sales_date DATE NOT NULL COMMENT 'Ngày doanh thu, map từ cột Date của CSV',
    revenue DECIMAL(18,6) NOT NULL COMMENT 'Doanh thu thuần hằng ngày',
    cogs DECIMAL(18,6) NOT NULL COMMENT 'Tổng giá vốn hằng ngày',
    PRIMARY KEY (sales_date),
    CONSTRAINT chk_sales_revenue_nonnegative CHECK (revenue >= 0),
    CONSTRAINT chk_sales_cogs_nonnegative CHECK (cogs >= 0)
) ENGINE=InnoDB COMMENT='Analytical: doanh thu train theo ngày';

CREATE TABLE inventory (
    snapshot_date DATE NOT NULL COMMENT 'Ngày chụp tồn kho cuối tháng',
    product_id INT NOT NULL COMMENT 'FK sang products.product_id',
    stock_on_hand INT NOT NULL COMMENT 'Tồn kho cuối tháng',
    units_received INT NOT NULL COMMENT 'Nhập kho trong tháng',
    units_sold INT NOT NULL COMMENT 'Bán ra trong tháng',
    stockout_days INT NOT NULL COMMENT 'Số ngày hết hàng trong tháng',
    days_of_supply DECIMAL(18,6) NOT NULL COMMENT 'Số ngày tồn kho có thể đáp ứng',
    fill_rate DECIMAL(10,6) NOT NULL COMMENT 'Tỷ lệ đáp ứng từ tồn kho',
    stockout_flag TINYINT NOT NULL COMMENT '1 nếu có hết hàng',
    overstock_flag TINYINT NOT NULL COMMENT '1 nếu tồn kho vượt mức',
    reorder_flag TINYINT NOT NULL COMMENT '1 nếu cần tái đặt hàng',
    sell_through_rate DECIMAL(10,6) NOT NULL COMMENT 'Tỷ lệ bán qua tổng hàng sẵn có',
    product_name VARCHAR(128) NOT NULL COMMENT 'Tên sản phẩm snapshot',
    category VARCHAR(32) NOT NULL COMMENT 'Danh mục snapshot',
    segment VARCHAR(32) NOT NULL COMMENT 'Phân khúc snapshot',
    year INT NOT NULL COMMENT 'Năm snapshot',
    month INT NOT NULL COMMENT 'Tháng snapshot',
    PRIMARY KEY (snapshot_date, product_id),
    KEY idx_inventory_product_date (product_id, snapshot_date),
    KEY idx_inventory_category_segment (category, segment),
    KEY idx_inventory_year_month (year, month),
    CONSTRAINT chk_inventory_nonnegative_counts CHECK (
        stock_on_hand >= 0 AND units_received >= 0 AND units_sold >= 0 AND stockout_days >= 0
    ),
    CONSTRAINT chk_inventory_flags CHECK (
        stockout_flag IN (0,1) AND overstock_flag IN (0,1) AND reorder_flag IN (0,1)
    ),
    CONSTRAINT chk_inventory_month CHECK (month BETWEEN 1 AND 12)
) ENGINE=InnoDB COMMENT='Operational: tồn kho theo sản phẩm theo tháng';

CREATE TABLE web_traffic (
    traffic_id BIGINT NOT NULL AUTO_INCREMENT COMMENT 'Surrogate key cho log traffic',
    traffic_date DATE NOT NULL COMMENT 'Ngày ghi nhận traffic, map từ cột date của CSV',
    sessions INT NOT NULL COMMENT 'Số phiên truy cập',
    unique_visitors INT NOT NULL COMMENT 'Số người dùng duy nhất',
    page_views INT NOT NULL COMMENT 'Số lượt xem trang',
    bounce_rate DECIMAL(10,6) NOT NULL COMMENT 'Tỷ lệ thoát',
    avg_session_duration_sec DECIMAL(18,6) NOT NULL COMMENT 'Thời gian phiên trung bình',
    traffic_source VARCHAR(32) NOT NULL COMMENT 'Nguồn traffic',
    PRIMARY KEY (traffic_id),
    UNIQUE KEY uq_web_traffic_date_source (traffic_date, traffic_source),
    KEY idx_web_traffic_date (traffic_date),
    KEY idx_web_traffic_source_date (traffic_source, traffic_date),
    CONSTRAINT chk_web_traffic_nonnegative CHECK (
        sessions >= 0 AND unique_visitors >= 0 AND page_views >= 0 AND avg_session_duration_sec >= 0
    ),
    CONSTRAINT chk_web_traffic_bounce CHECK (bounce_rate >= 0 AND bounce_rate <= 1)
) ENGINE=InnoDB COMMENT='Operational: lưu lượng truy cập website';

/* ================================================================================================
   3. LOAD DATA LOCAL INFILE
   Ghi chú:
   - Dùng PREPARE để chỉ cần đổi @DATA_DIR một lần.
   - OPTIONALLY ENCLOSED BY '"' xử lý CSV có dấu ngoặc kép nếu xuất từ Excel/Pandas.
   - LINES TERMINATED BY '\r\n' vì các file hiện là CRLF. Nếu import 0 rows, thử đổi thành '\n'.
   - Các cột nullable được load qua biến @... rồi NULLIF(TRIM(@...), '').
================================================================================================ */

-- geography.csv
SET @sql = CONCAT(
'LOAD DATA LOCAL INFILE ', QUOTE(CONCAT(@DATA_DIR, 'geography.csv')), '
INTO TABLE geography
CHARACTER SET utf8mb4
FIELDS TERMINATED BY '','' OPTIONALLY ENCLOSED BY ''"''
LINES TERMINATED BY ''\r\n''
IGNORE 1 ROWS
(zip, city, region, district)'
);
PREPARE stmt FROM @sql; EXECUTE stmt; DEALLOCATE PREPARE stmt;

-- products.csv
SET @sql = CONCAT(
'LOAD DATA LOCAL INFILE ', QUOTE(CONCAT(@DATA_DIR, 'products.csv')), '
INTO TABLE products
CHARACTER SET utf8mb4
FIELDS TERMINATED BY '','' OPTIONALLY ENCLOSED BY ''"''
LINES TERMINATED BY ''\r\n''
IGNORE 1 ROWS
(product_id, product_name, category, segment, size, color, price, cogs)'
);
PREPARE stmt FROM @sql; EXECUTE stmt; DEALLOCATE PREPARE stmt;

-- customers.csv
SET @sql = CONCAT(
'LOAD DATA LOCAL INFILE ', QUOTE(CONCAT(@DATA_DIR, 'customers.csv')), '
INTO TABLE customers
CHARACTER SET utf8mb4
FIELDS TERMINATED BY '','' OPTIONALLY ENCLOSED BY ''"''
LINES TERMINATED BY ''\r\n''
IGNORE 1 ROWS
(@customer_id, @zip, @city, @signup_date, @gender, @age_group, @acquisition_channel)
SET
    customer_id = CAST(@customer_id AS UNSIGNED),
    zip = CAST(@zip AS UNSIGNED),
    city = TRIM(@city),
    signup_date = STR_TO_DATE(TRIM(@signup_date), ''%Y-%m-%d''),
    gender = NULLIF(TRIM(@gender), ''''),
    age_group = NULLIF(TRIM(@age_group), ''''),
    acquisition_channel = NULLIF(TRIM(@acquisition_channel), '''')'
);
PREPARE stmt FROM @sql; EXECUTE stmt; DEALLOCATE PREPARE stmt;

-- promotions.csv
SET @sql = CONCAT(
'LOAD DATA LOCAL INFILE ', QUOTE(CONCAT(@DATA_DIR, 'promotions.csv')), '
INTO TABLE promotions
CHARACTER SET utf8mb4
FIELDS TERMINATED BY '','' OPTIONALLY ENCLOSED BY ''"''
LINES TERMINATED BY ''\r\n''
IGNORE 1 ROWS
(@promo_id, @promo_name, @promo_type, @discount_value, @start_date, @end_date,
 @applicable_category, @promo_channel, @stackable_flag, @min_order_value)
SET
    promo_id = TRIM(@promo_id),
    promo_name = TRIM(@promo_name),
    promo_type = TRIM(@promo_type),
    discount_value = CAST(@discount_value AS DECIMAL(18,6)),
    start_date = STR_TO_DATE(TRIM(@start_date), ''%Y-%m-%d''),
    end_date = STR_TO_DATE(TRIM(@end_date), ''%Y-%m-%d''),
    applicable_category = NULLIF(TRIM(@applicable_category), ''''),
    promo_channel = NULLIF(TRIM(@promo_channel), ''''),
    stackable_flag = CAST(@stackable_flag AS UNSIGNED),
    min_order_value = CASE
        WHEN NULLIF(TRIM(@min_order_value), '''') IS NULL THEN NULL
        ELSE CAST(@min_order_value AS DECIMAL(18,6))
    END'
);
PREPARE stmt FROM @sql; EXECUTE stmt; DEALLOCATE PREPARE stmt;

-- sales.csv
SET @sql = CONCAT(
'LOAD DATA LOCAL INFILE ', QUOTE(CONCAT(@DATA_DIR, 'sales.csv')), '
INTO TABLE sales
CHARACTER SET utf8mb4
FIELDS TERMINATED BY '','' OPTIONALLY ENCLOSED BY ''"''
LINES TERMINATED BY ''\r\n''
IGNORE 1 ROWS
(@sales_date, @revenue, @cogs)
SET
    sales_date = STR_TO_DATE(TRIM(@sales_date), ''%Y-%m-%d''),
    revenue = CAST(@revenue AS DECIMAL(18,6)),
    cogs = CAST(@cogs AS DECIMAL(18,6))'
);
PREPARE stmt FROM @sql; EXECUTE stmt; DEALLOCATE PREPARE stmt;

-- orders.csv
SET @sql = CONCAT(
'LOAD DATA LOCAL INFILE ', QUOTE(CONCAT(@DATA_DIR, 'orders.csv')), '
INTO TABLE orders
CHARACTER SET utf8mb4
FIELDS TERMINATED BY '','' OPTIONALLY ENCLOSED BY ''"''
LINES TERMINATED BY ''\r\n''
IGNORE 1 ROWS
(@order_id, @order_date, @customer_id, @zip, @order_status, @payment_method, @device_type, @order_source)
SET
    order_id = CAST(@order_id AS UNSIGNED),
    order_date = STR_TO_DATE(TRIM(@order_date), ''%Y-%m-%d''),
    customer_id = CAST(@customer_id AS UNSIGNED),
    zip = CAST(@zip AS UNSIGNED),
    order_status = TRIM(@order_status),
    payment_method = TRIM(@payment_method),
    device_type = TRIM(@device_type),
    order_source = TRIM(@order_source)'
);
PREPARE stmt FROM @sql; EXECUTE stmt; DEALLOCATE PREPARE stmt;

-- order_items.csv
SET @sql = CONCAT(
'LOAD DATA LOCAL INFILE ', QUOTE(CONCAT(@DATA_DIR, 'order_items.csv')), '
INTO TABLE order_items
CHARACTER SET utf8mb4
FIELDS TERMINATED BY '','' OPTIONALLY ENCLOSED BY ''"''
LINES TERMINATED BY ''\r\n''
IGNORE 1 ROWS
(@order_id, @product_id, @quantity, @unit_price, @discount_amount, @promo_id, @promo_id_2)
SET
    order_id = CAST(@order_id AS UNSIGNED),
    product_id = CAST(@product_id AS UNSIGNED),
    quantity = CAST(@quantity AS UNSIGNED),
    unit_price = CAST(@unit_price AS DECIMAL(18,6)),
    discount_amount = CAST(@discount_amount AS DECIMAL(18,6)),
    promo_id = NULLIF(TRIM(@promo_id), ''''),
    promo_id_2 = NULLIF(TRIM(@promo_id_2), '''')'
);
PREPARE stmt FROM @sql; EXECUTE stmt; DEALLOCATE PREPARE stmt;

-- payments.csv
SET @sql = CONCAT(
'LOAD DATA LOCAL INFILE ', QUOTE(CONCAT(@DATA_DIR, 'payments.csv')), '
INTO TABLE payments
CHARACTER SET utf8mb4
FIELDS TERMINATED BY '','' OPTIONALLY ENCLOSED BY ''"''
LINES TERMINATED BY ''\r\n''
IGNORE 1 ROWS
(@order_id, @payment_method, @payment_value, @installments)
SET
    order_id = CAST(@order_id AS UNSIGNED),
    payment_method = TRIM(@payment_method),
    payment_value = CAST(@payment_value AS DECIMAL(18,6)),
    installments = CAST(@installments AS UNSIGNED)'
);
PREPARE stmt FROM @sql; EXECUTE stmt; DEALLOCATE PREPARE stmt;

-- shipments.csv
SET @sql = CONCAT(
'LOAD DATA LOCAL INFILE ', QUOTE(CONCAT(@DATA_DIR, 'shipments.csv')), '
INTO TABLE shipments
CHARACTER SET utf8mb4
FIELDS TERMINATED BY '','' OPTIONALLY ENCLOSED BY ''"''
LINES TERMINATED BY ''\r\n''
IGNORE 1 ROWS
(@order_id, @ship_date, @delivery_date, @shipping_fee)
SET
    order_id = CAST(@order_id AS UNSIGNED),
    ship_date = STR_TO_DATE(TRIM(@ship_date), ''%Y-%m-%d''),
    delivery_date = STR_TO_DATE(TRIM(@delivery_date), ''%Y-%m-%d''),
    shipping_fee = CAST(@shipping_fee AS DECIMAL(18,6))'
);
PREPARE stmt FROM @sql; EXECUTE stmt; DEALLOCATE PREPARE stmt;

-- returns.csv
SET @sql = CONCAT(
'LOAD DATA LOCAL INFILE ', QUOTE(CONCAT(@DATA_DIR, 'returns.csv')), '
INTO TABLE returns
CHARACTER SET utf8mb4
FIELDS TERMINATED BY '','' OPTIONALLY ENCLOSED BY ''"''
LINES TERMINATED BY ''\r\n''
IGNORE 1 ROWS
(@return_id, @order_id, @product_id, @return_date, @return_reason, @return_quantity, @refund_amount)
SET
    return_id = TRIM(@return_id),
    order_id = CAST(@order_id AS UNSIGNED),
    product_id = CAST(@product_id AS UNSIGNED),
    return_date = STR_TO_DATE(TRIM(@return_date), ''%Y-%m-%d''),
    return_reason = TRIM(@return_reason),
    return_quantity = CAST(@return_quantity AS UNSIGNED),
    refund_amount = CAST(@refund_amount AS DECIMAL(18,6))'
);
PREPARE stmt FROM @sql; EXECUTE stmt; DEALLOCATE PREPARE stmt;

-- reviews.csv
SET @sql = CONCAT(
'LOAD DATA LOCAL INFILE ', QUOTE(CONCAT(@DATA_DIR, 'reviews.csv')), '
INTO TABLE reviews
CHARACTER SET utf8mb4
FIELDS TERMINATED BY '','' OPTIONALLY ENCLOSED BY ''"''
LINES TERMINATED BY ''\r\n''
IGNORE 1 ROWS
(@review_id, @order_id, @product_id, @customer_id, @review_date, @rating, @review_title)
SET
    review_id = TRIM(@review_id),
    order_id = CAST(@order_id AS UNSIGNED),
    product_id = CAST(@product_id AS UNSIGNED),
    customer_id = CAST(@customer_id AS UNSIGNED),
    review_date = STR_TO_DATE(TRIM(@review_date), ''%Y-%m-%d''),
    rating = CAST(@rating AS UNSIGNED),
    review_title = TRIM(@review_title)'
);
PREPARE stmt FROM @sql; EXECUTE stmt; DEALLOCATE PREPARE stmt;

-- inventory.csv
SET @sql = CONCAT(
'LOAD DATA LOCAL INFILE ', QUOTE(CONCAT(@DATA_DIR, 'inventory.csv')), '
INTO TABLE inventory
CHARACTER SET utf8mb4
FIELDS TERMINATED BY '','' OPTIONALLY ENCLOSED BY ''"''
LINES TERMINATED BY ''\r\n''
IGNORE 1 ROWS
(@snapshot_date, @product_id, @stock_on_hand, @units_received, @units_sold, @stockout_days,
 @days_of_supply, @fill_rate, @stockout_flag, @overstock_flag, @reorder_flag, @sell_through_rate,
 @product_name, @category, @segment, @year, @month)
SET
    snapshot_date = STR_TO_DATE(TRIM(@snapshot_date), ''%Y-%m-%d''),
    product_id = CAST(@product_id AS UNSIGNED),
    stock_on_hand = CAST(@stock_on_hand AS UNSIGNED),
    units_received = CAST(@units_received AS UNSIGNED),
    units_sold = CAST(@units_sold AS UNSIGNED),
    stockout_days = CAST(@stockout_days AS UNSIGNED),
    days_of_supply = CAST(@days_of_supply AS DECIMAL(18,6)),
    fill_rate = CAST(@fill_rate AS DECIMAL(10,6)),
    stockout_flag = CAST(@stockout_flag AS UNSIGNED),
    overstock_flag = CAST(@overstock_flag AS UNSIGNED),
    reorder_flag = CAST(@reorder_flag AS UNSIGNED),
    sell_through_rate = CAST(@sell_through_rate AS DECIMAL(10,6)),
    product_name = TRIM(@product_name),
    category = TRIM(@category),
    segment = TRIM(@segment),
    year = CAST(@year AS UNSIGNED),
    month = CAST(@month AS UNSIGNED)'
);
PREPARE stmt FROM @sql; EXECUTE stmt; DEALLOCATE PREPARE stmt;

-- web_traffic.csv
SET @sql = CONCAT(
'LOAD DATA LOCAL INFILE ', QUOTE(CONCAT(@DATA_DIR, 'web_traffic.csv')), '
INTO TABLE web_traffic
CHARACTER SET utf8mb4
FIELDS TERMINATED BY '','' OPTIONALLY ENCLOSED BY ''"''
LINES TERMINATED BY ''\r\n''
IGNORE 1 ROWS
(@traffic_date, @sessions, @unique_visitors, @page_views, @bounce_rate, @avg_session_duration_sec, @traffic_source)
SET
    traffic_date = STR_TO_DATE(TRIM(@traffic_date), ''%Y-%m-%d''),
    sessions = CAST(@sessions AS UNSIGNED),
    unique_visitors = CAST(@unique_visitors AS UNSIGNED),
    page_views = CAST(@page_views AS UNSIGNED),
    bounce_rate = CAST(@bounce_rate AS DECIMAL(10,6)),
    avg_session_duration_sec = CAST(@avg_session_duration_sec AS DECIMAL(18,6)),
    traffic_source = TRIM(@traffic_source)'
);
PREPARE stmt FROM @sql; EXECUTE stmt; DEALLOCATE PREPARE stmt;

/* ================================================================================================
   4. ADD FOREIGN KEYS
   Chạy sau khi load để:
   - import nhanh hơn;
   - lỗi FK, nếu có, hiện rõ tại bước này;
   - tránh tình trạng LOAD bị dừng giữa chừng nhưng khó biết nguyên nhân.
================================================================================================ */

ALTER TABLE customers
    ADD CONSTRAINT fk_customers_geography
    FOREIGN KEY (zip) REFERENCES geography(zip)
    ON UPDATE CASCADE ON DELETE RESTRICT;

ALTER TABLE orders
    ADD CONSTRAINT fk_orders_customers
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
    ON UPDATE CASCADE ON DELETE RESTRICT,
    ADD CONSTRAINT fk_orders_geography
    FOREIGN KEY (zip) REFERENCES geography(zip)
    ON UPDATE CASCADE ON DELETE RESTRICT;

ALTER TABLE order_items
    ADD CONSTRAINT fk_order_items_orders
    FOREIGN KEY (order_id) REFERENCES orders(order_id)
    ON UPDATE CASCADE ON DELETE RESTRICT,
    ADD CONSTRAINT fk_order_items_products
    FOREIGN KEY (product_id) REFERENCES products(product_id)
    ON UPDATE CASCADE ON DELETE RESTRICT,
    ADD CONSTRAINT fk_order_items_promotions_1
    FOREIGN KEY (promo_id) REFERENCES promotions(promo_id)
    ON UPDATE CASCADE ON DELETE SET NULL,
    ADD CONSTRAINT fk_order_items_promotions_2
    FOREIGN KEY (promo_id_2) REFERENCES promotions(promo_id)
    ON UPDATE CASCADE ON DELETE SET NULL;

ALTER TABLE payments
    ADD CONSTRAINT fk_payments_orders
    FOREIGN KEY (order_id) REFERENCES orders(order_id)
    ON UPDATE CASCADE ON DELETE RESTRICT;

ALTER TABLE shipments
    ADD CONSTRAINT fk_shipments_orders
    FOREIGN KEY (order_id) REFERENCES orders(order_id)
    ON UPDATE CASCADE ON DELETE RESTRICT;

ALTER TABLE returns
    ADD CONSTRAINT fk_returns_orders
    FOREIGN KEY (order_id) REFERENCES orders(order_id)
    ON UPDATE CASCADE ON DELETE RESTRICT,
    ADD CONSTRAINT fk_returns_products
    FOREIGN KEY (product_id) REFERENCES products(product_id)
    ON UPDATE CASCADE ON DELETE RESTRICT;

ALTER TABLE reviews
    ADD CONSTRAINT fk_reviews_orders
    FOREIGN KEY (order_id) REFERENCES orders(order_id)
    ON UPDATE CASCADE ON DELETE RESTRICT,
    ADD CONSTRAINT fk_reviews_products
    FOREIGN KEY (product_id) REFERENCES products(product_id)
    ON UPDATE CASCADE ON DELETE RESTRICT,
    ADD CONSTRAINT fk_reviews_customers
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
    ON UPDATE CASCADE ON DELETE RESTRICT;

ALTER TABLE inventory
    ADD CONSTRAINT fk_inventory_products
    FOREIGN KEY (product_id) REFERENCES products(product_id)
    ON UPDATE CASCADE ON DELETE RESTRICT;

/* ================================================================================================
   5. FUNCTIONS
================================================================================================ */

DELIMITER $$

CREATE FUNCTION fn_week_start(p_date DATE)
RETURNS DATE
DETERMINISTIC
NO SQL
COMMENT 'Trả về ngày thứ Hai đầu tuần của một ngày bất kỳ'
BEGIN
    RETURN DATE_SUB(p_date, INTERVAL WEEKDAY(p_date) DAY);
END$$

CREATE FUNCTION fn_gross_margin_rate(p_revenue DECIMAL(18,6), p_cogs DECIMAL(18,6))
RETURNS DECIMAL(18,6)
DETERMINISTIC
NO SQL
COMMENT 'Tính tỷ suất lợi nhuận gộp = (Revenue - COGS) / Revenue'
BEGIN
    IF p_revenue IS NULL OR p_revenue = 0 THEN
        RETURN NULL;
    END IF;

    RETURN (p_revenue - COALESCE(p_cogs, 0)) / p_revenue;
END$$

CREATE FUNCTION fn_discount_amount(
    p_promo_type VARCHAR(16),
    p_quantity INT,
    p_unit_price DECIMAL(18,6),
    p_discount_value DECIMAL(18,6)
)
RETURNS DECIMAL(18,6)
DETERMINISTIC
NO SQL
COMMENT 'Tính số tiền giảm giá theo logic percentage/fixed của promotions'
BEGIN
    IF p_promo_type IS NULL OR p_discount_value IS NULL THEN
        RETURN 0;
    END IF;

    IF p_promo_type = 'percentage' THEN
        RETURN COALESCE(p_quantity,0) * COALESCE(p_unit_price,0) * p_discount_value / 100;
    ELSEIF p_promo_type = 'fixed' THEN
        RETURN COALESCE(p_quantity,0) * p_discount_value;
    END IF;

    RETURN 0;
END$$

CREATE FUNCTION fn_order_net_value(p_order_id INT)
RETURNS DECIMAL(18,6)
READS SQL DATA
COMMENT 'Tính net merchandise value của một đơn từ order_items'
BEGIN
    DECLARE v_net DECIMAL(18,6);

    SELECT COALESCE(SUM(quantity * unit_price - discount_amount), 0)
    INTO v_net
    FROM order_items
    WHERE order_id = p_order_id;

    RETURN v_net;
END$$

DELIMITER ;

/* ================================================================================================
   6. VIEWS PHỤC VỤ EDA, POWER BI, FORECASTING
================================================================================================ */

CREATE OR REPLACE VIEW v_sales_daily AS
SELECT
    sales_date,
    revenue,
    cogs,
    revenue - cogs AS gross_profit,
    fn_gross_margin_rate(revenue, cogs) AS gross_margin_rate
FROM sales;

CREATE OR REPLACE VIEW v_sales_weekly AS
SELECT
    fn_week_start(sales_date) AS week_start,
    YEARWEEK(sales_date, 3) AS iso_year_week,
    SUM(revenue) AS weekly_revenue,
    SUM(cogs) AS weekly_cogs,
    SUM(revenue - cogs) AS weekly_gross_profit,
    fn_gross_margin_rate(SUM(revenue), SUM(cogs)) AS weekly_gross_margin_rate,
    COUNT(*) AS days_in_week
FROM sales
GROUP BY fn_week_start(sales_date), YEARWEEK(sales_date, 3);

CREATE OR REPLACE VIEW v_order_items_enriched AS
SELECT
    oi.order_item_id,
    oi.order_id,
    o.order_date,
    o.customer_id,
    o.zip,
    g.city AS ship_city,
    g.region AS ship_region,
    o.order_status,
    o.payment_method AS order_payment_method,
    o.device_type,
    o.order_source,
    oi.product_id,
    p.product_name,
    p.category,
    p.segment,
    p.size,
    p.color,
    oi.quantity,
    oi.unit_price,
    p.cogs AS product_cogs,
    oi.quantity * oi.unit_price AS gross_line_revenue,
    oi.discount_amount,
    GREATEST(oi.quantity * oi.unit_price - oi.discount_amount, 0) AS net_line_revenue_before_refund,
    oi.quantity * p.cogs AS estimated_line_cogs,
    GREATEST(oi.quantity * oi.unit_price - oi.discount_amount, 0) - (oi.quantity * p.cogs) AS estimated_line_gross_profit,
    oi.promo_id,
    pr1.promo_name AS promo_name_1,
    pr1.promo_type AS promo_type_1,
    oi.promo_id_2,
    pr2.promo_name AS promo_name_2,
    pr2.promo_type AS promo_type_2
FROM order_items oi
JOIN orders o
    ON o.order_id = oi.order_id
JOIN products p
    ON p.product_id = oi.product_id
LEFT JOIN geography g
    ON g.zip = o.zip
LEFT JOIN promotions pr1
    ON pr1.promo_id = oi.promo_id
LEFT JOIN promotions pr2
    ON pr2.promo_id = oi.promo_id_2;

CREATE OR REPLACE VIEW v_orders_daily_kpi AS
SELECT
    o.order_date,
    COUNT(DISTINCT o.order_id) AS total_orders,
    COUNT(DISTINCT CASE WHEN o.order_status = 'cancelled' THEN o.order_id END) AS cancelled_orders,
    COUNT(DISTINCT CASE WHEN o.order_status = 'returned' THEN o.order_id END) AS returned_orders,
    COUNT(DISTINCT o.customer_id) AS active_customers,
    SUM(oi.quantity) AS units_sold,
    SUM(oi.quantity * oi.unit_price) AS gross_item_revenue,
    SUM(oi.discount_amount) AS discount_amount,
    SUM(GREATEST(oi.quantity * oi.unit_price - oi.discount_amount, 0)) AS net_item_revenue_before_refund,
    CASE
        WHEN COUNT(DISTINCT o.order_id) = 0 THEN NULL
        ELSE SUM(GREATEST(oi.quantity * oi.unit_price - oi.discount_amount, 0)) / COUNT(DISTINCT o.order_id)
    END AS avg_order_value,
    COUNT(DISTINCT CASE WHEN oi.promo_id IS NOT NULL OR oi.promo_id_2 IS NOT NULL THEN o.order_id END) AS promo_orders
FROM orders o
LEFT JOIN order_items oi
    ON oi.order_id = o.order_id
GROUP BY o.order_date;

CREATE OR REPLACE VIEW v_revenue_reconciliation AS
SELECT
    s.sales_date,
    s.revenue AS sales_csv_revenue,
    s.cogs AS sales_csv_cogs,
    COALESCE(k.net_item_revenue_before_refund, 0) AS transaction_net_item_revenue_before_refund,
    COALESCE(k.discount_amount, 0) AS transaction_discount_amount,
    s.revenue - COALESCE(k.net_item_revenue_before_refund, 0) AS revenue_difference,
    CASE
        WHEN s.revenue = 0 THEN NULL
        ELSE (s.revenue - COALESCE(k.net_item_revenue_before_refund, 0)) / s.revenue
    END AS revenue_difference_pct
FROM sales s
LEFT JOIN v_orders_daily_kpi k
    ON k.order_date = s.sales_date;

CREATE OR REPLACE VIEW v_web_traffic_daily AS
SELECT
    traffic_date,
    SUM(sessions) AS sessions,
    SUM(unique_visitors) AS unique_visitors,
    SUM(page_views) AS page_views,
    CASE
        WHEN SUM(sessions) = 0 THEN NULL
        ELSE SUM(bounce_rate * sessions) / SUM(sessions)
    END AS weighted_bounce_rate,
    CASE
        WHEN SUM(sessions) = 0 THEN NULL
        ELSE SUM(avg_session_duration_sec * sessions) / SUM(sessions)
    END AS weighted_avg_session_duration_sec,
    COUNT(DISTINCT traffic_source) AS traffic_source_count
FROM web_traffic
GROUP BY traffic_date;

CREATE OR REPLACE VIEW v_conversion_daily AS
SELECT
    s.sales_date,
    s.revenue,
    wt.sessions,
    wt.unique_visitors,
    wt.page_views,
    wt.weighted_bounce_rate,
    wt.weighted_avg_session_duration_sec,
    k.total_orders,
    k.active_customers,
    CASE WHEN wt.sessions > 0 THEN k.total_orders / wt.sessions ELSE NULL END AS order_conversion_rate,
    CASE WHEN wt.unique_visitors > 0 THEN k.total_orders / wt.unique_visitors ELSE NULL END AS visitor_conversion_rate,
    CASE WHEN wt.sessions > 0 THEN s.revenue / wt.sessions ELSE NULL END AS revenue_per_session,
    CASE WHEN k.total_orders > 0 THEN s.revenue / k.total_orders ELSE NULL END AS revenue_per_order
FROM sales s
LEFT JOIN v_web_traffic_daily wt
    ON wt.traffic_date = s.sales_date
LEFT JOIN v_orders_daily_kpi k
    ON k.order_date = s.sales_date;

CREATE OR REPLACE VIEW v_inventory_monthly AS
SELECT
    snapshot_date,
    YEAR(snapshot_date) AS year,
    MONTH(snapshot_date) AS month,
    COUNT(DISTINCT product_id) AS sku_count,
    SUM(stock_on_hand) AS total_stock_on_hand,
    SUM(units_received) AS total_units_received,
    SUM(units_sold) AS total_units_sold,
    SUM(stockout_days) AS total_stockout_days,
    AVG(stockout_days) AS avg_stockout_days,
    AVG(days_of_supply) AS avg_days_of_supply,
    AVG(fill_rate) AS avg_fill_rate,
    AVG(sell_through_rate) AS avg_sell_through_rate,
    AVG(stockout_flag) AS stockout_sku_share,
    AVG(overstock_flag) AS overstock_sku_share,
    AVG(reorder_flag) AS reorder_sku_share
FROM inventory
GROUP BY snapshot_date, YEAR(snapshot_date), MONTH(snapshot_date);

CREATE OR REPLACE VIEW v_fulfillment_daily AS
SELECT
    o.order_date,
    COUNT(s.order_id) AS shipped_orders,
    AVG(DATEDIFF(s.ship_date, o.order_date)) AS avg_days_to_ship,
    AVG(DATEDIFF(s.delivery_date, s.ship_date)) AS avg_days_in_transit,
    AVG(DATEDIFF(s.delivery_date, o.order_date)) AS avg_days_order_to_delivery,
    SUM(s.shipping_fee) AS total_shipping_fee,
    AVG(s.shipping_fee) AS avg_shipping_fee
FROM orders o
JOIN shipments s
    ON s.order_id = o.order_id
GROUP BY o.order_date;

CREATE OR REPLACE VIEW v_returns_daily AS
SELECT
    return_date,
    COUNT(*) AS return_lines,
    COUNT(DISTINCT order_id) AS returned_orders,
    SUM(return_quantity) AS returned_units,
    SUM(refund_amount) AS refund_amount,
    AVG(refund_amount) AS avg_refund_amount
FROM returns
GROUP BY return_date;

CREATE OR REPLACE VIEW v_reviews_daily AS
SELECT
    review_date,
    COUNT(*) AS review_count,
    AVG(rating) AS avg_rating,
    SUM(CASE WHEN rating <= 2 THEN 1 ELSE 0 END) AS low_rating_count,
    SUM(CASE WHEN rating >= 4 THEN 1 ELSE 0 END) AS high_rating_count
FROM reviews
GROUP BY review_date;

CREATE OR REPLACE VIEW v_customer_cohort_monthly AS
SELECT
    DATE_FORMAT(c.signup_date, '%Y-%m-01') AS signup_month,
    DATE_FORMAT(o.order_date, '%Y-%m-01') AS order_month,
    PERIOD_DIFF(EXTRACT(YEAR_MONTH FROM o.order_date), EXTRACT(YEAR_MONTH FROM c.signup_date)) AS cohort_age_month,
    COUNT(DISTINCT c.customer_id) AS customers,
    COUNT(DISTINCT o.order_id) AS orders,
    SUM(oi.quantity * oi.unit_price - oi.discount_amount) AS net_item_revenue_before_refund
FROM customers c
JOIN orders o
    ON o.customer_id = c.customer_id
JOIN order_items oi
    ON oi.order_id = o.order_id
GROUP BY
    DATE_FORMAT(c.signup_date, '%Y-%m-01'),
    DATE_FORMAT(o.order_date, '%Y-%m-01'),
    PERIOD_DIFF(EXTRACT(YEAR_MONTH FROM o.order_date), EXTRACT(YEAR_MONTH FROM c.signup_date));

CREATE OR REPLACE VIEW v_forecast_feature_daily AS
SELECT
    s.sales_date AS feature_date,

    -- Target chính.
    s.revenue,
    s.cogs,
    s.revenue - s.cogs AS gross_profit,
    fn_gross_margin_rate(s.revenue, s.cogs) AS gross_margin_rate,

    -- Calendar features.
    YEAR(s.sales_date) AS year,
    MONTH(s.sales_date) AS month,
    DAYOFMONTH(s.sales_date) AS day_of_month,
    DAYOFWEEK(s.sales_date) AS day_of_week,
    WEEK(s.sales_date, 3) AS iso_week,
    QUARTER(s.sales_date) AS quarter,
    CASE WHEN DAYOFWEEK(s.sales_date) IN (1,7) THEN 1 ELSE 0 END AS is_weekend,

    -- Time-series memory features. Các biến này chỉ dùng quá khứ nên tránh leakage.
    LAG(s.revenue, 1) OVER (ORDER BY s.sales_date) AS revenue_lag_1,
    LAG(s.revenue, 7) OVER (ORDER BY s.sales_date) AS revenue_lag_7,
    LAG(s.revenue, 14) OVER (ORDER BY s.sales_date) AS revenue_lag_14,
    LAG(s.revenue, 28) OVER (ORDER BY s.sales_date) AS revenue_lag_28,
    AVG(s.revenue) OVER (ORDER BY s.sales_date ROWS BETWEEN 7 PRECEDING AND 1 PRECEDING) AS revenue_ma_7,
    AVG(s.revenue) OVER (ORDER BY s.sales_date ROWS BETWEEN 28 PRECEDING AND 1 PRECEDING) AS revenue_ma_28,

    -- Demand / traffic.
    wt.sessions,
    wt.unique_visitors,
    wt.page_views,
    wt.weighted_bounce_rate,
    wt.weighted_avg_session_duration_sec,

    -- Conversion / order behavior.
    k.total_orders,
    k.cancelled_orders,
    k.returned_orders,
    k.active_customers,
    k.units_sold,
    k.discount_amount,
    k.promo_orders,
    CASE WHEN wt.sessions > 0 THEN k.total_orders / wt.sessions ELSE NULL END AS order_conversion_rate,
    CASE WHEN k.total_orders > 0 THEN s.revenue / k.total_orders ELSE NULL END AS revenue_per_order,

    -- Promotion regime biết được theo lịch campaign.
    (
        SELECT COUNT(*)
        FROM promotions p
        WHERE s.sales_date BETWEEN p.start_date AND p.end_date
    ) AS active_promo_count,
    (
        SELECT AVG(p.discount_value)
        FROM promotions p
        WHERE s.sales_date BETWEEN p.start_date AND p.end_date
    ) AS avg_active_discount_value,

    -- Fulfillment / leakage / review quality.
    f.shipped_orders,
    f.avg_days_order_to_delivery,
    r.return_lines,
    r.returned_units,
    r.refund_amount,
    rv.review_count,
    rv.avg_rating,
    rv.low_rating_count,

    -- Inventory monthly snapshot được map theo tháng của ngày bán.
    im.total_stock_on_hand,
    im.total_units_received,
    im.total_units_sold,
    im.avg_stockout_days,
    im.avg_fill_rate,
    im.stockout_sku_share,
    im.overstock_sku_share,
    im.reorder_sku_share
FROM sales s
LEFT JOIN v_web_traffic_daily wt
    ON wt.traffic_date = s.sales_date
LEFT JOIN v_orders_daily_kpi k
    ON k.order_date = s.sales_date
LEFT JOIN v_fulfillment_daily f
    ON f.order_date = s.sales_date
LEFT JOIN v_returns_daily r
    ON r.return_date = s.sales_date
LEFT JOIN v_reviews_daily rv
    ON rv.review_date = s.sales_date
LEFT JOIN v_inventory_monthly im
    ON YEAR(im.snapshot_date) = YEAR(s.sales_date)
   AND MONTH(im.snapshot_date) = MONTH(s.sales_date);

/* ================================================================================================
   7. STORED PROCEDURES
================================================================================================ */

DELIMITER $$

CREATE PROCEDURE sp_validate_row_counts()
COMMENT 'So sánh số dòng thực tế sau import với số dòng kỳ vọng của các file CSV'
BEGIN
    SELECT 'geography' AS table_name, 39948 AS expected_rows, COUNT(*) AS actual_rows, COUNT(*) - 39948 AS diff FROM geography
    UNION ALL SELECT 'products', 2412, COUNT(*), COUNT(*) - 2412 FROM products
    UNION ALL SELECT 'customers', 121930, COUNT(*), COUNT(*) - 121930 FROM customers
    UNION ALL SELECT 'promotions', 50, COUNT(*), COUNT(*) - 50 FROM promotions
    UNION ALL SELECT 'orders', 646945, COUNT(*), COUNT(*) - 646945 FROM orders
    UNION ALL SELECT 'order_items', 714669, COUNT(*), COUNT(*) - 714669 FROM order_items
    UNION ALL SELECT 'payments', 646945, COUNT(*), COUNT(*) - 646945 FROM payments
    UNION ALL SELECT 'shipments', 566067, COUNT(*), COUNT(*) - 566067 FROM shipments
    UNION ALL SELECT 'returns', 39939, COUNT(*), COUNT(*) - 39939 FROM returns
    UNION ALL SELECT 'reviews', 113551, COUNT(*), COUNT(*) - 113551 FROM reviews
    UNION ALL SELECT 'sales', 3833, COUNT(*), COUNT(*) - 3833 FROM sales
    UNION ALL SELECT 'inventory', 60247, COUNT(*), COUNT(*) - 60247 FROM inventory
    UNION ALL SELECT 'web_traffic', 3652, COUNT(*), COUNT(*) - 3652 FROM web_traffic;
END$$

CREATE PROCEDURE sp_validate_fk_orphans()
COMMENT 'Kiểm tra orphan records giữa các bảng chính; nếu orphan_count = 0 là sạch'
BEGIN
    SELECT 'customers.zip -> geography.zip' AS relationship, COUNT(*) AS orphan_count
    FROM customers c LEFT JOIN geography g ON g.zip = c.zip
    WHERE g.zip IS NULL

    UNION ALL
    SELECT 'orders.customer_id -> customers.customer_id', COUNT(*)
    FROM orders o LEFT JOIN customers c ON c.customer_id = o.customer_id
    WHERE c.customer_id IS NULL

    UNION ALL
    SELECT 'orders.zip -> geography.zip', COUNT(*)
    FROM orders o LEFT JOIN geography g ON g.zip = o.zip
    WHERE g.zip IS NULL

    UNION ALL
    SELECT 'order_items.order_id -> orders.order_id', COUNT(*)
    FROM order_items oi LEFT JOIN orders o ON o.order_id = oi.order_id
    WHERE o.order_id IS NULL

    UNION ALL
    SELECT 'order_items.product_id -> products.product_id', COUNT(*)
    FROM order_items oi LEFT JOIN products p ON p.product_id = oi.product_id
    WHERE p.product_id IS NULL

    UNION ALL
    SELECT 'order_items.promo_id -> promotions.promo_id', COUNT(*)
    FROM order_items oi LEFT JOIN promotions p ON p.promo_id = oi.promo_id
    WHERE oi.promo_id IS NOT NULL AND p.promo_id IS NULL

    UNION ALL
    SELECT 'order_items.promo_id_2 -> promotions.promo_id', COUNT(*)
    FROM order_items oi LEFT JOIN promotions p ON p.promo_id = oi.promo_id_2
    WHERE oi.promo_id_2 IS NOT NULL AND p.promo_id IS NULL

    UNION ALL
    SELECT 'payments.order_id -> orders.order_id', COUNT(*)
    FROM payments p LEFT JOIN orders o ON o.order_id = p.order_id
    WHERE o.order_id IS NULL

    UNION ALL
    SELECT 'shipments.order_id -> orders.order_id', COUNT(*)
    FROM shipments s LEFT JOIN orders o ON o.order_id = s.order_id
    WHERE o.order_id IS NULL

    UNION ALL
    SELECT 'returns.order_id -> orders.order_id', COUNT(*)
    FROM returns r LEFT JOIN orders o ON o.order_id = r.order_id
    WHERE o.order_id IS NULL

    UNION ALL
    SELECT 'returns.product_id -> products.product_id', COUNT(*)
    FROM returns r LEFT JOIN products p ON p.product_id = r.product_id
    WHERE p.product_id IS NULL

    UNION ALL
    SELECT 'reviews.order_id -> orders.order_id', COUNT(*)
    FROM reviews r LEFT JOIN orders o ON o.order_id = r.order_id
    WHERE o.order_id IS NULL

    UNION ALL
    SELECT 'reviews.product_id -> products.product_id', COUNT(*)
    FROM reviews r LEFT JOIN products p ON p.product_id = r.product_id
    WHERE p.product_id IS NULL

    UNION ALL
    SELECT 'reviews.customer_id -> customers.customer_id', COUNT(*)
    FROM reviews r LEFT JOIN customers c ON c.customer_id = r.customer_id
    WHERE c.customer_id IS NULL

    UNION ALL
    SELECT 'inventory.product_id -> products.product_id', COUNT(*)
    FROM inventory i LEFT JOIN products p ON p.product_id = i.product_id
    WHERE p.product_id IS NULL;
END$$

CREATE PROCEDURE sp_get_forecast_features(
    IN p_start_date DATE,
    IN p_end_date DATE
)
COMMENT 'Lấy feature daily trong một khoảng ngày để export sang Python/Power BI'
BEGIN
    SELECT *
    FROM v_forecast_feature_daily
    WHERE feature_date BETWEEN p_start_date AND p_end_date
    ORDER BY feature_date;
END$$

CREATE PROCEDURE sp_refresh_daily_feature_store()
COMMENT 'Materialize v_forecast_feature_daily thành bảng daily_feature_store để query/export nhanh hơn'
BEGIN
    DROP TABLE IF EXISTS daily_feature_store;

    CREATE TABLE daily_feature_store AS
    SELECT *
    FROM v_forecast_feature_daily;

    ALTER TABLE daily_feature_store
        ADD PRIMARY KEY (feature_date);

    CREATE INDEX idx_daily_feature_store_year_month
        ON daily_feature_store (year, month);

    CREATE INDEX idx_daily_feature_store_revenue
        ON daily_feature_store (revenue);
END$$

DELIMITER ;

/* ================================================================================================
   8. QUICK CHECK SAU KHI CHẠY SCRIPT
================================================================================================ */

CALL sp_validate_row_counts();
CALL sp_validate_fk_orphans();

-- Tạo bảng feature materialized nếu muốn export nhanh sang Python.
CALL sp_refresh_daily_feature_store();

-- Một vài câu kiểm tra nhanh.
SELECT * FROM v_sales_daily ORDER BY sales_date LIMIT 10;
SELECT * FROM v_sales_weekly ORDER BY week_start LIMIT 10;
SELECT * FROM v_forecast_feature_daily ORDER BY feature_date LIMIT 10;
