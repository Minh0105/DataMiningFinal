# 🎓 HỆ THỐNG DỰ BÁO ĐIỂM CHUẨN ĐẠI HỌC Y DƯỢC 2026

## Thông tin nhóm
- **Nhóm:** [Điền tên nhóm]
- **Thành viên:** [Điền danh sách]
- **Ngày hoàn thành:** 30/11/2025

---

# 📋 MỤC LỤC

1. [Giới thiệu đề tài](#1-giới-thiệu-đề-tài)
2. [Thu thập dữ liệu](#2-thu-thập-dữ-liệu)
3. [Xử lý và làm sạch dữ liệu](#3-xử-lý-và-làm-sạch-dữ-liệu)
4. [Phương pháp & Mô hình](#4-phương-pháp--mô-hình)
5. [Kết quả Training](#5-kết-quả-training)
6. [Web Application](#6-web-application)
7. [Kết luận](#7-kết-luận)

---

# 1. GIỚI THIỆU ĐỀ TÀI

## 1.1 Bối cảnh
- Điểm chuẩn đại học Y Dược luôn cao và biến động qua các năm
- Thí sinh gặp khó khăn trong việc lựa chọn ngành/trường phù hợp
- Cần công cụ dự báo khoa học để hỗ trợ ra quyết định

## 1.2 Mục tiêu
1. **Thu thập** dữ liệu điểm chuẩn Y Dược 2018-2025
2. **Phân tích** xu hướng biến động điểm chuẩn
3. **Xây dựng mô hình** dự báo điểm chuẩn 2026
4. **Phát triển ứng dụng web** giúp thí sinh tìm ngành phù hợp

## 1.3 Phạm vi
- **Đối tượng:** Các trường Đại học Y Dược tại Việt Nam
- **Thời gian:** Dữ liệu từ 2018 đến 2025
- **Tổ hợp môn:** 9 tổ hợp Y Dược phổ biến (A00, A01, A02, B00, B08, D01, D07, D08, D13)

---

# 2. THU THẬP DỮ LIỆU

## 2.1 Nguồn dữ liệu

### 2.1.1 Điểm chuẩn đại học
- **Nguồn:** VnExpress Education
- **Phương pháp:** Web Crawling với Python (Requests + BeautifulSoup)
- **File script:** `1_crawlVNExpressSchoolId.py`, `2_crawlDiemChuan.py`

### 2.1.2 Phân phối điểm thi THPT
- **Nguồn:** Dữ liệu điểm thi THPT Quốc gia 2018-2025
- **Thư mục:** `diem_thi_thptqg/`
- **Số lượng thí sinh:** ~6.5 triệu thí sinh (2018-2024), 232,089 thí sinh (2025)

## 2.2 Thống kê dữ liệu thu thập

| Thông tin | Giá trị |
|-----------|---------|
| **Tổng số dòng dữ liệu** | 1,704 |
| **Số trường Y Dược** | 17 |
| **Số ngành học** | 83 |
| **Khoảng thời gian** | 2018 - 2025 (8 năm) |
| **Số tổ hợp môn** | 29 (lọc còn 9 tổ hợp Y Dược) |

## 2.3 Danh sách 9 tổ hợp môn Y Dược

| Mã | Tổ hợp | Các môn |
|----|--------|---------|
| A00 | Toán - Lý - Hóa | toan, vat_ly, hoa_hoc |
| A01 | Toán - Lý - Anh | toan, vat_ly, ngoai_ngu |
| A02 | Toán - Lý - Sinh | toan, vat_ly, sinh_hoc |
| B00 | Toán - Hóa - Sinh | toan, hoa_hoc, sinh_hoc |
| B08 | Toán - Sinh - Anh | toan, sinh_hoc, ngoai_ngu |
| D01 | Toán - Văn - Anh | toan, ngu_van, ngoai_ngu |
| D07 | Toán - Hóa - Anh | toan, hoa_hoc, ngoai_ngu |
| D08 | Toán - Sinh - Anh | toan, sinh_hoc, ngoai_ngu |
| D13 | Toán - Văn - Sinh | toan, ngu_van, sinh_hoc |

---

# 3. XỬ LÝ VÀ LÀM SẠCH DỮ LIỆU

## 3.1 Quy trình xử lý
```
Dữ liệu thô → Làm sạch → Chuẩn hóa → Quy đổi Percentile → Training Data
```

## 3.2 File script: `3_cleanData.py`

### Các bước xử lý:
1. **Loại bỏ dữ liệu trùng lặp**
2. **Chuẩn hóa tên trường, tên ngành**
3. **Xử lý giá trị null/missing**
4. **Tách dòng có nhiều tổ hợp** (vd: "A00, B00" → 2 dòng riêng)
5. **Lọc ngành có dữ liệu 2025** (để dự báo 2026)

## 3.3 Chuyển đổi điểm → Percentile

### Tại sao cần chuyển đổi?
- Điểm chuẩn thay đổi do **độ khó đề thi** khác nhau mỗi năm
- Percentile phản ánh **vị trí tương đối** của thí sinh (Top X%)
- So sánh công bằng giữa các năm

### Công thức:
```
Percentile = (Rank / Total Students) × 100
```

### Ví dụ:
- Điểm 27.0 năm 2020 → Top 0.5% (rất cao vì đề khó)
- Điểm 27.0 năm 2023 → Top 1.2% (thấp hơn vì đề dễ hơn)

## 3.4 Kết quả sau xử lý

| Chỉ số | Trước | Sau |
|--------|-------|-----|
| Số dòng | 1,704 | 1,254 |
| Số ngành (có 2025) | - | 329 |
| Missing values | Có | 0 |

---

# 4. PHƯƠNG PHÁP & MÔ HÌNH

## 4.1 Tổng quan kiến trúc

```
┌─────────────────────────────────────────────────────────────┐
│                    ENSEMBLE LEARNING                         │
├─────────────┬─────────────┬─────────────┬─────────────────┤
│ Weighted    │ Exponential │   ARIMA     │    Linear       │
│ Average     │ Smoothing   │  (1,1,1)    │  Regression     │
│   (WA)      │   (ETS)     │             │     (LR)        │
├─────────────┴─────────────┴─────────────┴─────────────────┤
│           Auto Model Selection (based on CI)               │
├─────────────────────────────────────────────────────────────┤
│              Confidence Interval Calculation                │
└─────────────────────────────────────────────────────────────┘
```

## 4.2 Các mô hình sử dụng

### 4.2.1 Weighted Average (WA)
```python
weights = exp(linspace(0, α, n))  # α = 2.0
prediction = Σ(values × weights) / Σ(weights)
```
- **Ưu điểm:** Đơn giản, ổn định, năm gần nhất có trọng số cao hơn
- **Nhược điểm:** Không bắt được xu hướng phức tạp

### 4.2.2 Exponential Smoothing (ETS)
```python
model = ExponentialSmoothing(values, trend='add')
prediction = model.fit().forecast(1)
```
- **Ưu điểm:** Bắt được xu hướng tăng/giảm
- **Nhược điểm:** Cần ít nhất 4 điểm dữ liệu

### 4.2.3 ARIMA (1,1,1)
```python
model = ARIMA(values, order=(1, 1, 1))
prediction = model.fit().get_forecast(1)
```
- **Ưu điểm:** Mô hình hóa chuỗi thời gian phức tạp
- **Nhược điểm:** Cần ít nhất 5 điểm dữ liệu

### 4.2.4 Linear Regression (LR)
```python
model = LinearRegression()
model.fit(years, values)
prediction = model.predict(next_year)
```
- **Ưu điểm:** Ngoại suy xu hướng tuyến tính
- **Nhược điểm:** Không xử lý được biến động

## 4.3 Ensemble Method

### Weighted Ensemble (dựa trên Confidence Interval):
```python
for each model:
    weight = 1 / CI  # CI nhỏ = weight cao
    
ensemble_pred = Σ(pred × weight) / Σ(weight)
```

### Lý do:
- Model có CI nhỏ → dự đoán chính xác hơn → weight cao hơn
- Kết hợp nhiều model giảm rủi ro overfitting

## 4.4 Cross-Validation cho Time Series

### Expanding Window CV:
```
Year:  2018  2019  2020  2021  2022  2023  2024  2025
       ├──────────────────────┤ Train
                               ├────┤ Test (2023)
       ├─────────────────────────────┤ Train  
                                     ├────┤ Test (2024)
       ├────────────────────────────────────┤ Train
                                            ├────┤ Test (2025)
```

## 4.5 Confidence Interval

### Công thức:
```
CI = 1.96 × std(residuals)  # 95% confidence
```

### Ý nghĩa độ tin cậy:
| CI | Độ tin cậy | Ý nghĩa |
|----|------------|---------|
| < 5 | ⭐⭐⭐ Rất cao | Dữ liệu ổn định, dự báo chính xác |
| 5-10 | ⭐⭐ Trung bình | Biến động vừa phải |
| > 10 | ⭐ Thấp | Dữ liệu biến động lớn |

---

# 5. KẾT QUẢ TRAINING

## 5.1 File script: `4_train_advanced_win.py`

## 5.2 Thống kê kết quả

### Phân bố Model Selection:
| Model | Số lượng | Tỷ lệ |
|-------|----------|-------|
| **WA (Weighted Average)** | 214 | 65.0% |
| **LR (Linear Regression)** | 90 | 27.4% |
| **ETS (Exponential Smoothing)** | 25 | 7.6% |
| ARIMA | 0 | 0% |

### Cross-Validation Error (MAE):
| Model | Mean ± Std |
|-------|------------|
| **WA** | 11.64 ± 9.23 (tốt nhất) |
| ETS | 14.77 ± 14.04 |
| LR | 14.96 ± 14.28 |

## 5.3 Nhận xét

1. **WA là model tốt nhất** với MAE thấp nhất (11.64)
2. **ARIMA không được chọn** vì đa số ngành chỉ có 4-6 năm dữ liệu
3. **Mean CI = 11.03** percentile points

## 5.4 Output Files

| File | Mô tả |
|------|-------|
| `university_ranking_model_2026.pkl` | Model dự đoán (329 ngành) |
| `score_distribution_2025.pkl` | Bảng tra percentile 2025 |
| `model_analytics.pkl` | CI, model selection, features |
| `training_report.png` | Biểu đồ visualization |

---

# 6. WEB APPLICATION

## 6.1 File: `web_app.py`

## 6.2 Công nghệ sử dụng
- **Framework:** Streamlit
- **Backend:** Python (Pandas, NumPy, Joblib)
- **Visualization:** Plotly (interactive charts)

## 6.3 Tính năng chính

### 6.3.1 Phần 1: Tìm ngành phù hợp
- Nhập điểm 6 môn (Toán, Lý, Hóa, Sinh, Văn, Anh)
- Tự động tính điểm tất cả 9 tổ hợp
- Hiển thị danh sách ngành theo khả năng đậu

### 6.3.2 Phần 2: Phân tích tổ hợp cho ngành cụ thể
- Chọn trường + ngành muốn xét
- So sánh các tổ hợp có thể dùng
- Gợi ý tổ hợp tốt nhất

## 6.4 Các cột hiển thị

| Cột | Ý nghĩa |
|-----|---------|
| Điểm bạn | Tổng điểm 3 môn + ưu tiên |
| ĐC 2025 | Điểm chuẩn năm 2025 |
| Dư/Thiếu | Chênh lệch (✅ +2.0 hoặc ❌ -3.5) |
| ĐC 2024, 2023 | Điểm chuẩn các năm trước |
| Khả năng | 🟢 Cao, 🟡 Trung bình, 🔴 Thấp |
| Tin cậy | ⭐⭐⭐, ⭐⭐, ⭐ |

## 6.5 Ý nghĩa màu sắc

| Màu | Khả năng đậu | Khoảng cách percentile |
|-----|--------------|------------------------|
| 🟢 Rất cao | > 80% | > 10 |
| 🟢 Cao | 70-80% | 2-10 |
| 🟡 Trung bình | 50-70% | 0-2 |
| 🟠 Thấp | 30-50% | -2 to 0 |
| 🔴 Rất thấp | < 30% | < -2 |

## 6.6 Chạy ứng dụng
```bash
streamlit run web_app.py
```
Truy cập: http://localhost:8501

---

# 7. KẾT LUẬN

## 7.1 Kết quả đạt được

✅ **Thu thập thành công** 1,704 dòng dữ liệu điểm chuẩn 2018-2025

✅ **Xây dựng pipeline** xử lý dữ liệu hoàn chỉnh

✅ **Train 329 nhóm ngành** với 4 mô hình khác nhau

✅ **Ensemble learning** với auto model selection

✅ **Web app** trực quan, dễ sử dụng

## 7.2 Hạn chế

⚠️ Chỉ có dữ liệu 8 năm (2018-2025) - chuỗi thời gian ngắn

⚠️ Một số ngành có dữ liệu không đầy đủ

⚠️ Chưa tính đến các yếu tố bên ngoài (số lượng thí sinh, chính sách...)

## 7.3 Hướng phát triển

🔮 Mở rộng sang các khối ngành khác (Kinh tế, Kỹ thuật...)

🔮 Tích hợp dữ liệu bổ sung (chỉ tiêu tuyển sinh, số lượng đăng ký...)

🔮 Phát triển mobile app

🔮 Export báo cáo PDF cá nhân hóa

---

# 📁 CẤU TRÚC DỰ ÁN

```
CrawlDataDiemChuan/
│
├── 📊 DATA COLLECTION
│   ├── 1_crawlVNExpressSchoolId.py    # Crawl ID trường
│   └── 2_crawlDiemChuan.py            # Crawl điểm chuẩn
│
├── 🔧 DATA PROCESSING
│   ├── 3_cleanData.py                 # Làm sạch dữ liệu
│   └── diem_chuan_cleaned.csv         # Dữ liệu đã xử lý
│
├── 🤖 MODEL TRAINING
│   ├── 4_train_advanced_win.py        # Training script (advanced)
│   └── model_artifacts/               # Output models
│       ├── university_ranking_model_2026.pkl
│       ├── score_distribution_2025.pkl
│       ├── model_analytics.pkl
│       └── training_report.png
│
├── 💻 WEB APPLICATION
│   ├── web_app.py                     # Streamlit app
│   └── 5_recommend_combination.py     # CLI recommendation tool
│
├── 📈 VISUALIZATION
│   ├── visualizeData.py
│   ├── plot_output/
│   └── Bieu_Do_Output/
│
├── 📓 NOTEBOOKS
│   └── Advanced_University_Prediction.ipynb
│
└── 📄 DOCUMENTATION
    ├── PROJECT_SUMMARY.md             # File này
    └── requirements.txt               # Dependencies
```

---

# 📦 DEPENDENCIES

```txt
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.2.0
statsmodels>=0.14.0
streamlit>=1.28.0
joblib>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.15.0
requests>=2.31.0
beautifulsoup4>=4.12.0
```

---

# 🚀 HƯỚNG DẪN CHẠY

## Bước 1: Cài đặt dependencies
```bash
pip install -r requirements.txt
```

## Bước 2: Training model
```bash
python 4_train_advanced_win.py
```

## Bước 3: Chạy web app
```bash
streamlit run web_app.py
```

## Bước 4: Truy cập
Mở browser: http://localhost:8501

---

**© 2025 - Hệ thống Dự báo Điểm chuẩn Đại học Y Dược**
