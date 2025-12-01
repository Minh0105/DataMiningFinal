"""
ADMISSION PROBABILITY MODEL v4 - ADVANCED VERSION (11 FEATURES)
Train model dự đoán xác suất đậu với các features nâng cao

Features (11 total):
1. student_percentile    - Vị trí thí sinh (Top X%)
2. percentile_required   - Ngành yêu cầu Top bao nhiêu %
3. uni_enc               - Encoded trường
4. nganh_enc             - Encoded ngành
5. block_enc             - Encoded tổ hợp
6. gap                   - Khoảng cách (student - required)
7. relative_position     - Tỷ lệ vị trí (student / required)
8. trend                 - Xu hướng điểm chuẩn qua các năm
9. volatility            - Độ biến động điểm chuẩn
10. school_prestige      - Độ khó trung bình của trường
11. block_competition    - Số thí sinh trong tổ hợp
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
import os
import glob
import joblib
import re
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import warnings
warnings.filterwarnings("ignore")

print("="*70)
print("   ADMISSION PROBABILITY MODEL v4 - ADVANCED (11 FEATURES)")
print("="*70)

# ================= CONFIG =================
SCORE_FOLDER = 'diem_thi_thptqg'
BENCHMARK_FILE = 'diem_chuan_cleaned.csv'
OUTPUT_DIR = 'model_artifacts'

BLOCK_MAP = {
    'A00': ['toan', 'vat_ly', 'hoa_hoc'],
    'A01': ['toan', 'vat_ly', 'ngoai_ngu'],
    'A02': ['toan', 'vat_ly', 'sinh_hoc'],
    'B00': ['toan', 'hoa_hoc', 'sinh_hoc'],
    'B08': ['toan', 'sinh_hoc', 'ngoai_ngu'],
    'D01': ['toan', 'ngu_van', 'ngoai_ngu'],
    'D07': ['toan', 'hoa_hoc', 'ngoai_ngu'],
    'D08': ['toan', 'sinh_hoc', 'ngoai_ngu'],
    'D13': ['toan', 'ngu_van', 'sinh_hoc'],
}

# ================= STEP 1: LOAD DATA =================
print("\n[STEP 1] Loading data...")

df_benchmark = pd.read_csv(BENCHMARK_FILE)
print(f"   📊 Điểm chuẩn: {len(df_benchmark):,} records")

# Load tất cả điểm thi để xây dựng percentile lookup
print("   📁 Loading điểm thi THPT (2018-2025)...")

all_score_data = {}  # {(year, block): scores array}
files = glob.glob(os.path.join(SCORE_FOLDER, '*.csv'))

for f in sorted(files):
    year_match = re.search(r'\d{4}', os.path.basename(f))
    if not year_match:
        continue
    year = int(year_match.group())
    
    print(f"      Năm {year}...", end=" ")
    df_scores = pd.read_csv(f)
    
    blocks_loaded = []
    for block, cols in BLOCK_MAP.items():
        if all(c in df_scores.columns for c in cols):
            temp = df_scores.dropna(subset=cols)
            scores = temp[cols].sum(axis=1).values
            all_score_data[(year, block)] = scores
            blocks_loaded.append(f"{block}({len(scores):,})")
    
    print(", ".join(blocks_loaded))

print(f"   ✅ Loaded {len(all_score_data)} (year, block) combinations")

# ================= STEP 2: TÍNH PERCENTILE CHO ĐIỂM CHUẨN =================
print("\n[STEP 2] Tính percentile cho điểm chuẩn...")

def get_percentile(score, year, block):
    """Tính percentile của điểm trong phân phối năm đó"""
    key = (year, block)
    if key not in all_score_data:
        return None
    scores = all_score_data[key]
    return (scores <= score).sum() / len(scores) * 100

# Thêm cột percentile_required cho mỗi ngành
df_benchmark['percentile_required'] = df_benchmark.apply(
    lambda row: get_percentile(row['diem_chuan'], row['nam'], row['to_hop_mon']), 
    axis=1
)
df_benchmark = df_benchmark.dropna(subset=['percentile_required'])
print(f"   ✅ {len(df_benchmark):,} records có percentile")

# ================= STEP 2.5: TÍNH ADVANCED FEATURES =================
print("\n[STEP 2.5] Tính toán Advanced Features...")

# a) School Prestige: Độ khó trung bình của trường (dựa trên điểm chuẩn trung bình)
school_prestige = df_benchmark.groupby('university_id')['diem_chuan'].mean().to_dict()
print(f"   📊 School Prestige: {len(school_prestige)} trường")

# b) Block Competition: Số lượng thí sinh trong mỗi tổ hợp (độ cạnh tranh)
block_competition = {}
for key, scores in all_score_data.items():
    nam, block = key
    if block not in block_competition:
        block_competition[block] = 0
    block_competition[block] += len(scores)
print(f"   📊 Block Competition: {len(block_competition)} tổ hợp")
for block, count in sorted(block_competition.items(), key=lambda x: -x[1]):
    print(f"      {block}: {count:,} thí sinh")

# c) Trend & Volatility: Xu hướng và biến động điểm chuẩn qua các năm
trend_data = {}
volatility_data = {}
for (uni, nganh, block), grp in df_benchmark.groupby(['university_id', 'ma_nganh', 'to_hop_mon']):
    sorted_grp = grp.sort_values('nam')
    dc_values = sorted_grp['diem_chuan'].values
    
    # Trend: hệ số góc của đường xu hướng (positive = tăng, negative = giảm)
    if len(dc_values) >= 2:
        years = np.arange(len(dc_values))
        slope, _ = np.polyfit(years, dc_values, 1)
        trend_data[(uni, nganh, block)] = slope
        volatility_data[(uni, nganh, block)] = np.std(dc_values)
    else:
        trend_data[(uni, nganh, block)] = 0
        volatility_data[(uni, nganh, block)] = 0

print(f"   📊 Trend & Volatility: {len(trend_data)} nhóm (uni, nganh, block)")

# Thống kê trend
trends = list(trend_data.values())
print(f"      Trend range: {min(trends):.2f} đến {max(trends):.2f} (điểm/năm)")
print(f"      Trend mean: {np.mean(trends):.2f}")

# ================= STEP 3: TẠO TRAINING DATA =================
print("\n[STEP 3] Tạo training data với 11 FEATURES...")

training_data = []
sample_per_major = 300

for _, row in df_benchmark.iterrows():
    block = str(row['to_hop_mon']).strip()
    nam = row['nam']
    key = (nam, block)
    
    if key not in all_score_data:
        continue
    
    scores = all_score_data[key]
    diem_chuan = row['diem_chuan']
    percentile_required = row['percentile_required']
    university_id = row['university_id']
    ma_nganh = row['ma_nganh']
    
    # Lấy các features bổ sung
    group_key = (university_id, ma_nganh, block)
    trend = trend_data.get(group_key, 0)
    volatility = volatility_data.get(group_key, 0)
    prestige = school_prestige.get(university_id, 25)  # default = 25 điểm
    competition = block_competition.get(block, 10000)   # default = 10000
    
    n_samples = min(sample_per_major, len(scores))
    sampled = np.random.choice(scores, size=n_samples, replace=False)
    
    for score in sampled:
        # Label: đậu nếu điểm >= điểm chuẩn
        label = 1 if score >= diem_chuan else 0
        student_percentile = (scores <= score).sum() / len(scores) * 100
        
        # === ADVANCED FEATURES ===
        gap = student_percentile - percentile_required
        relative_position = student_percentile / (percentile_required + 0.01)  # +0.01 tránh chia 0
        
        training_data.append({
            'university_id': university_id,
            'ma_nganh': ma_nganh,
            'to_hop_mon': block,
            'nam': nam,
            'diem_thi': score,
            # === Core Features ===
            'student_percentile': student_percentile,
            'percentile_required': percentile_required,
            # === NEW Advanced Features ===
            'gap': gap,
            'relative_position': relative_position,
            'trend': trend,
            'volatility': volatility,
            'school_prestige': prestige,
            'block_competition': competition,
            'label': label
        })

df_train = pd.DataFrame(training_data)
print(f"   ✅ Tạo được {len(df_train):,} training samples")
print(f"      Đậu: {(df_train['label']==1).sum():,} ({(df_train['label']==1).mean()*100:.1f}%)")
print(f"      Trượt: {(df_train['label']==0).sum():,} ({(df_train['label']==0).mean()*100:.1f}%)")

# ================= STEP 4: FEATURE ENGINEERING =================
print("\n[STEP 4] Feature Engineering (11 Features)...")

le_university = LabelEncoder()
le_nganh = LabelEncoder()
le_block = LabelEncoder()

df_train['uni_enc'] = le_university.fit_transform(df_train['university_id'].astype(str))
df_train['nganh_enc'] = le_nganh.fit_transform(df_train['ma_nganh'].astype(str))
df_train['block_enc'] = le_block.fit_transform(df_train['to_hop_mon'])

# 11 FEATURES - đầy đủ thông tin để dự đoán xác suất đậu
feature_cols = [
    # === Core Features (5) ===
    'student_percentile',    # Vị trí của thí sinh (Top bao nhiêu %)
    'percentile_required',   # Ngành yêu cầu Top bao nhiêu % (từ lịch sử)
    'uni_enc',               # Encoded trường
    'nganh_enc',             # Encoded ngành
    'block_enc',             # Encoded tổ hợp
    # === NEW Advanced Features (6) ===
    'gap',                   # Khoảng cách percentile (student - required)
    'relative_position',     # Tỷ lệ vị trí (student / required)
    'trend',                 # Xu hướng điểm chuẩn qua các năm
    'volatility',            # Độ biến động điểm chuẩn
    'school_prestige',       # Độ khó trung bình của trường
    'block_competition',     # Số thí sinh trong tổ hợp
]

X = df_train[feature_cols].values
y = df_train['label'].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"   ✅ Features ({len(feature_cols)} total):")
for i, col in enumerate(feature_cols):
    print(f"      {i+1}. {col}")
print(f"   ✅ X shape: {X_scaled.shape}")

# ================= STEP 5: TRAIN MODELS =================
print("\n[STEP 5] Training Gradient Boosting...")

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)
print(f"   Train: {len(X_train):,}, Test: {len(X_test):,}")

# Train Gradient Boosting
gb_model = GradientBoostingClassifier(
    n_estimators=150,    # Tăng từ 100 lên 150
    max_depth=6,         # Tăng từ 5 lên 6 (vì có nhiều features hơn)
    learning_rate=0.1,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)
gb_model.fit(X_train, y_train)

y_pred = gb_model.predict(X_test)
y_prob = gb_model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)

print(f"\n   📊 RESULTS:")
print(f"      Accuracy: {accuracy:.4f}")
print(f"      ROC-AUC: {roc_auc:.4f}")

# Cross validation
cv_scores = cross_val_score(gb_model, X_scaled, y, cv=5, scoring='roc_auc')
print(f"      CV Score: {cv_scores.mean():.4f} (±{cv_scores.std()*2:.4f})")

# Feature importance
print(f"\n   📈 Feature Importance:")
importance_df = pd.DataFrame({
    'feature': feature_cols,
    'importance': gb_model.feature_importances_
}).sort_values('importance', ascending=False)

for _, row in importance_df.iterrows():
    bar = '█' * int(row['importance'] * 50)
    print(f"      {row['feature']:20s}: {row['importance']:.4f} {bar}")

# ================= STEP 6: TÍNH PERCENTILE YÊU CẦU TRUNG BÌNH =================
print("\n[STEP 6] Tính percentile trung bình cho mỗi ngành...")

avg_percentile = df_benchmark.groupby(['university_id', 'ma_nganh', 'to_hop_mon']).agg({
    'percentile_required': 'mean',
    'diem_chuan': 'mean'
}).reset_index()
avg_percentile.columns = ['university_id', 'ma_nganh', 'to_hop_mon', 'avg_percentile_required', 'avg_diem_chuan']

print(f"   ✅ {len(avg_percentile)} ngành")

# ================= STEP 7: SAVE MODELS =================
print("\n[STEP 7] Saving models...")

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Save model
joblib.dump(gb_model, os.path.join(OUTPUT_DIR, 'admission_probability_model.pkl'))
print(f"   ✅ Saved: admission_probability_model.pkl")

# Save encoders với feature_cols
encoders = {
    'university': le_university,
    'nganh': le_nganh,
    'block': le_block,
    'feature_cols': feature_cols  # QUAN TRỌNG: để web_app biết dùng bao nhiêu features
}
joblib.dump(encoders, os.path.join(OUTPUT_DIR, 'admission_encoders.pkl'))
print(f"   ✅ Saved: admission_encoders.pkl")

# Save scaler
joblib.dump(scaler, os.path.join(OUTPUT_DIR, 'admission_scaler.pkl'))
print(f"   ✅ Saved: admission_scaler.pkl")

# Save avg_percentile lookup
joblib.dump(avg_percentile, os.path.join(OUTPUT_DIR, 'major_percentile_lookup.pkl'))
print(f"   ✅ Saved: major_percentile_lookup.pkl")

# Save score data 2025 để dùng cho web app
score_data_2025 = {k: v for k, v in all_score_data.items() if k[0] == 2025}
joblib.dump(score_data_2025, os.path.join(OUTPUT_DIR, 'score_data_2025.pkl'))
print(f"   ✅ Saved: score_data_2025.pkl")

# *** MỚI: Save advanced features metadata ***
advanced_features = {
    'school_prestige': school_prestige,      # dict: university_id -> avg_diem_chuan
    'block_competition': block_competition,  # dict: block -> num_students
    'trend': trend_data,                     # dict: (uni, nganh, block) -> slope
    'volatility': volatility_data            # dict: (uni, nganh, block) -> std
}
joblib.dump(advanced_features, os.path.join(OUTPUT_DIR, 'advanced_features.pkl'))
print(f"   ✅ Saved: advanced_features.pkl (NEW - trend, volatility, prestige, competition)")

# ================= STEP 8: DEMO =================
print("\n" + "="*70)
print("   DEMO: DỰ ĐOÁN XÁC SUẤT ĐẬU 2026 (11 Features)")
print("="*70)

def predict_admission_probability_v4(diem, block, university_id, ma_nganh):
    """Dự đoán xác suất đậu cho năm 2026 với 11 features"""
    
    # 1. Tính percentile của thí sinh (dùng phân phối 2025)
    key = (2025, block)
    if key not in all_score_data:
        return None, None, None
    
    scores = all_score_data[key]
    student_percentile = (scores <= diem).sum() / len(scores) * 100
    
    # 2. Lấy percentile yêu cầu trung bình của ngành
    row = avg_percentile[
        (avg_percentile['university_id'] == university_id) & 
        (avg_percentile['ma_nganh'] == ma_nganh) &
        (avg_percentile['to_hop_mon'] == block)
    ]
    
    if row.empty:
        return None, None, None
    
    percentile_required = row.iloc[0]['avg_percentile_required']
    
    # 3. Encode features
    try:
        uni_enc = le_university.transform([str(university_id)])[0]
    except:
        uni_enc = 0
    try:
        nganh_enc = le_nganh.transform([str(ma_nganh)])[0]
    except:
        nganh_enc = 0
    try:
        block_enc = le_block.transform([block])[0]
    except:
        block_enc = 0
    
    # 4. Tính advanced features
    gap = student_percentile - percentile_required
    relative_position = student_percentile / (percentile_required + 0.01)
    
    group_key = (university_id, ma_nganh, block)
    trend = trend_data.get(group_key, 0)
    volatility = volatility_data.get(group_key, 0)
    prestige = school_prestige.get(university_id, 25)
    competition = block_competition.get(block, 10000)
    
    # 5. Predict với 11 features
    features = np.array([[
        student_percentile, percentile_required, uni_enc, nganh_enc, block_enc,
        gap, relative_position, trend, volatility, prestige, competition
    ]])
    features_scaled = scaler.transform(features)
    prob = gb_model.predict_proba(features_scaled)[0][1]
    
    return prob, student_percentile, percentile_required

# Test cases
test_cases = [
    (27.5, 'B00', 215, '7720101', 'Y khoa Huế - Điểm rất cao (27.5)'),
    (26.0, 'B00', 215, '7720101', 'Y khoa Huế - Điểm cao (26.0)'),
    (24.5, 'B00', 215, '7720101', 'Y khoa Huế - Điểm TB (24.5)'),
    (23.0, 'B00', 215, '7720101', 'Y khoa Huế - Điểm thấp (23.0)'),
    (26.0, 'B00', 215, '7720201', 'Răng Hàm Mặt Huế (26.0)'),
    (24.0, 'B00', 215, '7720301', 'Dược học Huế (24.0)'),
]

for diem, block, uni, nganh, desc in test_cases:
    result = predict_admission_probability_v4(diem, block, uni, nganh)
    if result[0] is not None:
        prob, student_pct, required_pct = result
        print(f"\n🎓 {desc}")
        print(f"   Tổ hợp: {block} | Điểm: {diem}")
        print(f"   Bạn: Top {100-student_pct:.2f}% | Yêu cầu: Top {100-required_pct:.2f}%")
        
        # Thêm thông tin về các features bổ sung
        group_key = (uni, nganh, block)
        trend_val = trend_data.get(group_key, 0)
        vol_val = volatility_data.get(group_key, 0)
        
        trend_emoji = "📈" if trend_val > 0 else "📉" if trend_val < 0 else "➡️"
        print(f"   Trend: {trend_emoji} {trend_val:+.2f} điểm/năm | Biến động: {vol_val:.2f}")
        
        if prob >= 0.7:
            print(f"   >>> 🟢 XÁC SUẤT ĐẬU: {prob*100:.1f}% (Cao)")
        elif prob >= 0.4:
            print(f"   >>> 🟡 XÁC SUẤT ĐẬU: {prob*100:.1f}% (Trung bình)")
        else:
            print(f"   >>> 🔴 XÁC SUẤT ĐẬU: {prob*100:.1f}% (Thấp)")
    else:
        print(f"\n❌ {desc}: Không có dữ liệu")

print("\n" + "="*70)
print("   ✅ HOÀN THÀNH - Model v4 với 11 Features!")
print("="*70)
print("\n📁 Files đã lưu:")
print("   - admission_probability_model.pkl (Gradient Boosting)")
print("   - admission_encoders.pkl (LabelEncoders + feature_cols)")
print("   - admission_scaler.pkl (StandardScaler)")
print("   - major_percentile_lookup.pkl (Lookup table)")
print("   - score_data_2025.pkl (Phân phối điểm 2025)")
print("   - advanced_features.pkl (trend, volatility, prestige, competition)")
