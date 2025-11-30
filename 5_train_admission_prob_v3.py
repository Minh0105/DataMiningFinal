"""
ADMISSION PROBABILITY MODEL v3 - REALISTIC VERSION
Train model du doan xac suat dau KHONG CAN BIET DIEM CHUAN

Key changes:
- KHONG dung 'gap' (vi chua biet diem chuan 2026)
- Dung percentile va thong tin nganh/truong de du doan
- Model hoc pattern: "Nganh nay thuong lay Top X%"
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
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import warnings
warnings.filterwarnings("ignore")

print("="*60)
print("   ADMISSION PROBABILITY MODEL v3 - REALISTIC")
print("="*60)

# ================= CONFIG =================
SCORE_FOLDER = 'diem_thi_thptqg'
BENCHMARK_FILE = 'diem_chuan_cleaned.csv'
OUTPUT_DIR = 'model_artifacts'

BLOCK_MAP = {
    'A00': ['toan', 'vat_ly', 'hoa_hoc'],
    'B00': ['toan', 'hoa_hoc', 'sinh_hoc'],
    'D01': ['toan', 'ngu_van', 'ngoai_ngu'],
    'D07': ['toan', 'hoa_hoc', 'ngoai_ngu'],
}

# ================= STEP 1: LOAD DATA =================
print("\n[STEP 1] Loading data...")

df_benchmark = pd.read_csv(BENCHMARK_FILE)
print(f"   Diem chuan: {len(df_benchmark)} records")

# Load tat ca diem thi de xay dung percentile lookup
print("   Loading diem thi THPT (2020-2025)...")

all_score_data = {}  # {(year, block): scores array}
files = glob.glob(os.path.join(SCORE_FOLDER, '*.csv'))

for f in sorted(files):
    year = int(re.search(r'\d{4}', os.path.basename(f)).group())
    if year < 2020:
        continue
    
    print(f"   -> Nam {year}...")
    df_scores = pd.read_csv(f)
    
    for block, cols in BLOCK_MAP.items():
        if all(c in df_scores.columns for c in cols):
            temp = df_scores.dropna(subset=cols)
            scores = temp[cols].sum(axis=1).values
            all_score_data[(year, block)] = scores
            print(f"      {block}: {len(scores)} thi sinh")

# ================= STEP 2: TINH PERCENTILE CHO DIEM CHUAN =================
print("\n[STEP 2] Tinh percentile cho diem chuan...")

def get_percentile(score, year, block):
    """Tinh percentile cua diem trong phan phoi nam do"""
    key = (year, block)
    if key not in all_score_data:
        return None
    scores = all_score_data[key]
    return (scores <= score).sum() / len(scores) * 100

# Them cot percentile_required cho moi nganh
df_benchmark['percentile_required'] = df_benchmark.apply(
    lambda row: get_percentile(row['diem_chuan'], row['nam'], row['to_hop_mon']), 
    axis=1
)
df_benchmark = df_benchmark.dropna(subset=['percentile_required'])
print(f"   -> {len(df_benchmark)} records co percentile")

# ================= STEP 3: TAO TRAINING DATA =================
print("\n[STEP 3] Tao training data...")

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
    
    n_samples = min(sample_per_major, len(scores))
    sampled = np.random.choice(scores, size=n_samples, replace=False)
    
    for score in sampled:
        label = 1 if score >= diem_chuan else 0
        student_percentile = (scores <= score).sum() / len(scores) * 100
        
        training_data.append({
            'university_id': university_id,
            'ma_nganh': ma_nganh,
            'to_hop_mon': block,
            'nam': nam,
            'diem_thi': score,
            'student_percentile': student_percentile,
            'percentile_required': percentile_required,  # Percentile yeu cau cua nganh (tu lich su)
            'label': label
        })

df_train = pd.DataFrame(training_data)
print(f"   Created {len(df_train)} samples")
print(f"   Dau: {(df_train['label']==1).sum()} ({(df_train['label']==1).mean()*100:.1f}%)")
print(f"   Truot: {(df_train['label']==0).sum()} ({(df_train['label']==0).mean()*100:.1f}%)")

# ================= STEP 4: FEATURE ENGINEERING =================
print("\n[STEP 4] Feature Engineering...")

le_university = LabelEncoder()
le_nganh = LabelEncoder()
le_block = LabelEncoder()

df_train['uni_enc'] = le_university.fit_transform(df_train['university_id'].astype(str))
df_train['nganh_enc'] = le_nganh.fit_transform(df_train['ma_nganh'].astype(str))
df_train['block_enc'] = le_block.fit_transform(df_train['to_hop_mon'])

# QUAN TRONG: Khong dung 'diem_thi' truc tiep, chi dung percentile
# Vi the model se hoc: "Nganh X thuong yeu cau Top Y%"
feature_cols = [
    'student_percentile',    # Percentile cua thi sinh
    'percentile_required',   # Percentile yeu cau cua nganh (trung binh lich su)
    'uni_enc',               # Ma truong
    'nganh_enc',             # Ma nganh
    'block_enc',             # To hop
]

X = df_train[feature_cols].values
y = df_train['label'].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"   Features: {feature_cols}")
print(f"   X shape: {X_scaled.shape}")

# ================= STEP 5: TRAIN MODELS =================
print("\n[STEP 5] Training models...")

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)
print(f"   Train: {len(X_train)}, Test: {len(X_test)}")

# Train Gradient Boosting (tot hon cho bai toan nay)
print("\n   Training Gradient Boosting...")
gb_model = GradientBoostingClassifier(
    n_estimators=100, 
    max_depth=5, 
    learning_rate=0.1,
    random_state=42
)
gb_model.fit(X_train, y_train)

y_pred = gb_model.predict(X_test)
y_prob = gb_model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)

print(f"   Accuracy: {accuracy:.4f}")
print(f"   ROC-AUC: {roc_auc:.4f}")

# Cross validation
cv_scores = cross_val_score(gb_model, X_scaled, y, cv=5, scoring='roc_auc')
print(f"   CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")

# Feature importance
print("\n   Feature Importance:")
for i, col in enumerate(feature_cols):
    print(f"   - {col}: {gb_model.feature_importances_[i]:.4f}")

# ================= STEP 6: TINH PERCENTILE YEU CAU TRUNG BINH CHO MOI NGANH =================
print("\n[STEP 6] Tinh percentile trung binh cho moi nganh...")

# Tinh trung binh percentile yeu cau cho moi (university, nganh, block)
avg_percentile = df_benchmark.groupby(['university_id', 'ma_nganh', 'to_hop_mon']).agg({
    'percentile_required': 'mean',
    'diem_chuan': 'mean'
}).reset_index()
avg_percentile.columns = ['university_id', 'ma_nganh', 'to_hop_mon', 'avg_percentile_required', 'avg_diem_chuan']

print(f"   -> {len(avg_percentile)} nganh")

# ================= STEP 7: SAVE MODELS =================
print("\n[STEP 7] Saving models...")

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Save model
joblib.dump(gb_model, os.path.join(OUTPUT_DIR, 'admission_probability_model.pkl'))
print(f"   Saved: admission_probability_model.pkl")

# Save encoders
encoders = {
    'university': le_university,
    'nganh': le_nganh,
    'block': le_block,
    'feature_cols': feature_cols
}
joblib.dump(encoders, os.path.join(OUTPUT_DIR, 'admission_encoders.pkl'))
print(f"   Saved: admission_encoders.pkl")

# Save scaler
joblib.dump(scaler, os.path.join(OUTPUT_DIR, 'admission_scaler.pkl'))
print(f"   Saved: admission_scaler.pkl")

# Save avg_percentile lookup
joblib.dump(avg_percentile, os.path.join(OUTPUT_DIR, 'major_percentile_lookup.pkl'))
print(f"   Saved: major_percentile_lookup.pkl")

# Save score data 2025 de dung cho web app
joblib.dump({k: v for k, v in all_score_data.items() if k[0] == 2025}, 
            os.path.join(OUTPUT_DIR, 'score_data_2025.pkl'))
print(f"   Saved: score_data_2025.pkl")

# ================= STEP 8: DEMO =================
print("\n" + "="*60)
print("   DEMO: DU DOAN XAC SUAT DAU 2026")
print("="*60)

def predict_admission_probability(diem, block, university_id, ma_nganh):
    """Du doan xac suat dau cho nam 2026"""
    
    # 1. Tinh percentile cua thi sinh (dung phan phoi 2025)
    key = (2025, block)
    if key not in all_score_data:
        return None, None
    
    scores = all_score_data[key]
    student_percentile = (scores <= diem).sum() / len(scores) * 100
    
    # 2. Lay percentile yeu cau trung binh cua nganh
    row = avg_percentile[
        (avg_percentile['university_id'] == university_id) & 
        (avg_percentile['ma_nganh'] == ma_nganh) &
        (avg_percentile['to_hop_mon'] == block)
    ]
    
    if row.empty:
        return None, None
    
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
    
    # 4. Predict
    features = np.array([[student_percentile, percentile_required, uni_enc, nganh_enc, block_enc]])
    features_scaled = scaler.transform(features)
    prob = gb_model.predict_proba(features_scaled)[0][1]
    
    return prob, student_percentile, percentile_required

# Test cases
test_cases = [
    (27.0, 'B00', 215, '7720101', 'Y khoa Hue - Diem cao (27.0)'),
    (25.0, 'B00', 215, '7720101', 'Y khoa Hue - Diem TB (25.0)'),
    (23.0, 'B00', 215, '7720101', 'Y khoa Hue - Diem thap (23.0)'),
    (26.0, 'B00', 215, '7720201', 'Rang Ham Mat Hue (26.0)'),
    (24.0, 'B00', 215, '7720301', 'Duoc hoc Hue (24.0)'),
]

for diem, block, uni, nganh, desc in test_cases:
    result = predict_admission_probability(diem, block, uni, nganh)
    if result[0] is not None:
        prob, student_pct, required_pct = result
        print(f"\n{desc}")
        print(f"   To hop: {block} | Diem: {diem}")
        print(f"   Ban: Top {100-student_pct:.1f}% | Yeu cau: Top {100-required_pct:.1f}%")
        if student_pct >= required_pct:
            print(f"   >>> XAC SUAT DAU: {prob*100:.1f}% (Du diem)")
        else:
            print(f"   >>> XAC SUAT DAU: {prob*100:.1f}% (Chua du)")
    else:
        print(f"\n{desc}: Khong co du lieu")

print("\n" + "="*60)
print("   HOAN THANH!")
print("="*60)
