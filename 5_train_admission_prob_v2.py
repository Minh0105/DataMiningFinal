"""
ADMISSION PROBABILITY MODEL v2
Train model ML de du doan xac suat dau dai hoc
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
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import warnings
warnings.filterwarnings("ignore")

print("="*60)
print("   ADMISSION PROBABILITY MODEL TRAINING")
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

# Load benchmark
df_benchmark = pd.read_csv(BENCHMARK_FILE)
df_benchmark = df_benchmark[df_benchmark['nam'] >= 2020]
print(f"   Diem chuan: {len(df_benchmark)} records (2020-2025)")

# Load diem thi - chi load 1 nam de nhanh
print("   Loading diem thi 2025...")
df_scores_2025 = pd.read_csv(os.path.join(SCORE_FOLDER, 'diem_thi_thpt_2025.csv'))

# Tinh diem theo tung to hop
score_data = {}
for block, cols in BLOCK_MAP.items():
    if all(c in df_scores_2025.columns for c in cols):
        temp = df_scores_2025.dropna(subset=cols)
        score_data[block] = temp[cols].sum(axis=1).values
        print(f"   {block}: {len(score_data[block])} thi sinh")

# ================= STEP 2: CREATE TRAINING DATA VỚI STRATIFIED SAMPLING =================
print("\n[STEP 2] Creating training data voi Stratified Sampling...")

def stratified_sample(scores, diem_chuan, total_samples=200):
    """
    Stratified Sampling: Lay mau thong minh theo 4 zones
    - Zone 1 (rot chac): gap < -5     -> 15% samples
    - Zone 2 (co the rot): -5 <= gap < -1  -> 25% samples
    - Zone 3 (ranh gioi): -1 <= gap < +3   -> 40% samples
    - Zone 4 (dau chac): gap >= +3    -> 20% samples
    """
    gaps = scores - diem_chuan
    
    zone1_mask = gaps < -5
    zone2_mask = (gaps >= -5) & (gaps < -1)
    zone3_mask = (gaps >= -1) & (gaps < 3)
    zone4_mask = gaps >= 3
    
    zone_scores = [
        scores[zone1_mask],
        scores[zone2_mask],
        scores[zone3_mask],
        scores[zone4_mask]
    ]
    
    zone_ratios = [0.15, 0.25, 0.40, 0.20]
    
    sampled_all = []
    for zone_data, ratio in zip(zone_scores, zone_ratios):
        n_want = int(total_samples * ratio)
        if len(zone_data) > 0:
            n_take = min(n_want, len(zone_data))
            sampled = np.random.choice(zone_data, size=n_take, replace=False)
            sampled_all.extend(sampled)
    
    return np.array(sampled_all)

training_data = []
sample_per_major = 200  # Giam de nhanh hon

for _, row in df_benchmark.iterrows():
    block = str(row['to_hop_mon']).strip()
    if block not in score_data:
        continue
    
    diem_chuan = row['diem_chuan']
    university_id = row['university_id']
    ma_nganh = row['ma_nganh']
    nam = row['nam']
    
    scores = score_data[block]
    # STRATIFIED SAMPLING thay vi random
    sampled = stratified_sample(scores, diem_chuan, sample_per_major)
    
    for score in sampled:
        label = 1 if score >= diem_chuan else 0
        gap = score - diem_chuan
        percentile = (scores <= score).sum() / len(scores) * 100
        
        training_data.append({
            'university_id': university_id,
            'ma_nganh': ma_nganh,
            'to_hop_mon': block,
            'nam': nam,
            'diem_thi': score,
            'diem_chuan': diem_chuan,
            'gap': gap,
            'percentile': percentile,
            'label': label
        })

df_train = pd.DataFrame(training_data)
print(f"   Created {len(df_train)} samples")
print(f"   Dau: {(df_train['label']==1).sum()} ({(df_train['label']==1).mean()*100:.1f}%)")
print(f"   Truot: {(df_train['label']==0).sum()} ({(df_train['label']==0).mean()*100:.1f}%)")

# ================= STEP 3: FEATURE ENGINEERING =================
print("\n[STEP 3] Feature Engineering...")

le_university = LabelEncoder()
le_nganh = LabelEncoder()
le_block = LabelEncoder()

df_train['uni_enc'] = le_university.fit_transform(df_train['university_id'].astype(str))
df_train['nganh_enc'] = le_nganh.fit_transform(df_train['ma_nganh'].astype(str))
df_train['block_enc'] = le_block.fit_transform(df_train['to_hop_mon'])

feature_cols = ['diem_thi', 'gap', 'percentile', 'uni_enc', 'nganh_enc', 'block_enc', 'nam']
X = df_train[feature_cols].values
y = df_train['label'].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"   Features: {feature_cols}")
print(f"   X shape: {X_scaled.shape}")

# ================= STEP 4: TRAIN MODELS =================
print("\n[STEP 4] Training models...")

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)
print(f"   Train: {len(X_train)}, Test: {len(X_test)}")

# Train Random Forest
print("\n   Training Random Forest...")
rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_test)
y_prob = rf_model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)

print(f"   Accuracy: {accuracy:.4f}")
print(f"   ROC-AUC: {roc_auc:.4f}")

# Cross validation
cv_scores = cross_val_score(rf_model, X_scaled, y, cv=5, scoring='roc_auc')
print(f"   CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")

# ================= STEP 5: SAVE MODELS =================
print("\n[STEP 5] Saving models...")

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Save model
joblib.dump(rf_model, os.path.join(OUTPUT_DIR, 'admission_probability_model.pkl'))
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

# ================= STEP 6: DEMO =================
print("\n" + "="*60)
print("   DEMO: DU DOAN XAC SUAT DAU")
print("="*60)

def predict_probability(diem, block, university_id, ma_nganh, nam=2025):
    """Du doan xac suat dau"""
    try:
        # Encode
        if str(university_id) in le_university.classes_:
            uni_enc = le_university.transform([str(university_id)])[0]
        else:
            uni_enc = 0
            
        if str(ma_nganh) in le_nganh.classes_:
            nganh_enc = le_nganh.transform([str(ma_nganh)])[0]
        else:
            nganh_enc = 0
            
        if block in le_block.classes_:
            block_enc = le_block.transform([block])[0]
        else:
            block_enc = 0
        
        # Tinh gap va percentile (uoc luong)
        if block in score_data:
            scores = score_data[block]
            percentile = (scores <= diem).sum() / len(scores) * 100
        else:
            percentile = 50
        
        # Lay diem chuan gan nhat
        dc_row = df_benchmark[
            (df_benchmark['university_id'] == university_id) & 
            (df_benchmark['ma_nganh'] == ma_nganh) &
            (df_benchmark['to_hop_mon'] == block)
        ]
        if not dc_row.empty:
            diem_chuan = dc_row.iloc[-1]['diem_chuan']
            gap = diem - diem_chuan
        else:
            gap = 0
            diem_chuan = None
        
        # Features
        features = np.array([[diem, gap, percentile, uni_enc, nganh_enc, block_enc, nam]])
        features_scaled = scaler.transform(features)
        
        # Predict
        prob = rf_model.predict_proba(features_scaled)[0][1]
        
        return prob, diem_chuan, gap
    except Exception as e:
        return None, None, None

# Test cases
test_cases = [
    (27.0, 'B00', 215, '7720101', 'Y khoa - DH Y Duoc Hue - Diem cao'),
    (24.5, 'B00', 215, '7720101', 'Y khoa - DH Y Duoc Hue - Diem TB'),
    (22.0, 'B00', 215, '7720101', 'Y khoa - DH Y Duoc Hue - Diem thap'),
    (25.0, 'B00', 215, '7720201', 'Rang Ham Mat - DH Y Duoc Hue'),
]

for diem, block, uni, nganh, desc in test_cases:
    prob, dc, gap = predict_probability(diem, block, uni, nganh)
    if prob is not None:
        print(f"\n{desc}")
        print(f"   Diem: {diem} | To hop: {block}")
        if dc:
            print(f"   Diem chuan 2025: {dc} | Gap: {gap:+.1f}")
        print(f"   >>> XAC SUAT DAU: {prob*100:.1f}%")
    else:
        print(f"\n{desc}: Khong the du doan")

print("\n" + "="*60)
print("   HOAN THANH!")
print("="*60)
