"""
ADMISSION PROBABILITY MODEL
===============================
Train model ML de du doan XAC SUAT DAU dai hoc
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

print("--- ADMISSION PROBABILITY MODEL - KHOI DONG ---")

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
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import warnings

warnings.filterwarnings("ignore")

# ================= CẤU HÌNH =================
SCORE_FOLDER = r'diem_thi_thptqg'
BENCHMARK_FILE = 'diem_chuan_cleaned.csv'
OUTPUT_DIR = 'model_artifacts'

# Output files
PROB_MODEL_OUTPUT = os.path.join(OUTPUT_DIR, 'admission_probability_model.pkl')
ENCODERS_OUTPUT = os.path.join(OUTPUT_DIR, 'admission_encoders.pkl')
SCALER_OUTPUT = os.path.join(OUTPUT_DIR, 'admission_scaler.pkl')

# Tổ hợp môn Y Dược
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

# ================= 1. TẠO TRAINING DATA =================
def create_training_data(score_folder, benchmark_file, sample_per_major=500):
    """
    Tạo training data bằng cách:
    1. Lấy điểm chuẩn từng ngành/năm
    2. Sample điểm thi thật từ THPT
    3. Label: 1 nếu điểm >= điểm chuẩn, 0 nếu không
    """
    print("\n🚀 BƯỚC 1: Tạo Training Data...")
    
    # Load điểm chuẩn
    df_benchmark = pd.read_csv(benchmark_file)
    print(f"   -> Loaded {len(df_benchmark)} dòng điểm chuẩn")
    
    # Chỉ lấy năm 2020-2025 để có dữ liệu gần đây
    df_benchmark = df_benchmark[df_benchmark['nam'] >= 2020]
    
    # Load tất cả điểm thi THPT
    all_scores = {}
    files = glob.glob(os.path.join(score_folder, "*.csv"))
    
    for file_path in files:
        filename = os.path.basename(file_path)
        try:
            year = int(re.search(r'\d{4}', filename).group())
        except:
            continue
        
        if year < 2020:
            continue
            
        print(f"   -> Đang load điểm thi năm {year}...")
        
        all_cols = list(set([col for cols in BLOCK_MAP.values() for col in cols]))
        
        try:
            df_scores = pd.read_csv(file_path, usecols=all_cols + ['sbd'] if 'sbd' in pd.read_csv(file_path, nrows=1).columns else all_cols)
        except:
            continue
        
        # Tính điểm cho mỗi tổ hợp
        for block, cols in BLOCK_MAP.items():
            if not all(col in df_scores.columns for col in cols):
                continue
            
            temp = df_scores.dropna(subset=cols).copy()
            if temp.empty:
                continue
                
            temp['block_score'] = temp[cols].sum(axis=1)
            
            key = (year, block)
            if key not in all_scores:
                all_scores[key] = temp['block_score'].values
            else:
                all_scores[key] = np.concatenate([all_scores[key], temp['block_score'].values])
    
    print(f"   -> Loaded điểm thi cho {len(all_scores)} cặp (năm, tổ hợp)")
    
    # Tạo training samples
    training_data = []
    
    for _, row in df_benchmark.iterrows():
        year = row['nam']
        block = str(row['to_hop_mon']).strip()
        diem_chuan = row['diem_chuan']
        university_id = row['university_id']
        ma_nganh = row['ma_nganh']
        
        if block not in BLOCK_MAP:
            continue
            
        key = (year, block)
        if key not in all_scores:
            continue
        
        scores = all_scores[key]
        
        # Sample điểm ngẫu nhiên
        n_samples = min(sample_per_major, len(scores))
        sampled_scores = np.random.choice(scores, size=n_samples, replace=False)
        
        for score in sampled_scores:
            # Label: 1 = Đậu, 0 = Trượt
            label = 1 if score >= diem_chuan else 0
            
            # Tính khoảng cách so với điểm chuẩn
            gap = score - diem_chuan
            
            # Tính percentile của điểm trong năm đó
            percentile = (scores <= score).sum() / len(scores) * 100
            
            training_data.append({
                'university_id': university_id,
                'ma_nganh': ma_nganh,
                'to_hop_mon': block,
                'nam': year,
                'diem_thi': score,
                'diem_chuan': diem_chuan,
                'gap': gap,
                'percentile': percentile,
                'label': label
            })
    
    df_train = pd.DataFrame(training_data)
    print(f"\n✅ Tạo được {len(df_train)} training samples")
    print(f"   -> Đậu: {(df_train['label']==1).sum()} ({(df_train['label']==1).mean()*100:.1f}%)")
    print(f"   -> Trượt: {(df_train['label']==0).sum()} ({(df_train['label']==0).mean()*100:.1f}%)")
    
    return df_train

# ================= 2. FEATURE ENGINEERING =================
def prepare_features(df_train):
    """
    Tạo features cho model
    """
    print("\n🚀 BƯỚC 2: Feature Engineering...")
    
    # Encode categorical features
    le_university = LabelEncoder()
    le_nganh = LabelEncoder()
    le_block = LabelEncoder()
    
    df_train['university_encoded'] = le_university.fit_transform(df_train['university_id'])
    df_train['nganh_encoded'] = le_nganh.fit_transform(df_train['ma_nganh'])
    df_train['block_encoded'] = le_block.fit_transform(df_train['to_hop_mon'])
    
    # Feature columns
    feature_cols = [
        'diem_thi',           # Điểm thi của thí sinh
        'gap',                # Khoảng cách so với điểm chuẩn năm đó
        'percentile',         # Percentile trong phân phối điểm
        'university_encoded', # Mã trường (encoded)
        'nganh_encoded',      # Mã ngành (encoded)
        'block_encoded',      # Tổ hợp môn (encoded)
        'nam'                 # Năm
    ]
    
    X = df_train[feature_cols].values
    y = df_train['label'].values
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    encoders = {
        'university': le_university,
        'nganh': le_nganh,
        'block': le_block,
        'feature_cols': feature_cols
    }
    
    print(f"   -> Features: {feature_cols}")
    print(f"   -> Shape: X={X_scaled.shape}, y={y.shape}")
    
    return X_scaled, y, encoders, scaler

# ================= 3. TRAIN MODELS =================
def train_models(X, y):
    """
    Train và so sánh các models
    """
    print("\n🚀 BƯỚC 3: Training Models...")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"   -> Train: {len(X_train)}, Test: {len(X_test)}")
    
    models = {
        'Random Forest': RandomForestClassifier(
            n_estimators=100, 
            max_depth=10, 
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        ),
        'Logistic Regression': LogisticRegression(
            max_iter=1000,
            random_state=42
        )
    }
    
    results = {}
    best_model = None
    best_score = 0
    
    for name, model in models.items():
        print(f"\n   📊 Training {name}...")
        
        # Train
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob)
        
        # Cross validation
        cv_scores = cross_val_score(model, X, y, cv=5, scoring='roc_auc')
        
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'roc_auc': roc_auc,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std()
        }
        
        print(f"      Accuracy: {accuracy:.4f}")
        print(f"      ROC-AUC: {roc_auc:.4f}")
        print(f"      CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
        
        if roc_auc > best_score:
            best_score = roc_auc
            best_model = model
            best_name = name
    
    print(f"\n🏆 Best Model: {best_name} (ROC-AUC: {best_score:.4f})")
    
    return best_model, results

# ================= 4. SAVE MODEL =================
def save_models(model, encoders, scaler, results):
    """
    Lưu model và các artifacts
    """
    print("\n💾 Đang lưu models...")
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    # Save main model
    joblib.dump(model, PROB_MODEL_OUTPUT)
    print(f"   -> Saved: {PROB_MODEL_OUTPUT}")
    
    # Save encoders
    joblib.dump(encoders, ENCODERS_OUTPUT)
    print(f"   -> Saved: {ENCODERS_OUTPUT}")
    
    # Save scaler
    joblib.dump(scaler, SCALER_OUTPUT)
    print(f"   -> Saved: {SCALER_OUTPUT}")
    
    # Save results summary
    results_summary = {name: {k: v for k, v in r.items() if k != 'model'} 
                       for name, r in results.items()}
    joblib.dump(results_summary, os.path.join(OUTPUT_DIR, 'admission_results.pkl'))
    
    print("\n✅ Đã lưu tất cả models!")

# ================= 5. DEMO PREDICTION =================
def demo_prediction(model, encoders, scaler):
    """
    Demo dự đoán xác suất đậu
    """
    print("\n" + "="*50)
    print("🎯 DEMO: DỰ ĐOÁN XÁC SUẤT ĐẬU")
    print("="*50)
    
    # Ví dụ: Thí sinh có điểm B00 = 24.5, muốn vào Y khoa ĐH Y Hà Nội
    test_cases = [
        {'diem': 27.0, 'university': 'YHN', 'nganh': '7720101', 'block': 'B00', 'desc': 'Y khoa Hà Nội - Điểm cao'},
        {'diem': 24.0, 'university': 'YHN', 'nganh': '7720101', 'block': 'B00', 'desc': 'Y khoa Hà Nội - Điểm TB'},
        {'diem': 22.0, 'university': 'YHN', 'nganh': '7720101', 'block': 'B00', 'desc': 'Y khoa Hà Nội - Điểm thấp'},
        {'diem': 24.0, 'university': 'DHY', 'nganh': '7720101', 'block': 'B00', 'desc': 'Y khoa Huế - Điểm TB'},
    ]
    
    for case in test_cases:
        try:
            # Encode
            uni_enc = encoders['university'].transform([case['university']])[0] if case['university'] in encoders['university'].classes_ else 0
            nganh_enc = encoders['nganh'].transform([case['nganh']])[0] if case['nganh'] in encoders['nganh'].classes_ else 0
            block_enc = encoders['block'].transform([case['block']])[0] if case['block'] in encoders['block'].classes_ else 0
            
            # Features (giả sử gap=0, percentile=50 cho demo)
            features = np.array([[case['diem'], 0, 50, uni_enc, nganh_enc, block_enc, 2025]])
            features_scaled = scaler.transform(features)
            
            # Predict
            prob = model.predict_proba(features_scaled)[0][1]
            
            print(f"\n📌 {case['desc']}")
            print(f"   Điểm: {case['diem']} | Tổ hợp: {case['block']}")
            print(f"   🎯 Xác suất đậu: {prob*100:.1f}%")
            
        except Exception as e:
            print(f"   ⚠️ Không thể dự đoán: {e}")

# ================= MAIN =================
if __name__ == "__main__":
    print("\n" + "="*60)
    print("   🎓 ADMISSION PROBABILITY MODEL TRAINING")
    print("="*60)
    
    # 1. Tạo training data
    df_train = create_training_data(SCORE_FOLDER, BENCHMARK_FILE, sample_per_major=300)
    
    if df_train.empty:
        print("❌ Không tạo được training data!")
        exit()
    
    # 2. Prepare features
    X, y, encoders, scaler = prepare_features(df_train)
    
    # 3. Train models
    best_model, results = train_models(X, y)
    
    # 4. Save
    save_models(best_model, encoders, scaler, results)
    
    # 5. Demo
    demo_prediction(best_model, encoders, scaler)
    
    print("\n" + "="*60)
    print("🎉 HOÀN THÀNH! Model đã sẵn sàng sử dụng.")
    print("="*60)
    
    input("\nẤn Enter để thoát...")
