print("--- SCRIPT ĐANG KHỞI ĐỘNG... ---")

import pandas as pd
import numpy as np
import os
import glob
import joblib
import re
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error
import warnings

warnings.filterwarnings("ignore")

# ================= CẤU HÌNH (SỬA Ở ĐÂY) =================
SCORE_FOLDER = r'diem_thi_thptqg'  # <--- Đường dẫn folder điểm thi
BENCHMARK_FILE = 'diem_chuan_cleaned.csv'     # File điểm chuẩn (có nhiều data và cột he_dao_tao)

# CẤU HÌNH THƯ MỤC OUTPUT MỚI
OUTPUT_DIR = 'model_artifacts'  # Tên thư mục bạn muốn chứa file .pkl

# Tự động nối đường dẫn (Không cần sửa 2 dòng dưới)
MODEL_OUTPUT = os.path.join(OUTPUT_DIR, 'university_ranking_model_2026.pkl')
LOOKUP_OUTPUT = os.path.join(OUTPUT_DIR, 'score_distribution_2025.pkl')

# Định nghĩa TẤT CẢ các tổ hợp môn phổ biến
BLOCK_MAP = {
    'A00': ['toan', 'vat_ly', 'hoa_hoc'],
    'A01': ['toan', 'vat_ly', 'ngoai_ngu'],
    'A02': ['toan', 'vat_ly', 'sinh_hoc'],
    'B00': ['toan', 'hoa_hoc', 'sinh_hoc'],
    'B08': ['toan', 'sinh_hoc', 'ngoai_ngu'],
    'C00': ['ngu_van', 'lich_su', 'dia_ly'],
    'C03': ['ngu_van', 'toan', 'lich_su'],
    'C04': ['ngu_van', 'toan', 'dia_ly'],
    'D01': ['toan', 'ngu_van', 'ngoai_ngu'],
    'D07': ['toan', 'hoa_hoc', 'ngoai_ngu'],
    'D08': ['toan', 'sinh_hoc', 'ngoai_ngu'],  # Giống B08 nhưng tên khác
    'D13': ['toan', 'ngu_van', 'sinh_hoc'],
    'D66': ['ngu_van', 'gdcd', 'ngoai_ngu'],
}

# ================= 1. HÀM XỬ LÝ ĐIỂM THI =================
def build_percentile_lookup(score_folder):
    print(f"\n🚀 BƯỚC 1: Xây dựng phân phối điểm thi từ: {score_folder}")
    lookup_dict = {}
    files = glob.glob(os.path.join(score_folder, "*.csv"))
    
    if not files:
        print(f"❌ LỖI: Không tìm thấy file .csv nào trong {score_folder}")
        return {}

    print(f"   -> Tìm thấy {len(files)} file điểm thi.")

    for file_path in files:
        filename = os.path.basename(file_path)
        try:
            year = int(re.search(r'\d{4}', filename).group())
        except:
            continue
            
        print(f"   -> Đang xử lý file {filename}...")
        print(f"   -> Đang xử lý năm {year}...")
        all_cols = list(set([col for cols in BLOCK_MAP.values() for col in cols]))
        
        try:
            chunks = pd.read_csv(file_path, usecols=all_cols, chunksize=200000)
        except ValueError:
            continue

        block_data = {k: [] for k in BLOCK_MAP.keys()}
        
        for chunk in chunks:
            for block, cols in BLOCK_MAP.items():
                if not all(col in chunk.columns for col in cols): continue
                temp = chunk.dropna(subset=cols)
                if not temp.empty:
                    scores = temp[cols].sum(axis=1).tolist()
                    block_data[block].extend(scores)
        
        for block, scores in block_data.items():
            if not scores: continue
            scores_np = np.array(scores)
            scores_np.sort()
            scores_np = scores_np[::-1]
            
            df_score = pd.DataFrame({'score': scores_np})
            df_score['rank'] = df_score['score'].rank(method='min', ascending=False)
            total = len(df_score)
            
            lookup = df_score.groupby('score')['rank'].min().reset_index()
            lookup['percentile'] = (lookup['rank'] / total) * 100
            lookup_dict[(year, block)] = lookup.sort_values('score')

    print(f"\n✅ Hoàn thành Bước 1. Đã xử lý {len(lookup_dict)} cặp (Năm, Tổ hợp).")
    return lookup_dict

# ================= 2. HÀM QUY ĐỔI ĐIỂM CHUẨN =================
def expand_multiple_blocks(df):
    """
    Tách các dòng có nhiều tổ hợp (vd: 'A00, B00') thành nhiều dòng riêng
    """
    expanded_rows = []
    
    for _, row in df.iterrows():
        to_hop = str(row['to_hop_mon']).strip()
        
        # Tách nếu có nhiều tổ hợp (phân cách bằng dấu phẩy)
        blocks = [b.strip() for b in to_hop.split(',')]
        
        for block in blocks:
            if block and block in BLOCK_MAP:  # Chỉ lấy tổ hợp đã định nghĩa
                new_row = row.copy()
                new_row['to_hop_mon'] = block
                expanded_rows.append(new_row)
    
    return pd.DataFrame(expanded_rows)

def normalize_benchmark(benchmark_file, lookup_dict):
    print(f"\n🚀 BƯỚC 2: Quy đổi điểm chuẩn từ {benchmark_file}...")
    try:
        df = pd.read_csv(benchmark_file)
    except:
        return pd.DataFrame()
    
    # Tách các dòng có nhiều tổ hợp
    print(f"   -> Đang tách các dòng có nhiều tổ hợp...")
    df_expanded = expand_multiple_blocks(df)
    print(f"   -> Số dòng sau khi tách: {len(df_expanded)} (từ {len(df)} dòng gốc)")
    
    # LỌC CHỈ LẤY NGÀNH CÓ DỮ LIỆU NĂM 2025
    print(f"   -> Đang lọc các ngành có dữ liệu năm 2025...")
    nganh_co_2025 = df_expanded[df_expanded['nam'] == 2025][['university_id', 'ma_nganh', 'to_hop_mon']].drop_duplicates()
    print(f"   -> Tìm thấy {len(nganh_co_2025)} ngành có dữ liệu 2025")
    
    # Merge để chỉ giữ lại các dòng thuộc ngành có năm 2025
    df_filtered = df_expanded.merge(
        nganh_co_2025, 
        on=['university_id', 'ma_nganh', 'to_hop_mon'], 
        how='inner'
    )
    print(f"   -> Số dòng sau khi lọc: {len(df_filtered)}")
    
    def get_percentile(row):
        year = row['nam']
        block = str(row['to_hop_mon']).strip()
        score = row['diem_chuan']
        lookup = lookup_dict.get((year, block))
        if lookup is None: return np.nan 
        idx = np.searchsorted(lookup['score'], score, side='left')
        if idx < len(lookup): return lookup.iloc[idx]['percentile']
        else: return 0.01

    df_filtered['percentile_rank'] = df_filtered.apply(get_percentile, axis=1)
    df_clean = df_filtered.dropna(subset=['percentile_rank'])
    print(f"✅ Hoàn thành Bước 2. Dữ liệu huấn luyện: {len(df_clean)} dòng.")
    return df_clean

# ================= 3. MODEL ENGINE =================
def predict_weighted_average(values):
    n = len(values)
    if n == 0: return 0
    weights = np.arange(1, n + 1)
    return np.sum(values * weights) / weights.sum()

def predict_ets(values):
    try:
        if len(values) < 4: return None
        model = ExponentialSmoothing(values, trend='add', seasonal=None, initialization_method="estimated")
        fit = model.fit()
        return fit.forecast(1)[0]
    except: return None

def train_and_forecast_smart(df_labeled):
    print("\n🚀 BƯỚC 3: Đấu Model (Backtesting) & Dự báo 2026...")
    groups = df_labeled.groupby(['university_id', 'ma_nganh', 'to_hop_mon'])
    forecast_results = {}
    stats = {'WA': 0, 'ETS': 0}
    processed = 0
    total_groups = len(groups)

    for name, group in groups:
        processed += 1
        if processed % 100 == 0: print(f"   -> Đã xử lý {processed}/{total_groups} ngành...", end="\r")

        group = group.sort_values('nam')
        percentiles = group['percentile_rank'].values
        
        if len(percentiles) < 4:
            pred = predict_weighted_average(percentiles)
            forecast_results[name] = pred * 0.95
            continue

        train = percentiles[:-1]
        test = percentiles[-1]
        
        pred_wa = predict_weighted_average(train)
        pred_ets = predict_ets(train)
        
        err_wa = abs(pred_wa - test)
        err_ets = abs(pred_ets - test) if pred_ets is not None else float('inf')
        
        final_pred = 0
        if pred_ets is not None and err_ets < err_wa:
            stats['ETS'] += 1
            full_ets = predict_ets(percentiles)
            final_pred = full_ets if full_ets is not None else predict_weighted_average(percentiles)
        else:
            stats['WA'] += 1
            final_pred = predict_weighted_average(percentiles)
            
        forecast_results[name] = final_pred * 0.95

    print(f"\n✅ Hoàn thành. Tỉ số: Weighted Avg thắng {stats['WA']} - ETS thắng {stats['ETS']}")
    return forecast_results

# ================= MAIN =================
if __name__ == "__main__":
    print("\n================================================")
    print(f"INPUT FOLDER: {SCORE_FOLDER}")
    print(f"OUTPUT DIR:   {OUTPUT_DIR}")
    print("================================================")
    
    if not os.path.exists(SCORE_FOLDER):
        print(f"❌ LỖI: Thư mục input '{SCORE_FOLDER}' không tồn tại!")
        exit()

    # --- TẠO THƯ MỤC OUTPUT NẾU CHƯA CÓ ---
    if not os.path.exists(OUTPUT_DIR):
        print(f"📁 Đang tạo thư mục mới: {OUTPUT_DIR}")
        os.makedirs(OUTPUT_DIR)
    
    # 1. Build
    lookup_map = build_percentile_lookup(SCORE_FOLDER)
    if not lookup_map: exit()
        
    # 2. Normalize
    df_train = normalize_benchmark(BENCHMARK_FILE, lookup_map)
    if df_train.empty: exit()
        
    # 3. Train
    model_2026 = train_and_forecast_smart(df_train)
    
    # 4. Save
    print(f"\n💾 Đang lưu file kết quả vào '{OUTPUT_DIR}'...")
    joblib.dump(model_2026, MODEL_OUTPUT)
    
    lookup_2025 = {k: v for k, v in lookup_map.items() if k[0] == 2025}
    joblib.dump(lookup_2025, LOOKUP_OUTPUT)
    
    print("\n🎉🎉🎉 THÀNH CÔNG! KIỂM TRA THƯ MỤC:", os.path.abspath(OUTPUT_DIR))
    input("Ấn Enter để thoát...")