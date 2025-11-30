"""
🎯 CÔNG CỤ GỢI Ý TỔ HỢP MÔN PHÙ HỢP
- Input: Điểm từng môn của thí sinh + Ngành/Trường muốn đăng ký
- Output: Xếp hạng các tổ hợp theo khả năng đậu
"""

import pandas as pd
import numpy as np
import joblib
import os

# ================= CẤU HÌNH =================
MODEL_FILE = 'model_artifacts/university_ranking_model_2026.pkl'
LOOKUP_FILE = 'model_artifacts/score_distribution_2025.pkl'
ANALYTICS_FILE = 'model_artifacts/model_analytics.pkl'
BENCHMARK_FILE = 'diem_chuan_cleaned.csv'

# Định nghĩa TẤT CẢ tổ hợp môn Y Dược
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

BLOCK_NAMES = {
    'A00': 'Toán - Lý - Hóa',
    'A01': 'Toán - Lý - Anh',
    'A02': 'Toán - Lý - Sinh',
    'B00': 'Toán - Hóa - Sinh', 
    'B08': 'Toán - Sinh - Anh',
    'D01': 'Toán - Văn - Anh',
    'D07': 'Toán - Hóa - Anh',
    'D08': 'Toán - Sinh - Anh',
    'D13': 'Toán - Văn - Sinh',
}

# ================= LOAD DATA =================
def load_resources():
    """Load model, lookup table và analytics"""
    print("📂 Đang tải dữ liệu...")
    
    if not os.path.exists(MODEL_FILE):
        print(f"❌ Không tìm thấy {MODEL_FILE}. Hãy chạy 4_train_advanced_win.py trước!")
        return None, None, None, None
    
    if not os.path.exists(LOOKUP_FILE):
        print(f"❌ Không tìm thấy {LOOKUP_FILE}. Hãy chạy 4_train_advanced_win.py trước!")
        return None, None, None, None
        
    model_2026 = joblib.load(MODEL_FILE)
    lookup_2025 = joblib.load(LOOKUP_FILE)
    df_benchmark = pd.read_csv(BENCHMARK_FILE)
    
    # Load analytics (confidence intervals, model selection)
    analytics = None
    if os.path.exists(ANALYTICS_FILE):
        analytics = joblib.load(ANALYTICS_FILE)
        print(f"✅ Đã tải analytics (CI, Model Selection)")
    
    print(f"✅ Đã tải {len(model_2026)} dự đoán ngành 2026")
    print(f"✅ Đã tải phân phối điểm {len(lookup_2025)} tổ hợp")
    
    return model_2026, lookup_2025, df_benchmark, analytics


# ================= TÍNH ĐIỂM THEO TỔ HỢP =================
def calculate_block_scores(student_scores):
    """
    Tính tổng điểm cho mỗi tổ hợp từ điểm của thí sinh
    
    Args:
        student_scores: dict {'toan': 8.5, 'vat_ly': 7.0, ...}
    
    Returns:
        dict: {'A00': 23.5, 'B00': 22.0, ...}
    """
    block_scores = {}
    
    for block, subjects in BLOCK_MAP.items():
        # Kiểm tra có đủ điểm các môn không
        if all(subj in student_scores and student_scores[subj] is not None for subj in subjects):
            total = sum(student_scores[subj] for subj in subjects)
            block_scores[block] = round(total, 2)
        else:
            block_scores[block] = None  # Không đủ điểm
            
    return block_scores


def get_percentile_from_score(score, block, lookup_2025):
    """Chuyển điểm thô -> percentile dựa trên phân phối 2025"""
    key = (2025, block)
    if key not in lookup_2025:
        return None
    
    lookup = lookup_2025[key]
    idx = np.searchsorted(lookup['score'].values, score, side='left')
    
    if idx < len(lookup):
        return lookup.iloc[idx]['percentile']
    else:
        return 0.01  # Top cao nhất


# ================= GỢI Ý TỔ HỢP =================
def recommend_combinations(student_scores, university_id, ma_nganh, 
                           model_2026, lookup_2025, df_benchmark, analytics=None):
    """
    Gợi ý tổ hợp môn phù hợp nhất cho thí sinh
    
    Returns:
        DataFrame với các cột: to_hop, diem_cua_ban, diem_chuan_du_doan, 
                               khoang_cach, kha_nang_dau, xep_hang
    """
    
    # Lấy confidence intervals từ analytics
    confidence_intervals = analytics.get('confidence_intervals', {}) if analytics else {}
    model_selection = analytics.get('model_selection', {}) if analytics else {}
    
    # 1. Tính điểm theo từng tổ hợp
    block_scores = calculate_block_scores(student_scores)
    
    # 2. Lấy các tổ hợp mà ngành này xét tuyển
    nganh_info = df_benchmark[
        (df_benchmark['university_id'] == university_id) & 
        (df_benchmark['ma_nganh'] == ma_nganh)
    ]
    
    if nganh_info.empty:
        print(f"❌ Không tìm thấy ngành {ma_nganh} tại trường {university_id}")
        return None
    
    available_blocks = nganh_info['to_hop_mon'].unique()
    ten_truong = nganh_info['ten_truong'].iloc[0]
    ten_nganh = nganh_info['ten_nganh'].iloc[0]
    
    print(f"\n🏫 Trường: {ten_truong}")
    print(f"📚 Ngành: {ten_nganh} ({ma_nganh})")
    print(f"📋 Các tổ hợp xét tuyển: {', '.join(available_blocks)}")
    
    # 3. Phân tích từng tổ hợp
    results = []
    
    for block in available_blocks:
        if block not in block_scores or block_scores[block] is None:
            continue
            
        diem_cua_ban = block_scores[block]
        
        # Lấy điểm chuẩn dự đoán 2026 (dạng percentile)
        key = (university_id, ma_nganh, block)
        
        if key in model_2026:
            predicted_percentile = model_2026[key]
            
            # Chuyển percentile của thí sinh
            student_percentile = get_percentile_from_score(diem_cua_ban, block, lookup_2025)
            
            if student_percentile is not None:
                # Tính khoảng cách (percentile thấp = điểm cao = tốt hơn)
                # Nếu student_percentile < predicted_percentile => Đậu
                khoang_cach = predicted_percentile - student_percentile
                
                # Ước tính khả năng đậu (đơn giản)
                if khoang_cach > 10:
                    kha_nang = "🟢 Cao (>80%)"
                    do_uu_tien = 3
                elif khoang_cach > 0:
                    kha_nang = "🟡 Trung bình (50-80%)"
                    do_uu_tien = 2
                elif khoang_cach > -5:
                    kha_nang = "🟠 Thấp (30-50%)"
                    do_uu_tien = 1
                else:
                    kha_nang = "🔴 Rất thấp (<30%)"
                    do_uu_tien = 0
                  # Lấy điểm chuẩn năm gần nhất để tham khảo
                diem_chuan_2025 = nganh_info[
                    (nganh_info['to_hop_mon'] == block) & 
                    (nganh_info['nam'] == nganh_info['nam'].max())
                ]['diem_chuan'].values
                
                diem_chuan_ref = diem_chuan_2025[0] if len(diem_chuan_2025) > 0 else None
                
                # Lấy confidence interval
                ci = confidence_intervals.get(key, 0)
                used_model = model_selection.get(key, 'WA')
                
                # Tính độ tin cậy
                if ci < 5:
                    do_tin_cay = "⭐⭐⭐"
                elif ci < 10:
                    do_tin_cay = "⭐⭐"
                else:
                    do_tin_cay = "⭐"
                
                results.append({
                    'to_hop': block,
                    'ten_to_hop': BLOCK_NAMES.get(block, block),
                    'diem_cua_ban': diem_cua_ban,
                    'diem_chuan_2025': diem_chuan_ref,
                    'percentile_ban': round(student_percentile, 2),
                    'percentile_chuan': round(predicted_percentile, 2),
                    'khoang_cach': round(khoang_cach, 2),
                    'kha_nang_dau': kha_nang,
                    'do_tin_cay': do_tin_cay,
                    'model': used_model,
                    'do_uu_tien': do_uu_tien
                })
    
    if not results:
        print("❌ Không thể phân tích - thiếu dữ liệu!")
        return None
      # 4. Sắp xếp theo độ ưu tiên
    df_result = pd.DataFrame(results)
    df_result = df_result.sort_values('do_uu_tien', ascending=False)
    df_result['xep_hang'] = range(1, len(df_result) + 1)
    
    return df_result[['xep_hang', 'to_hop', 'ten_to_hop', 'diem_cua_ban', 
                      'diem_chuan_2025', 'kha_nang_dau', 'do_tin_cay', 'khoang_cach']]


# ================= TÌM NGÀNH PHÙ HỢP =================
def find_suitable_majors(student_scores, model_2026, lookup_2025, df_benchmark, 
                         analytics=None, top_n=20, block_filter=None):
    """
    Tìm các ngành phù hợp với điểm của thí sinh
    
    Args:
        student_scores: Điểm các môn
        analytics: Analytics từ training (CI, model selection)
        top_n: Số ngành muốn hiển thị
        block_filter: Lọc theo tổ hợp cụ thể (vd: 'A00')
    
    Returns:
        DataFrame các ngành phù hợp nhất
    """
    
    # Lấy confidence intervals từ analytics
    confidence_intervals = analytics.get('confidence_intervals', {}) if analytics else {}
    
    block_scores = calculate_block_scores(student_scores)
    results = []
    
    for key, predicted_percentile in model_2026.items():
        university_id, ma_nganh, block = key
        
        if block_filter and block != block_filter:
            continue
        
        if block not in block_scores or block_scores[block] is None:
            continue
        
        diem_cua_ban = block_scores[block]
        student_percentile = get_percentile_from_score(diem_cua_ban, block, lookup_2025)
        
        if student_percentile is None:
            continue
        
        khoang_cach = predicted_percentile - student_percentile
        
        # Lấy thông tin ngành
        nganh_info = df_benchmark[
            (df_benchmark['university_id'] == university_id) & 
            (df_benchmark['ma_nganh'] == ma_nganh) &
            (df_benchmark['to_hop_mon'] == block)
        ]
        
        if nganh_info.empty:
            continue
        
        # Lấy confidence interval
        ci = confidence_intervals.get(key, 0)
        if ci < 5:
            do_tin_cay = "⭐⭐⭐"
        elif ci < 10:
            do_tin_cay = "⭐⭐"
        else:
            do_tin_cay = "⭐"
            
        results.append({
            'university_id': university_id,
            'ten_truong': nganh_info['ten_truong'].iloc[0],
            'ma_nganh': ma_nganh,
            'ten_nganh': nganh_info['ten_nganh'].iloc[0],
            'to_hop': block,
            'diem_cua_ban': diem_cua_ban,
            'khoang_cach': round(khoang_cach, 2),
            'do_tin_cay': do_tin_cay,
            'percentile_chuan': round(predicted_percentile, 2)
        })
    
    if not results:
        return None
    
    df_result = pd.DataFrame(results)
    
    # Lọc những ngành có khả năng đậu (khoảng cách > 0)
    df_dau = df_result[df_result['khoang_cach'] > 0].copy()
    df_dau = df_dau.sort_values('khoang_cach', ascending=False)
    
    return df_dau.head(top_n)


# ================= MAIN - DEMO =================
if __name__ == "__main__":
    print("\n" + "="*60)
    print("🎓 CÔNG CỤ GỢI Ý TỔ HỢP MÔN & NGÀNH PHÙ HỢP (V2.0)")
    print("="*60)
    
    # Load dữ liệu
    model_2026, lookup_2025, df_benchmark, analytics = load_resources()
    
    if model_2026 is None:
        print("\n⚠️ Hãy chạy file 4_train_advanced_win.py trước để tạo model!")
        input("Ấn Enter để thoát...")
        exit()
    
    # ===== DEMO: Điểm mẫu của thí sinh =====
    print("\n" + "-"*60)
    print("📝 NHẬP ĐIỂM CỦA BẠN - 6 môn Y Dược")
    print("   (Enter để dùng điểm mẫu)")
    print("-"*60)
    
        # Điểm mẫu - 6 môn Y Dược
    demo_scores = {
        'toan': 8.5,
        'vat_ly': 7.5,
        'hoa_hoc': 8.0,
        'sinh_hoc': 7.0,
        'ngu_van': 7.5,
        'ngoai_ngu': 8.0
    }
    
    try:
        user_input = input("Điểm Toán (mặc định 8.5): ").strip()
        demo_scores['toan'] = float(user_input) if user_input else 8.5
        
        user_input = input("Điểm Vật lý (mặc định 7.5): ").strip()
        demo_scores['vat_ly'] = float(user_input) if user_input else 7.5
        
        user_input = input("Điểm Hóa học (mặc định 8.0): ").strip()
        demo_scores['hoa_hoc'] = float(user_input) if user_input else 8.0
        
        user_input = input("Điểm Sinh học (mặc định 7.0): ").strip()
        demo_scores['sinh_hoc'] = float(user_input) if user_input else 7.0
        
        user_input = input("Điểm Ngữ văn (mặc định 7.5): ").strip()
        demo_scores['ngu_van'] = float(user_input) if user_input else 7.5
        
        user_input = input("Điểm Ngoại ngữ (mặc định 8.0): ").strip()
        demo_scores['ngoai_ngu'] = float(user_input) if user_input else 8.0
        
    except ValueError:
        print("⚠️ Điểm không hợp lệ, sử dụng điểm mẫu!")
    
    # Hiển thị điểm các tổ hợp
    print("\n" + "-"*60)
    print("📊 ĐIỂM THEO TỪNG TỔ HỢP:")
    print("-"*60)
    
    block_scores = calculate_block_scores(demo_scores)
    for block, score in block_scores.items():
        if score:
            print(f"   {block} ({BLOCK_NAMES[block]}): {score} điểm")
      # ===== TÍNH NĂNG 1: Gợi ý tổ hợp cho ngành cụ thể =====
    print("\n" + "="*60)
    print("🎯 TÍNH NĂNG 1: GỢI Ý TỔ HỢP CHO NGÀNH CỤ THỂ")
    print("="*60)
    
    # Hiển thị một số ngành mẫu
    print("\n📋 Một số ngành mẫu:")
    sample_nganh = df_benchmark.groupby(['university_id', 'ma_nganh', 'ten_truong', 'ten_nganh']).size().reset_index()
    print(sample_nganh[['university_id', 'ma_nganh', 'ten_truong', 'ten_nganh']].head(10).to_string(index=False))
    
    try:
        uni_id = int(input("\nNhập university_id (vd: 215): ").strip() or "215")
        ma_nganh = input("Nhập mã ngành (vd: 7720101): ").strip() or "7720101"
        
        result = recommend_combinations(
            demo_scores, uni_id, ma_nganh,
            model_2026, lookup_2025, df_benchmark, analytics
        )
        
        if result is not None:
            print("\n" + "-"*60)
            print("📊 KẾT QUẢ GỢI Ý TỔ HỢP:")
            print("-"*60)
            print(result.to_string(index=False))
            
    except Exception as e:
        print(f"⚠️ Lỗi: {e}")
    
    # ===== TÍNH NĂNG 2: Tìm ngành phù hợp =====
    print("\n" + "="*60)
    print("🔍 TÍNH NĂNG 2: TÌM NGÀNH PHÙ HỢP VỚI ĐIỂM CỦA BẠN")
    print("="*60)
    
    suitable = find_suitable_majors(
        demo_scores, model_2026, lookup_2025, df_benchmark, analytics, top_n=15
    )
    
    if suitable is not None and not suitable.empty:
        print("\n🏆 TOP 15 NGÀNH CÓ KHẢ NĂNG ĐẬU CAO NHẤT:")
        print("-"*80)
        print(suitable.to_string(index=False))
    else:
        print("❌ Không tìm thấy ngành phù hợp!")
    
    print("\n" + "="*60)
    input("Ấn Enter để thoát...")
