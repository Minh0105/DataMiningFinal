import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os

# ================= CẤU HÌNH & LOAD DATA =================
st.set_page_config(page_title="AI Dự Báo Đại Học 2026", page_icon="🎓", layout="wide")

MODEL_PATH = r'model_artifacts/university_ranking_model_2026.pkl'
LOOKUP_PATH = r'model_artifacts/score_distribution_2025.pkl'
ANALYTICS_PATH = r'model_artifacts/model_analytics.pkl'
INFO_PATH = 'diem_chuan_cleaned.csv'

# Admission Probability Model paths
PROB_MODEL_PATH = r'model_artifacts/admission_probability_model.pkl'
PROB_ENCODERS_PATH = r'model_artifacts/admission_encoders.pkl'
PROB_SCALER_PATH = r'model_artifacts/admission_scaler.pkl'

# Định nghĩa tổ hợp môn Y Dược (không có Sử, Địa)
BLOCK_MAP = {
    'A00': {'name': 'Toán - Lý - Hóa', 'subjects': ['toan', 'vat_ly', 'hoa_hoc']},
    'A01': {'name': 'Toán - Lý - Anh', 'subjects': ['toan', 'vat_ly', 'ngoai_ngu']},
    'A02': {'name': 'Toán - Lý - Sinh', 'subjects': ['toan', 'vat_ly', 'sinh_hoc']},
    'B00': {'name': 'Toán - Hóa - Sinh', 'subjects': ['toan', 'hoa_hoc', 'sinh_hoc']},
    'B08': {'name': 'Toán - Sinh - Anh', 'subjects': ['toan', 'sinh_hoc', 'ngoai_ngu']},
    'D01': {'name': 'Toán - Văn - Anh', 'subjects': ['toan', 'ngu_van', 'ngoai_ngu']},
    'D07': {'name': 'Toán - Hóa - Anh', 'subjects': ['toan', 'hoa_hoc', 'ngoai_ngu']},
    'D08': {'name': 'Toán - Sinh - Anh', 'subjects': ['toan', 'sinh_hoc', 'ngoai_ngu']},
    'D13': {'name': 'Toán - Văn - Sinh', 'subjects': ['toan', 'ngu_van', 'sinh_hoc']},
}

@st.cache_resource
def load_resources():
    """Load Model, Lookup table & Analytics"""
    try:
        model = joblib.load(MODEL_PATH)
        lookup = joblib.load(LOOKUP_PATH)
        
        # Load analytics (confidence intervals, model selection)
        analytics = None
        if os.path.exists(ANALYTICS_PATH):
            analytics = joblib.load(ANALYTICS_PATH)
        
        df_info = pd.read_csv(INFO_PATH)
        info_map = df_info.groupby(['university_id', 'ma_nganh', 'to_hop_mon']).first().reset_index()
        info_map = info_map.set_index(['university_id', 'ma_nganh', 'to_hop_mon']).to_dict('index')
        
        return model, lookup, info_map, df_info, analytics
    except Exception as e:
        st.error(f"Lỗi load dữ liệu: {e}")
        return None, None, None, None, None

@st.cache_resource
def load_probability_model():
    """Load Admission Probability Model"""
    try:
        if os.path.exists(PROB_MODEL_PATH):
            prob_model = joblib.load(PROB_MODEL_PATH)
            prob_encoders = joblib.load(PROB_ENCODERS_PATH)
            prob_scaler = joblib.load(PROB_SCALER_PATH)
            return prob_model, prob_encoders, prob_scaler
        return None, None, None
    except Exception as e:
        return None, None, None

model, lookup_2025, school_info, df_benchmark, analytics = load_resources()
prob_model, prob_encoders, prob_scaler = load_probability_model()

# Lấy confidence intervals từ analytics
confidence_intervals = analytics.get('confidence_intervals', {}) if analytics else {}
model_selection = analytics.get('model_selection', {}) if analytics else {}

# ================= HÀM TÍNH TOÁN CỐT LÕI =================

def format_percentile(percentile):
    """Format percentile thành text dễ đọc"""
    if percentile is None:
        return "N/A"
    if percentile < 1:
        return f"Top {percentile:.2f}%"
    elif percentile < 10:
        return f"Top {percentile:.1f}%"
    else:
        return f"Top {percentile:.0f}%"

def percentile_to_score(percentile, block, lookup_dict):
    """Chuyển đổi percentile ngược lại thành điểm (để tính điểm chuẩn dự báo 2026)
    
    Lookup table structure:
    - score: tăng dần (4.05 -> 30.0)
    - percentile: giảm dần (100% -> 0.002%)
    - Top X% nhỏ = điểm cao
    """
    key = (2025, block)
    if key not in lookup_dict:
        return None
    
    table = lookup_dict[key]
    
    # Percentile trong table giảm dần, nên cần tìm ngược
    # Ví dụ: Top 2% → tìm dòng có percentile <= 2 → lấy score tương ứng
    # Dùng searchsorted trên mảng đảo ngược
    percentile_values = table['percentile'].values[::-1]  # Đảo thành tăng dần
    score_values = table['score'].values[::-1]  # Đảo tương ứng
    
    idx = np.searchsorted(percentile_values, percentile, side='left')
    
    if idx < len(score_values):
        return score_values[idx]
    else:
        return score_values[-1]

def predict_admission_probability(diem, block, university_id, ma_nganh, predicted_percentile=None):
    """
    Tính xác suất đậu dựa trên PERCENTILE (không dùng ML model bị overfit)
    
    Logic:
    - So sánh percentile của thí sinh vs percentile yêu cầu của ngành
    - Nếu student_pct < required_pct (Top nhỏ hơn = điểm cao hơn) → xác suất cao
    - Dùng sigmoid function để smooth xác suất
    """
    try:
        # Get percentile của thí sinh
        pct_info = get_user_percentile_info(diem, block, lookup_2025)
        if pct_info is None:
            return None
        student_percentile = pct_info['percentile']
        
        # Lấy percentile yêu cầu của ngành (từ model dự báo 2026)
        if predicted_percentile is None:
            key = (university_id, ma_nganh, block)
            if key in model:
                predicted_percentile = model[key]
            else:
                return None
        
        # Tính khoảng cách percentile
        # Nếu student_pct < required_pct → dư điểm → gap dương
        # Nếu student_pct > required_pct → thiếu điểm → gap âm
        gap = predicted_percentile - student_percentile
        
        # Sigmoid function để smooth xác suất
        # gap = 0 → 50%
        # gap = 5 → ~88%
        # gap = 10 → ~99%
        # gap = -5 → ~12%
        # gap = -10 → ~1%
        import math
        probability = 1 / (1 + math.exp(-gap * 0.5))
        
        return probability * 100
        
    except Exception as e:
        return None

def get_user_percentile_info(score, block, lookup_dict):
    """Quy đổi điểm thi user sang Top % với đầy đủ thông tin
    Returns: dict với percentile, rank, total_students hoặc None nếu không tìm thấy
    """
    key = (2025, block)
    if key not in lookup_dict:
        return None
    
    table = lookup_dict[key]
    idx = np.searchsorted(table['score'], score, side='left')
    
    total_students = int(table.iloc[0]['rank']) if len(table) > 0 else 0
    
    if idx < len(table):
        percentile = table.iloc[idx]['percentile']
        rank = int(table.iloc[idx]['rank'])
    else:
        percentile = 0.01
        rank = 1
    
    return {
        'percentile': percentile,
        'rank': rank,
        'total_students': total_students,
        'formatted': format_percentile(percentile)
    }

def get_user_percentile(score, block, lookup_dict):
    """Quy đổi điểm thi user sang Top % dựa trên thước đo 2025 (backward compatible)"""
    info = get_user_percentile_info(score, block, lookup_dict)
    return info['percentile'] if info else None

def calculate_all_block_scores(student_scores, priority=0):
    """Tính điểm cho TẤT CẢ các tổ hợp từ điểm 6 môn"""
    block_scores = {}
    
    for block, info in BLOCK_MAP.items():
        subjects = info['subjects']
        if all(subj in student_scores and student_scores[subj] is not None for subj in subjects):
            total = sum(student_scores[subj] for subj in subjects) + priority
            block_scores[block] = round(total, 2)
        else:
            block_scores[block] = None
            
    return block_scores

def recommend_best_combination(student_scores, university_id, ma_nganh, priority=0):
    """Gợi ý tổ hợp môn tốt nhất cho một ngành cụ thể"""
    if df_benchmark is None or model is None:
        return None
    
    nganh_info = df_benchmark[
        (df_benchmark['university_id'] == university_id) & 
        (df_benchmark['ma_nganh'] == ma_nganh)
    ]
    
    if nganh_info.empty:
        return None
    
    available_blocks = nganh_info['to_hop_mon'].unique()
    block_scores = calculate_all_block_scores(student_scores, priority)
    
    results = []
    for block in available_blocks:
        if block not in block_scores or block_scores[block] is None:
            continue
        if block not in BLOCK_MAP:
            continue
            
        diem_cua_ban = block_scores[block]
        key = (university_id, ma_nganh, block)
        
        if key not in model:
            continue
            
        predicted_percentile = model[key]
        student_pct_info = get_user_percentile_info(diem_cua_ban, block, lookup_2025)
        
        if student_pct_info is None:
            continue
        
        student_percentile = student_pct_info['percentile']
        khoang_cach = predicted_percentile - student_percentile
        
        # Lấy điểm chuẩn các năm cho tổ hợp này
        history = nganh_info[nganh_info['to_hop_mon'] == block].sort_values('nam', ascending=False)
        dc_2025 = history[history['nam'] == 2025]['diem_chuan'].values
        dc_2024 = history[history['nam'] == 2024]['diem_chuan'].values
        dc_2023 = history[history['nam'] == 2023]['diem_chuan'].values
        
        # Tính chênh lệch điểm so với ĐC 2025 (dễ hiểu hơn)
        dc_2025_val = dc_2025[0] if len(dc_2025) > 0 else None
        if dc_2025_val is not None:
            chenh_lech = diem_cua_ban - dc_2025_val
            if chenh_lech >= 0:
                chenh_lech_str = f"✅ +{chenh_lech:.1f}"
            else:
                chenh_lech_str = f"❌ {chenh_lech:.1f}"
        else:
            chenh_lech_str = "N/A"
        
        if khoang_cach > 10:
            kha_nang = "🟢 Rất cao"
            do_uu_tien = 4
        elif khoang_cach > 2:
            kha_nang = "🟢 Cao"
            do_uu_tien = 3
        elif khoang_cach > 0:
            kha_nang = "🟡 Trung bình"
            do_uu_tien = 2
        elif khoang_cach > -2:
            kha_nang = "🟠 Thấp"
            do_uu_tien = 1
        else:
            kha_nang = "🔴 Rất thấp"
            do_uu_tien = 0
        
        # Lấy confidence interval và model từ analytics
        ci = confidence_intervals.get(key, 0)
        used_model = model_selection.get(key, 'WA')
        
        # Tính độ tin cậy dựa trên CI (CI nhỏ = tin cậy cao)
        if ci < 5:
            do_tin_cay = "⭐⭐⭐"
        elif ci < 10:
            do_tin_cay = "⭐⭐"
        else:
            do_tin_cay = "⭐"
        
        results.append({
            'Tổ hợp': block,
            'Tên tổ hợp': BLOCK_MAP[block]['name'],
            'Điểm của bạn': diem_cua_ban,
            'Bạn (%)': student_pct_info['formatted'],
            'Yêu cầu (%)': format_percentile(predicted_percentile),
            'ĐC 2025': dc_2025_val,
            'Dư/Thiếu': chenh_lech_str,
            'Khả năng đậu': kha_nang,            'Độ tin cậy': do_tin_cay,
            'do_uu_tien': do_uu_tien
        })
    
    if not results:
        return None
    
    df_result = pd.DataFrame(results)
    df_result = df_result.sort_values('do_uu_tien', ascending=False)
    df_result['Xếp hạng'] = range(1, len(df_result) + 1)
    
    return df_result[['Xếp hạng', 'Tổ hợp', 'Tên tổ hợp', 'Điểm của bạn', 
                      'Bạn (%)', 'Yêu cầu (%)', 'ĐC 2025', 'Dư/Thiếu', 'Khả năng đậu', 'Độ tin cậy']]

def find_suitable_majors(student_scores, priority=0, top_n=1000):
    """Tìm tất cả ngành phù hợp với điểm của thí sinh"""
    if model is None or df_benchmark is None:
        return None
    
    block_scores = calculate_all_block_scores(student_scores, priority)
    results = []
    
    for key, predicted_percentile in model.items():
        university_id, ma_nganh, block = key
        
        if block not in block_scores or block_scores[block] is None:
            continue
        
        diem_cua_ban = block_scores[block]
        student_pct_info = get_user_percentile_info(diem_cua_ban, block, lookup_2025)
        
        if student_pct_info is None:
            continue
        
        student_percentile = student_pct_info['percentile']
        khoang_cach = predicted_percentile - student_percentile
        
        info = school_info.get((university_id, ma_nganh, block))
        if not info:
            continue
        
        # Lấy điểm chuẩn các năm
        history = df_benchmark[
            (df_benchmark['university_id'] == university_id) & 
            (df_benchmark['ma_nganh'] == ma_nganh) &
            (df_benchmark['to_hop_mon'] == block)
        ].sort_values('nam', ascending=False)
        
        dc_2025 = history[history['nam'] == 2025]['diem_chuan'].values
        dc_2024 = history[history['nam'] == 2024]['diem_chuan'].values
        dc_2023 = history[history['nam'] == 2023]['diem_chuan'].values
        
        # Tính điểm chuẩn DỰ BÁO 2026 từ percentile
        dc_2026_dubao = percentile_to_score(predicted_percentile, block, lookup_2025)
        
        # Tính chênh lệch điểm so với ĐC DỰ BÁO 2026 (thay vì 2025)
        dc_2025_val = dc_2025[0] if len(dc_2025) > 0 else None
        if dc_2026_dubao is not None:
            chenh_lech = diem_cua_ban - dc_2026_dubao
            if chenh_lech >= 0:
                chenh_lech_str = f"✅ +{chenh_lech:.1f}"
            else:
                chenh_lech_str = f"❌ {chenh_lech:.1f}"
        else:
            chenh_lech_str = "N/A"
        
        if khoang_cach > 10:
            kha_nang = "🟢 Rất cao"
            do_uu_tien = 4
        elif khoang_cach > 2:
            kha_nang = "🟢 Cao"
            do_uu_tien = 3
        elif khoang_cach > 0:
            kha_nang = "🟡 Trung bình"
            do_uu_tien = 2
        elif khoang_cach > -2:
            kha_nang = "🟠 Thấp"
            do_uu_tien = 1
        else:
            kha_nang = "🔴 Rất thấp"
            do_uu_tien = 0
        
        # Lấy confidence interval từ analytics
        ci = confidence_intervals.get(key, 0)
        if ci < 5:
            do_tin_cay = "⭐⭐⭐"
        elif ci < 10:
            do_tin_cay = "⭐⭐"
        else:
            do_tin_cay = "⭐"
        
        # Format percentile yêu cầu của ngành
        req_pct_str = format_percentile(predicted_percentile)
          # Dự đoán xác suất đậu dựa trên percentile
        admission_prob = predict_admission_probability(diem_cua_ban, block, university_id, ma_nganh, predicted_percentile)
        prob_str = f"{admission_prob:.0f}%" if admission_prob is not None else "N/A"
        
        results.append({
            'Trường': info['ten_truong'],
            'Ngành': info['ten_nganh'],
            'Tổ hợp': block,
            'Điểm bạn': diem_cua_ban,
            'Bạn (%)': student_pct_info['formatted'],
            'Yêu cầu (%)': req_pct_str,
            'Xác suất': prob_str,
            'ĐC 2026 (DB)': round(dc_2026_dubao, 1) if dc_2026_dubao else None,
            'Dư/Thiếu': chenh_lech_str,
            'ĐC 2025': dc_2025_val,
            'ĐC 2024': dc_2024[0] if len(dc_2024) > 0 else None,
            'ĐC 2023': dc_2023[0] if len(dc_2023) > 0 else None,
            'Khả năng': kha_nang,
            'Tin cậy': do_tin_cay,
            'do_uu_tien': do_uu_tien,
            'Hệ': info.get('he_dao_tao', 'Đại trà'),
            'university_id': university_id,
            'ma_nganh': ma_nganh
        })
    
    if not results:
        return None
    
    df_result = pd.DataFrame(results)
    # Sắp xếp theo độ ưu tiên (cao nhất lên đầu)
    df_result = df_result.sort_values(['do_uu_tien'], ascending=[False])
    
    return df_result.head(top_n)

# ================= GIAO DIỆN NGƯỜI DÙNG (UI) =================

# --- SIDEBAR: NHẬP ĐIỂM 6 MÔN ---
with st.sidebar:
    st.header("🎯 Nhập Điểm Của Bạn")
    
    col1, col2 = st.columns(2)
    with col1:
        diem_toan = st.number_input("Toán", 0.0, 10.0, 8.0, step=0.25, key="toan")
        diem_ly = st.number_input("Vật lý", 0.0, 10.0, 7.5, step=0.25, key="ly")
        diem_hoa = st.number_input("Hóa học", 0.0, 10.0, 8.0, step=0.25, key="hoa")
    
    with col2:
        diem_sinh = st.number_input("Sinh học", 0.0, 10.0, 7.0, step=0.25, key="sinh")
        diem_van = st.number_input("Ngữ văn", 0.0, 10.0, 7.5, step=0.25, key="van")
        diem_anh = st.number_input("Ngoại ngữ", 0.0, 10.0, 8.0, step=0.25, key="anh")
    
        student_scores = {
        'toan': diem_toan,
        'vat_ly': diem_ly,
        'hoa_hoc': diem_hoa,
        'sinh_hoc': diem_sinh,
        'ngu_van': diem_van,
        'ngoai_ngu': diem_anh
    }
    
    priority = st.number_input("Điểm ưu tiên:", 0.0, 3.0, 0.0, step=0.25)
    
    st.write("---")
    st.write("**📊 Điểm & Xếp hạng theo tổ hợp:**")
    block_scores = calculate_all_block_scores(student_scores, priority)
    
    # Hiển thị điểm và percentile cho mỗi tổ hợp
    for block, score in block_scores.items():
        if score is not None:
            pct_info = get_user_percentile_info(score, block, lookup_2025)
            if pct_info:
                st.write(f"**{block}:** {score:.2f} → **{pct_info['formatted']}**")
            else:
                st.write(f"**{block}:** {score:.2f}")
    
    st.write("---")
    st.caption("📈 *Top X% = Bạn xếp hạng cao hơn (100-X)% thí sinh*")

# --- MAIN PAGE ---
st.title("🎓 AI Dự Báo Cơ Hội Đại Học 2026")
st.markdown("*Phân tích dựa trên dữ liệu điểm chuẩn Y Dược 2018-2025*")

if model is None:
    st.error("❌ Chưa load được Model. Hãy kiểm tra lại file .pkl")
else:
    # ===== PHẦN 1: TÌM NGÀNH PHÙ HỢP =====
    st.write("---")
    st.header("🔍 Tìm Ngành Phù Hợp Với Điểm Của Bạn")
    
    suitable_majors = find_suitable_majors(student_scores, priority, top_n=1000)
    
    # Các cột hiển thị (thêm cột percentile)
    display_cols = ['Trường', 'Ngành', 'Tổ hợp', 'Điểm bạn', 'Bạn (%)', 'Yêu cầu (%)', 'Xác suất', 'ĐC 2026 (DB)', 'Dư/Thiếu', 'Khả năng', 'Tin cậy', 'Hệ']
    
    if suitable_majors is not None and not suitable_majors.empty:
        high_chance = suitable_majors[suitable_majors['Khả năng'].str.contains('Rất cao|Cao')]
        medium_chance = suitable_majors[suitable_majors['Khả năng'].str.contains('Trung bình')]
        low_chance = suitable_majors[suitable_majors['Khả năng'].str.contains('Thấp|Rất thấp')]
        
        # Hiển thị TOP ngành phù hợp nhất
        if not high_chance.empty:
            st.success(f"🏆 **TOP NGÀNH PHÙ HỢP NHẤT:** {high_chance.iloc[0]['Ngành']} - {high_chance.iloc[0]['Trường']} (Tổ hợp {high_chance.iloc[0]['Tổ hợp']})")
        
        tab1, tab2, tab3 = st.tabs([
            f"🟢 CƠ HỘI CAO ({len(high_chance)})",
            f"🟡 CƠ HỘI VỪA ({len(medium_chance)})", 
            f"🔴 CƠ HỘI THẤP ({len(low_chance)})"
        ])
        
        with tab1:
            if not high_chance.empty:
                st.success("✨ Các ngành bạn có khả năng đậu cao! (Sắp xếp từ phù hợp nhất)")
                st.dataframe(
                    high_chance[display_cols],
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.info("Chưa tìm thấy ngành nào trong nhóm này.")
        
        with tab2:
            if not medium_chance.empty:
                st.warning("⚠️ Các ngành có cơ hội 50/50, cần cân nhắc kỹ.")
                st.dataframe(
                    medium_chance[display_cols],
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.info("Chưa tìm thấy ngành nào trong nhóm này.")
        
        with tab3:
            if not low_chance.empty:
                st.error("⚡ Các ngành có cơ hội thấp, chỉ nên đặt làm NV cuối.")
                st.dataframe(
                    low_chance[display_cols],
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.info("Chưa tìm thấy ngành nào trong nhóm này.")
    else:
        st.warning("Không tìm thấy ngành phù hợp. Hãy kiểm tra lại điểm số.")
    
    # ===== PHẦN 2: GỢI Ý TỔ HỢP CHO NGÀNH CỤ THỂ =====
    st.write("---")
    st.header("🎯 Phân Tích Tổ Hợp Cho Ngành Cụ Thể")
    st.markdown("*Chọn trường và ngành bạn muốn để xem nên xét tổ hợp nào (chỉ ngành có dữ liệu 2025)*")
    
    if df_benchmark is not None:
        # Chỉ lấy các trường/ngành có dữ liệu năm 2025
        df_2025 = df_benchmark[df_benchmark['nam'] == 2025]
        
        truong_list = df_2025[['university_id', 'ten_truong']].drop_duplicates().sort_values('ten_truong')
        truong_dict = dict(zip(truong_list['university_id'], truong_list['ten_truong']))
        
        col1, col2 = st.columns(2)
        
        with col1:
            selected_uni = st.selectbox(
                "🏫 Chọn trường:",
                options=list(truong_dict.keys()),
                format_func=lambda x: truong_dict.get(x, str(x))
            )
        
        with col2:
            # Chỉ lấy ngành có dữ liệu 2025 của trường đã chọn
            nganh_cua_truong = df_2025[df_2025['university_id'] == selected_uni]
            nganh_list = nganh_cua_truong[['ma_nganh', 'ten_nganh']].drop_duplicates()
            nganh_dict = dict(zip(nganh_list['ma_nganh'], nganh_list['ten_nganh']))
            
            selected_nganh = st.selectbox(
                "📚 Chọn ngành:",
                options=list(nganh_dict.keys()),
                format_func=lambda x: f"{x} - {nganh_dict.get(x, '')}"
            )
        
        # Tự động phân tích khi chọn
        result = recommend_best_combination(student_scores, selected_uni, selected_nganh, priority)
        
        if result is not None and not result.empty:
            best = result.iloc[0]
            
            col_info1, col_info2 = st.columns([2, 1])
            with col_info1:
                st.info(f"**🏫 {truong_dict.get(selected_uni)}**  \n📚 {nganh_dict.get(selected_nganh)} ({selected_nganh})")
            with col_info2:
                st.success(f"✨ **Tổ hợp tốt nhất:** {best['Tổ hợp']}  \n{best['Khả năng đậu']}")
            
            st.write("**📊 So sánh các tổ hợp:**")
            st.dataframe(result, use_container_width=True, hide_index=True)
        else:
            st.warning("Không thể phân tích ngành này. Có thể chưa có dữ liệu.")

# --- FOOTER ---
st.write("---")
with st.expander("📖 Hướng dẫn sử dụng"):
    st.markdown("""
    ### Cách sử dụng:
    
    1. **Nhập điểm 6 môn** ở thanh bên trái (Toán, Lý, Hóa, Sinh, Văn, Anh)
    2. Hệ thống **tự động tính điểm** cho tất cả các tổ hợp Y Dược
    3. **Phần 1:** Xem danh sách ngành phù hợp nhất với điểm của bạn
    4. **Phần 2:** Chọn trường + ngành cụ thể để xem nên xét tổ hợp nào
    
    ### 📊 Hiểu về Percentile (Top X%):
    
    | Bạn (%) | Ý nghĩa |
    |---------|---------|
    | Top 1% | Bạn nằm trong **1% thí sinh điểm cao nhất** cả nước |
    | Top 5% | Bạn nằm trong **5% thí sinh điểm cao nhất** |
    | Top 10% | Bạn cao hơn **90%** thí sinh toàn quốc |
    
    **Ví dụ:** Nếu "Bạn (%)" là **Top 3.5%** và "Yêu cầu (%)" là **Top 2.0%**, nghĩa là:
    - Bạn đang ở vị trí Top 3.5% (cao hơn 96.5% thí sinh)
    - Ngành này yêu cầu Top 2.0% (cao hơn 98% thí sinh)
    - → Bạn cần cải thiện thêm để vào nhóm Top 2.0%
    
    ### Ý nghĩa màu sắc (Khả năng đậu):
    - 🟢 **Cao/Rất cao:** Bạn (%) tốt hơn Yêu cầu (%)
    - 🟡 **Trung bình:** Bạn sát ngưỡng yêu cầu
    - 🟠 **Thấp:** Bạn dưới ngưỡng một chút
    - 🔴 **Rất thấp:** Bạn cách xa ngưỡng yêu cầu
    
    ### Độ tin cậy (dựa trên Confidence Interval):
    - ⭐⭐⭐ **Rất cao:** Dự đoán chính xác (CI < 5)
    - ⭐⭐ **Trung bình:** Dự đoán tương đối (CI 5-10)
    - ⭐ **Thấp:** Dữ liệu biến động lớn (CI > 10)
    
    ### Dữ liệu phân phối điểm:
    - Sử dụng dữ liệu **THPT Quốc Gia 2018-2025** (~6.5 triệu thí sinh/năm)
    - Tính percentile theo từng tổ hợp môn (A00, B00, D01,...)
    - So sánh công bằng giữa các năm dù độ khó đề thi khác nhau
    
    ---
    ⚠️ **Lưu ý:** Kết quả chỉ mang tính tham khảo dựa trên dữ liệu lịch sử 2018-2025.
    """)