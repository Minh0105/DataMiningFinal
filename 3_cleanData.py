# Bộ code này sẽ giải quyết triệt để 3 vấn đề chúng ta đã thảo luận:
# - Tách tổ hợp: Biến dòng "A00, B00" thành 2 dòng riêng biệt.
# - Xử lý trùng lặp: Giữ lại điểm cao nhất nếu trùng thông tin.
# - Chuẩn hóa ngành: Tách mã ngành rối rắm thành các cột thuộc tính sạch sẽ (Hệ đào tạo, Yêu cầu phụ, Phân hiệu...).

import pandas as pd
import numpy as np
# ================= CẤU HÌNH =================
# Tên file input của bạn (đảm bảo file csv đã có header mới như bạn cung cấp)
INPUT_FILE = 'DiemChuan_Final_2018_2025.csv'
OUTPUT_FILE = 'diem_chuan_cleaned.csv'

def clean_and_normalize_data():
    print(f"🔄 Đang đọc file: {INPUT_FILE}...")
    try:
        df = pd.read_csv(INPUT_FILE)
    except FileNotFoundError:
        print("❌ Lỗi: Không tìm thấy file csv đầu vào.")
        return

    # Kiểm tra nhanh xem cột có đúng chuẩn snake_case chưa
    expected_cols = ['nam', 'ten_truong', 'ma_nganh', 'ten_nganh', 'diem_chuan', 'to_hop_mon', 'ghi_chu', 'university_id']
    if not all(col in df.columns for col in expected_cols):
        print("⚠️ Cảnh báo: Tên cột trong file CSV chưa khớp hoàn toàn với cấu hình mới.")
        print(f"Các cột tìm thấy: {list(df.columns)}")
        # Có thể return hoặc tiếp tục tùy bạn, ở đây mình cho chạy tiếp nhưng báo lỗi nếu thiếu

    print(f"📊 Số dòng ban đầu: {len(df)}")

    # ---------------------------------------------------------
    # BƯỚC 1: XỬ LÝ TỔ HỢP MÔN (SPLIT & EXPLODE)
    # ---------------------------------------------------------
    # Chuyển đổi sang string và tách dấu phẩy
    df['to_hop_mon'] = df['to_hop_mon'].astype(str).str.split(', ')
    # Tách dòng (Explode)
    df = df.explode('to_hop_mon')
    # Xóa khoảng trắng thừa
    df['to_hop_mon'] = df['to_hop_mon'].str.strip()
    
    print(f"✅ Sau khi tách tổ hợp môn: {len(df)} dòng")

    # ---------------------------------------------------------
    # BƯỚC 2: XỬ LÝ ĐIỂM CHUẨN & TRÙNG LẶP
    # ---------------------------------------------------------
    # Chuyển điểm chuẩn sang số
    df['diem_chuan'] = pd.to_numeric(df['diem_chuan'], errors='coerce')
    df.dropna(subset=['diem_chuan'], inplace=True)

    # Sắp xếp giảm dần theo điểm để giữ lại điểm cao nhất (ưu tiên an toàn)
    # Các cột sort cũng dùng snake_case
    df = df.sort_values(by=['university_id', 'ma_nganh', 'nam', 'to_hop_mon', 'diem_chuan'], 
                        ascending=[True, True, True, True, False])
    
    # Xóa trùng lặp
    df = df.drop_duplicates(subset=['university_id', 'ma_nganh', 'nam', 'to_hop_mon'], keep='first')
    
    print(f"✅ Sau khi xử lý trùng lặp: {len(df)} dòng")

    # ---------------------------------------------------------
    # BƯỚC 3: TRÍCH XUẤT ĐẶC TRƯNG (FEATURE EXTRACTION)
    # ---------------------------------------------------------
    print("🔄 Đang chuẩn hóa mã ngành và tên ngành...")

    def extract_attributes(row):
        # Lấy dữ liệu thô từ các cột snake_case
        raw_code = str(row['ma_nganh']).strip()
        raw_name = str(row['ten_nganh']).lower().strip()
        
        # 1. ma_goc (Base Code)
        base_code = raw_code.split('_')[0].split('|')[0]
        if len(base_code) > 7 and base_code[-1].isalpha():
             base_code = base_code[:7]
        
        # 2. he_dao_tao (Training System)
        program = 'dai_tra' # Dùng không dấu hoặc snake_case cho value luôn để dễ code sau này
        if 'clc' in raw_code.lower() or 'chất lượng cao' in raw_name or 'tiên tiến' in raw_name:
            program = 'chat_luong_cao'
        elif 'liên thông' in raw_name or 'vừa làm vừa học' in raw_name:
            program = 'lien_thong'
        elif 'địa chỉ' in raw_name or 'đặt hàng' in raw_name:
            program = 'dat_hang'

        # 3. yeu_cau_phu (Sub Criteria)
        criteria = 'khong'
        if 'chứng chỉ' in raw_name or 'tiếng anh' in raw_name or 'ngoại ngữ' in raw_name:
            criteria = 'co_chung_chi'
        elif '_ap' in raw_code.lower() or '_a' in raw_code.lower():
             criteria = 'co_chung_chi'

        # 4. phan_hieu (Campus)
        campus = 'co_so_chinh'
        if 'thanh hóa' in raw_name or '_yht' in raw_code.lower():
            campus = 'phan_hieu_thanh_hoa'
        elif 'phân hiệu' in raw_name:
            campus = 'phan_hieu_tinh'

        # 5. doi_tuong (Target Group)
        target = 'toan_quoc'
        
        if '|' in raw_code:
            gender = 'chung'
            if 'nam' in raw_name and 'nữ' not in raw_name: gender = 'nam'
            elif 'nữ' in raw_name: gender = 'nu'
            
            region = ''
            if 'bắc' in raw_name: region = 'mien_bac'
            elif 'nam' in raw_name and 'miền' in raw_name: region = 'mien_nam'
            
            if gender != 'chung' or region != '':
                target = f"{gender}_{region}".strip('_')
        
        elif 'tp.hcm' in raw_name or raw_code.endswith('TP'):
            target = 'ho_khau_hcm'
        elif 'tỉnh' in raw_name or raw_code.endswith('TQ'):
            target = 'ho_khau_tinh'

        return pd.Series([base_code, program, criteria, campus, target])

    # Tạo các cột mới cũng theo chuẩn snake_case
    df[['ma_goc', 'he_dao_tao', 'yeu_cau_phu', 'phan_hieu', 'doi_tuong']] = df.apply(extract_attributes, axis=1)

    # ---------------------------------------------------------
    # BƯỚC 4: LỌC RÁC & LƯU FILE
    # ---------------------------------------------------------
    # Lọc bỏ hệ liên thông
    df_final = df[df['he_dao_tao'] != 'lien_thong'].copy()
    
    # Sắp xếp cột output
    cols_order = [
        'nam', 'ten_truong', 'ma_goc', 'ten_nganh', 'diem_chuan', 'to_hop_mon',
        'he_dao_tao', 'yeu_cau_phu', 'phan_hieu', 'doi_tuong',
        'ma_nganh', 'ghi_chu', 'university_id'
    ]
    # Lấy giao của 2 tập hợp cột để tránh lỗi
    cols_to_save = [c for c in cols_order if c in df_final.columns]
    
    df_final = df_final[cols_to_save]

    df_final.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
    
    print("------------------------------------------------")
    print(f"🎉 HOÀN TẤT! File sạch: {OUTPUT_FILE}")
    print(f"📊 Số dòng cuối cùng: {len(df_final)}")
    print("------------------------------------------------")

if __name__ == "__main__":
    clean_and_normalize_data()