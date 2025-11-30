import requests
import pandas as pd
import time
import random

# ================= CẤU HÌNH DANH SÁCH TRƯỜNG =================
# Thêm trường 'readable_name' để lưu tên hiển thị đẹp trong Excel
target_configs = [
    {'id': 432, 'name': 'YHB', 'readable_name': 'Đại học Y Hà Nội'},
    {'id': 247, 'name': 'YKV', 'readable_name': 'Đại học Y Khoa Vinh'},
    {'id': 431, 'name': 'YTC', 'readable_name': 'Đại học Y tế Công cộng'},
    {'id': 239, 'name': 'YTB', 'readable_name': 'Đại học Y Dược Thái Bình'},
    {'id': 215, 'name': 'DHY', 'readable_name': 'Đại học Y Dược - Đại học Huế'},
    {'id': 430, 'name': 'YQH', 'readable_name': 'Học viện Quân Y - Hệ Quân sự'},
    {'id': 321, 'name': 'YPB', 'readable_name': 'Đại học Y Dược Hải Phòng'},
    {'id': 301, 'name': 'TYS', 'readable_name': 'Đại học Y khoa Phạm Ngọc Thạch'},
    {'id': 298, 'name': 'YDS', 'readable_name': 'Đại học Y Dược TP HCM'},
    {'id': 374, 'name': 'DYH', 'readable_name': 'Học Viện Quân Y - Hệ Dân sự'},
    {'id': 312, 'name': 'DDY', 'readable_name': 'Khoa Y Dược - Đại học Đà Nẵng'},
    {'id': 235, 'name': 'YCT', 'readable_name': 'Đại học Y Dược Cần Thơ'},
    {'id': 335, 'name': 'DKY', 'readable_name': 'Đại học Kỹ thuật Y tế Hải Dương'},
    {'id': 315, 'name': 'YDN', 'readable_name': 'Đại học Kỹ thuật Y Dược Đà Nẵng'},
    {'id': 390, 'name': 'HYD', 'readable_name': 'Học viện Y Dược học cổ truyền Việt Nam'},
    {'id': 421, 'name': 'QHY', 'readable_name': 'Đại học Y Dược - Đại học Quốc Gia Hà Nội'},
    {'id': 300, 'name': 'QSY', 'readable_name': 'Khoa Y - Đại học Quốc Gia TP HCM'},
    {'id': 228, 'name': 'DTY', 'readable_name': 'Đại học Y Dược - Đại học Thái Nguyên'},
]

years = range(2018, 2026) 
api_url_pattern = "https://diemthi.vnexpress.net/tra-cuu-dai-hoc/loadbenchmark/id/{uid}/year/{year}/sortby/1/block_name/all"

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0',
    'X-Requested-With': 'XMLHttpRequest',
    'Referer': 'https://diemthi.vnexpress.net/'
}

all_data = []

print("Bắt đầu cào dữ liệu...")

# --- PHẦN 1: CÀO DỮ LIỆU ---
for config in target_configs:
    uni_id = config['id']
    uni_readable_name = config['readable_name']
    
    for year in years:
        url = api_url_pattern.format(uid=uni_id, year=year)
        try:
            print(f"[{config['name']}] Năm {year}...", end="")
            response = requests.get(url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                try:
                    data_json = response.json()
                    html_content = data_json.get('html', '')
                    
                    if html_content:
                        dfs = pd.read_html(html_content)
                        if len(dfs) > 0:
                            df = dfs[0]
                            # Gán thông tin cơ bản
                            df['Nam'] = year
                            df['University_ID'] = uni_id
                            
                            # YÊU CẦU 2: Thêm cột Tên trường đại học
                            df['Tên trường'] = uni_readable_name
                            
                            # Chuẩn hóa header (xóa khoảng trắng thừa)
                            df.columns = [str(c).strip() for c in df.columns]
                            
                            all_data.append(df)
                            print(" -> OK")
                except:
                    pass
            time.sleep(random.uniform(0.5, 1.5))
        except Exception as e:
            print(f"Lỗi: {e}")

# --- PHẦN 2: XỬ LÝ & LÀM SẠCH (QUAN TRỌNG) ---
if all_data:
    final_df = pd.concat(all_data, ignore_index=True)
    
    print("\nĐang xử lý dữ liệu...")

    # YÊU CẦU 1: Loại bỏ dòng trống (Dòng rác)
    # Logic: Dòng nào không có điểm chuẩn thì là dòng rác
    final_df = final_df.dropna(subset=['Điểm chuẩn'])

    # YÊU CẦU 3: Tách Mã ngành và Tên ngành riêng biệt
    # Hàm xử lý từng dòng
    def process_major_info(row):
        text = str(row['Tên, mã ngành']).strip()
        # Tìm khoảng trắng cuối cùng để cắt
        # Ví dụ: "Bảo hiểm 7340204" -> cắt chỗ khoảng trắng trước số 7
        parts = text.rsplit(' ', 1) 
        
        if len(parts) == 2:
            name = parts[0].strip()
            code = parts[1].strip()
            return name, code
        else:
            return text, "" # Không tìm thấy mã thì để trống

    # Áp dụng hàm trên
    final_df[['Tên ngành', 'Mã ngành']] = final_df.apply(
        lambda row: pd.Series(process_major_info(row)), axis=1
    )

    # Sắp xếp lại thứ tự cột cho đẹp mắt
    cols_order = [
        'Nam', 
        'Tên trường', 
        'Mã ngành', 
        'Tên ngành', 
        'Điểm chuẩn', 
        'Tổ hợp môn', 
        'Ghi chú',
        'University_ID'
    ]
    # Chỉ lấy những cột có tồn tại (đề phòng thiếu cột Ghi chú)
    available_cols = [c for c in cols_order if c in final_df.columns]
    final_df = final_df[available_cols]

    # Lưu file
    output_filename = f"DiemChuan_Final_{min(years)}_{max(years)}.xlsx"
    final_df.to_excel(output_filename, index=False)
    
    # Xuất CSV (encoding='utf-8-sig' để Excel đọc được tiếng Việt)
    file_csv = f"DiemChuan_Final_{min(years)}_{max(years)}.csv"
    final_df.to_csv(file_csv, index=False, encoding='utf-8-sig')
    print(f"-> Đã lưu CSV:   {file_csv}")
    
    print("-" * 30)
    print(f"HOÀN THÀNH! File: {output_filename}")
    print(f"Tổng số dòng sạch: {len(final_df)}")
    print(final_df.head()) # In thử 5 dòng đầu xem kết quả
else:
    print("Không có dữ liệu.")