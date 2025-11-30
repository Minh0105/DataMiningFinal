import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 1. LOAD & CLEAN DATA
try:
    df = pd.read_csv('DiemChuan_Final_2018_2025.csv')
except:
    print("Không tìm thấy file!")
    exit()

# Chuyển điểm sang số
df['Điểm chuẩn'] = pd.to_numeric(df['Điểm chuẩn'], errors='coerce')
df.dropna(subset=['Điểm chuẩn'], inplace=True)

# LỌC DỮ LIỆU "CHUẨN" (Bỏ CLC, Phân hiệu, Liên thông...)
# Chỉ lấy những ngành có tên "Y khoa" hoặc "Y đa khoa" thuần túy
def is_standard_track(name):
    name = name.lower()
    keywords_exclude = ['clc', 'chất lượng cao', 'phân hiệu', 'liên thông', 
                        'hệ', 'địa chỉ', 'kết hợp', 'chứng chỉ', 'nước ngoài', 'qt', 'tiên tiến']
    # Phải có chữ Y khoa/Y đa khoa
    if 'y khoa' not in name and 'y đa khoa' not in name:
        return False
    # Không được chứa từ khóa loại trừ
    for kw in keywords_exclude:
        if kw in name:
            return False
    return True

df['Is_Standard'] = df['Tên ngành'].apply(is_standard_track)
df_clean = df[df['Is_Standard']].copy()

# 2. FEATURE ENGINEERING (TẠO CỘT VÙNG MIỀN)
# Tạo từ điển map vùng miền thủ công cho các trường lớn (Demo)
region_map = {
    'Đại học Y Hà Nội': 'Miền Bắc',
    'Đại học Y Dược Thái Bình': 'Miền Bắc',
    'Đại học Y Dược Hải Phòng': 'Miền Bắc',
    'Đại học Y Dược - Đại học Thái Nguyên': 'Miền Bắc',
    'Đại học Y Khoa Vinh': 'Miền Trung',
    'Đại học Y Dược - Đại học Huế': 'Miền Trung',
    'Khoa Y Dược - Đại học Đà Nẵng': 'Miền Trung',
    'Đại học Tây Nguyên': 'Miền Trung',
    'Đại học Y Dược TP.HCM': 'Miền Nam',
    'Đại học Y Khoa Phạm Ngọc Thạch': 'Miền Nam',
    'Đại học Y Dược Cần Thơ': 'Miền Nam',
    'Khoa Y - Đại học Quốc Gia TP HCM': 'Miền Nam',
    'Đại học Trà Vinh': 'Miền Nam'
}

# Hàm map vùng miền, mặc định là 'Khác' nếu không tìm thấy
df_clean['Vùng miền'] = df_clean['Tên trường'].map(region_map).fillna('Khác')
# Lọc bỏ nhóm 'Khác' để biểu đồ sạch
df_region = df_clean[df_clean['Vùng miền'] != 'Khác']

# 3. TÍNH TOÁN RANKING (CHO BUMP CHART)
# Với mỗi năm, xếp hạng các trường dựa trên điểm chuẩn cao nhất
df_rank = df_clean.groupby(['Nam', 'Tên trường'])['Điểm chuẩn'].max().reset_index()
df_rank['Rank'] = df_rank.groupby('Nam')['Điểm chuẩn'].rank(method='first', ascending=False)

# Chỉ lấy Top 10 trường xuất hiện nhiều nhất để vẽ cho đẹp
top_schools_list = df_rank['Tên trường'].value_counts().head(10).index
df_rank_filtered = df_rank[df_rank['Tên trường'].isin(top_schools_list)]


# 4. VẼ BIỂU ĐỒ DASHBOARD
fig = make_subplots(
    rows=2, cols=1,
    subplot_titles=("Sự thay đổi thứ hạng (Ranking) của Top 10 trường Y", 
                    "Chênh lệch độ khó giữa các Vùng miền (Trung bình điểm chuẩn)"),
    vertical_spacing=0.15
)

# --- Chart 1: Bump Chart (Ranking) ---
# Trong Plotly không có Bump Chart có sẵn, ta dùng Line Chart với y đảo ngược
for school in top_schools_list:
    school_data = df_rank_filtered[df_rank_filtered['Tên trường'] == school]
    fig.add_trace(
        go.Scatter(x=school_data['Nam'], y=school_data['Rank'], mode='lines+markers', name=school),
        row=1, col=1
    )
# Đảo ngược trục Y để Hạng 1 nằm trên cao nhất
fig.update_yaxes(autorange="reversed", title="Thứ hạng (Cao -> Thấp)", row=1, col=1)

# --- Chart 2: Regional Trend (Line Chart) ---
# Tính điểm trung bình theo Vùng và Năm
df_region_agg = df_region.groupby(['Nam', 'Vùng miền'])['Điểm chuẩn'].mean().reset_index()

colors_region = {'Miền Bắc': '#ef553b', 'Miền Trung': '#00cc96', 'Miền Nam': '#636efa'}
for region in ['Miền Bắc', 'Miền Trung', 'Miền Nam']:
    reg_data = df_region_agg[df_region_agg['Vùng miền'] == region]
    fig.add_trace(
        go.Scatter(x=reg_data['Nam'], y=reg_data['Điểm chuẩn'], mode='lines', 
                   name=region, line=dict(color=colors_region[region], width=4)),
        row=2, col=1
    )

fig.update_layout(height=1000, width=1200, title_text="PHÂN TÍCH VĨ MÔ TUYỂN SINH Y KHOA (HỆ CHUẨN)", hovermode="x unified")
fig.write_html("macro_analysis.html")
print("Đã tạo file macro_analysis.html")