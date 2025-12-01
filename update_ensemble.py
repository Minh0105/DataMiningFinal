import json

# Load notebook
with open('Advanced_University_Prediction.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

print(f"Total cells: {len(nb['cells'])}")

# Find cell with "BƯỚC 6: TÍNH PERCENTILE TRUNG BÌNH"
target_idx = None
for i, c in enumerate(nb['cells']):
    src = ''.join(c.get('source', []))
    if 'BƯỚC 6' in src and 'PERCENTILE' in src.upper():
        target_idx = i
        print(f"Found STEP 6 at cell {i}")
        break

if target_idx is None:
    print("ERROR: Could not find STEP 6 cell")
    exit(1)

# New code using Ensemble model_2026
new_code = '''# =====================================================
# BƯỚC 6: SỬ DỤNG ENSEMBLE MODEL 2026 (thay vì trung bình lịch sử)
# =====================================================
# Lập luận: Dùng output của Ensemble (WA, ETS, ARIMA, LR) làm percentile_required
# thay vì dùng trung bình lịch sử - vì Ensemble đã predict cho 2026!

print("\\n[STEP 6] Sử dụng Ensemble Model 2026...")

# model_2026 đã được train ở phần 1 (Ensemble: WA, ETS, ARIMA, LR)
# Format: {(university_id, ma_nganh, to_hop_mon): predicted_percentile_2026}

if 'model_2026' in dir() and model_2026:
    print("   ✅ Sử dụng Ensemble output (model_2026)")
    
    # Tạo DataFrame từ model_2026
    ensemble_percentile = pd.DataFrame([
        {'university_id': k[0], 'ma_nganh': k[1], 'to_hop_mon': k[2], 'predicted_percentile_2026': v}
        for k, v in model_2026.items()
    ])
    print(f"   -> {len(ensemble_percentile)} dự đoán percentile 2026 từ Ensemble")
    
    # Merge với thông tin điểm chuẩn để có avg_diem_chuan
    avg_diem = df_benchmark_v3.groupby(['university_id', 'ma_nganh', 'to_hop_mon']).agg({
        'diem_chuan': 'mean'
    }).reset_index()
    avg_diem.columns = ['university_id', 'ma_nganh', 'to_hop_mon', 'avg_diem_chuan']
    
    avg_percentile = ensemble_percentile.merge(avg_diem, on=['university_id', 'ma_nganh', 'to_hop_mon'], how='left')
    avg_percentile.columns = ['university_id', 'ma_nganh', 'to_hop_mon', 'avg_percentile_required', 'avg_diem_chuan']
else:
    print("   ⚠️ Không tìm thấy model_2026, sử dụng trung bình lịch sử")
    avg_percentile = df_benchmark_v3.groupby(['university_id', 'ma_nganh', 'to_hop_mon']).agg({
        'percentile_required': 'mean',
        'diem_chuan': 'mean'
    }).reset_index()
    avg_percentile.columns = ['university_id', 'ma_nganh', 'to_hop_mon', 'avg_percentile_required', 'avg_diem_chuan']

print(f"   -> {len(avg_percentile)} ngành")

# Hiển thị mẫu
print("\\n📋 Top 10 ngành khó nhất (yêu cầu percentile cao nhất):")
top_hard = avg_percentile.nsmallest(10, 'avg_percentile_required').copy()

# Merge để lấy tên
top_hard = top_hard.merge(
    df_benchmark_v3[['university_id', 'ma_nganh', 'ten_truong', 'ten_nganh']].drop_duplicates(),
    on=['university_id', 'ma_nganh']
)
print(top_hard[['ten_truong', 'ten_nganh', 'to_hop_mon', 'avg_diem_chuan', 'avg_percentile_required']].to_string(index=False))'''

# Update cell source
nb['cells'][target_idx]['source'] = [line + '\n' for line in new_code.split('\n')]
nb['cells'][target_idx]['source'][-1] = nb['cells'][target_idx]['source'][-1].rstrip('\n')

print(f"Updated cell {target_idx}")

# Save
with open('Advanced_University_Prediction.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("Done! Cell updated to use Ensemble model_2026")
