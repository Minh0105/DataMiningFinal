import json
import sys

print("Starting script...", flush=True)

# Load notebook
print("Loading notebook...", flush=True)
with open('Advanced_University_Prediction.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

print(f"Total cells: {len(nb['cells'])}", flush=True)

# Find cell index that contains "BƯỚC 2: TÍNH PERCENTILE"
target_idx = None
for i, c in enumerate(nb['cells']):
    src = ''.join(c.get('source', []))
    if 'BƯỚC 2: TÍNH PERCENTILE CHO ĐIỂM CHUẨN' in src:
        target_idx = i
        print(f"Found STEP 2 at cell {i}")
        break

if target_idx is None:
    print("ERROR: Could not find STEP 2 cell")
    exit(1)

# New cell code for Stratified Sampling
new_cell_code = '''# =====================================================
# BƯỚC 3: TẠO TRAINING DATA VỚI STRATIFIED SAMPLING
# =====================================================
# Cách tiếp cận v3 + Stratified Sampling:
# - Sample điểm thi thật từ phân phối theo ZONES
# - Tính student_percentile (vị trí của thí sinh)
# - Label: 1 nếu student_percentile >= percentile_required, 0 nếu không

print("\\n[STEP 3] Tạo training data với Stratified Sampling...")

def stratified_sample(scores, diem_chuan, total_samples=300):
    """
    Stratified Sampling: Lấy mẫu thông minh theo 4 zones
    - Zone 1 (rớt chắc): gap < -5     -> 15% samples
    - Zone 2 (có thể rớt): -5 <= gap < -1  -> 25% samples
    - Zone 3 (ranh giới): -1 <= gap < +3   -> 40% samples
    - Zone 4 (đậu chắc): gap >= +3    -> 20% samples
    
    Ưu điểm:
    - Giữ lại extreme cases (model vẫn học điểm cao=đậu, thấp=rớt)
    - Tập trung vào vùng ranh giới (nơi model cần học phân biệt)
    - Model generalize tốt hơn khi deploy
    """
    gaps = scores - diem_chuan
    
    # Phân chia zones
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
    
    # Tỷ lệ samples cho mỗi zone
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
sample_per_major = 300  # Số samples mỗi ngành

for _, row in df_benchmark_v3.iterrows():
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
    
    # STRATIFIED SAMPLING thay vì random sampling
    sampled = stratified_sample(scores, diem_chuan, sample_per_major)
    
    for score in sampled:
        # Tính student_percentile
        student_percentile = (scores <= score).sum() / len(scores) * 100
        
        # Label: đậu nếu percentile >= percentile_required
        # (percentile cao = top đầu, percentile_required thấp = ngành khó)
        label = 1 if student_percentile >= percentile_required else 0
        
        training_data.append({
            'university_id': university_id,
            'ma_nganh': ma_nganh,
            'to_hop_mon': block,
            'student_percentile': student_percentile,
            'percentile_required': percentile_required,
            'label': label
        })

df_train_prob = pd.DataFrame(training_data)
print(f"\\n✅ Tạo được {len(df_train_prob):,} training samples")
print(f"   -> Đậu: {(df_train_prob['label']==1).sum():,} ({(df_train_prob['label']==1).mean()*100:.1f}%)")
print(f"   -> Trượt: {(df_train_prob['label']==0).sum():,} ({(df_train_prob['label']==0).mean()*100:.1f}%)")'''

# Create new cell
new_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": new_cell_code.split('\n')
}

# Convert source to list with newlines
new_cell['source'] = [line + '\n' for line in new_cell_code.split('\n')]
new_cell['source'][-1] = new_cell['source'][-1].rstrip('\n')  # Last line no newline

# Insert after target_idx
nb['cells'].insert(target_idx + 1, new_cell)
print(f"Inserted new cell at index {target_idx + 1}")

# Save
with open('Advanced_University_Prediction.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"Done! Now {len(nb['cells'])} cells")
print("Stratified Sampling cell added successfully!")
