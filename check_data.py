import pandas as pd
df = pd.read_csv('diem_chuan_cleaned.csv')

# Lấy trường đầu tiên
first_uni = df['university_id'].iloc[0]
first_nganh = df[df['university_id'] == first_uni]['ma_nganh'].iloc[0]

print(f'Truong: {first_uni}')
print(f'Nganh: {first_nganh}')
print()

# Xem dữ liệu
data = df[(df['university_id'] == first_uni) & (df['ma_nganh'] == first_nganh)]
print('Du lieu:')
print(data[['nam', 'to_hop_mon', 'diem_chuan', 'ten_truong', 'ten_nganh']])
