from bs4 import BeautifulSoup
import pandas as pd
import os

# 1. Đường dẫn file HTML bạn đã lưu
file_path = 'sample.html'

# Kiểm tra file có tồn tại không
if not os.path.exists(file_path):
    print(f"Lỗi: Không tìm thấy file '{file_path}'")
else:
    # 2. Đọc file HTML
    with open(file_path, 'r', encoding='utf-8') as f:
        html_content = f.read()

    # 3. Khởi tạo BeautifulSoup để phân tích
    soup = BeautifulSoup(html_content, 'html.parser')

    # Tìm tất cả các thẻ <li> chứa thông tin trường
    # Class trong file bạn gửi là "lookup__result"
    items = soup.find_all('li', class_='lookup__result')

    results = []

    print(f"Đang xử lý {len(items)} trường từ file HTML...")

    for item in items:
        try:
            # --- TRÍCH XUẤT DỮ LIỆU ---
            
            # A. Lấy Mã trường (Code) - nằm trong class "lookup__result-code"
            code_div = item.find('div', class_='lookup__result-code')
            uni_code = code_div.get_text(strip=True) if code_div else ""

            # B. Lấy Tên trường (Name) - nằm trong class "lookup__result-name"
            # Lưu ý: Tên nằm trong thẻ <strong> bên trong div này
            name_div = item.find('div', class_='lookup__result-name')
            uni_name = ""
            if name_div and name_div.find('strong'):
                uni_name = name_div.find('strong').get_text(strip=True)

            # C. Lấy ID (Quan trọng nhất) - nằm trong thuộc tính 'data-id' của button
            # Cách này an toàn hơn regex URL nhiều
            action_div = item.find('div', class_='lookup__result-action')
            uni_id = None
            if action_div:
                btn = action_div.find('button', attrs={'data-id': True})
                if btn:
                    uni_id = btn['data-id']

            # D. Nếu muốn lấy URL (để tham khảo)
            url_suffix = ""
            link_tag = code_div.find('a') if code_div else None
            if link_tag and link_tag.get('href'):
                url_suffix = link_tag['href']

            # Thêm vào danh sách kết quả
            if uni_id and uni_code:
                results.append({
                    'id': int(uni_id),
                    'code': uni_code,
                    'name': uni_name,
                    'url': f"https://diemthi.vnexpress.net{url_suffix}"
                })
        
        except Exception as e:
            print(f"Lỗi khi parse một dòng: {e}")

    # 4. Xuất kết quả
    if results:
        df = pd.DataFrame(results)
        print("\n=== KẾT QUẢ ĐỌC TỪ FILE ===")
        print(df[['id', 'code', 'name']].to_string(index=False))

        # TẠO CODE CONFIG ĐỂ BẠN COPY NGAY
        print("\n" + "="*40)
        print("COPY ĐOẠN NÀY VÀO SCRIPT CÀO ĐIỂM CHUẨN:")
        print("="*40)
        print("target_configs = [")
        for item in results:
            print(f"    {{'id': {item['id']}, 'name': '{item['code']}', 'readable_name': '{item['name']}'}},")
        print("]")
        print("="*40)
    else:
        print("Không tìm thấy dữ liệu nào trong file HTML.")