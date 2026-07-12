import json # Thêm thư viện json
from qdrant_client import QdrantClient
from qdrant_client.http import models

client = QdrantClient(url="http://localhost:6333")

all_extracted_chunks = [] # Danh sách chứa toàn bộ payload
offset = None # Biến đánh dấu trang hiện tại

print("Đang truy xuất dữ liệu từ Qdrant...")

# Dùng vòng lặp while để lấy toàn bộ dữ liệu nếu có nhiều trang
while True:
    records, next_page_offset = client.scroll(
        collection_name="medical_docs",
        scroll_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="metadata.disease",
                    match=models.MatchText(text="thanh nhiệt") 
                )
            ]
        ),
        offset=offset, # Truyền offset để Qdrant biết đang đọc tới đâu
        limit=100      # Lấy 100 chunks mỗi trang cho nhanh
    )

    # Đưa phần nội dung (payload) của các chunk vừa lấy vào danh sách
    for record in records:
        all_extracted_chunks.append(record.payload)

    # Nếu next_page_offset trả về None, nghĩa là đã đọc hết dữ liệu -> thoát vòng lặp
    if next_page_offset is None:
        break
    
    # Nếu còn dữ liệu, cập nhật offset để vòng lặp sau lấy trang tiếp theo
    offset = next_page_offset

# Xử lý lưu file JSON
if not all_extracted_chunks:
    print("Không tìm thấy chunk nào khớp với bộ lọc!")
else:
    output_filename = "extracted_chunks.json"
    
    # Mở file và ghi dữ liệu dưới dạng JSON
    with open(output_filename, "w", encoding="utf-8") as json_file:
        json.dump(all_extracted_chunks, json_file, ensure_ascii=False, indent=4)
        
    print(f"\n✅ Đã trích xuất thành công {len(all_extracted_chunks)} chunks!")
    print(f"📁 Dữ liệu đã được lưu vào file: {output_filename}")