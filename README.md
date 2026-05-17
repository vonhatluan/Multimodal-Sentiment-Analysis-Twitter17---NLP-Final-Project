## 📁 Cấu Trúc Thư Mục Dự Án
```text
├── .gradio/               # Thư mục lưu cấu hình bộ đệm của giao diện Web
├── Datasets/              # Thư mục chứa tập dữ liệu thực nghiệm Twitter-17
│   ├── twitter_training.csv      # Tập dữ liệu thô dùng để huấn luyện mô hình
│   └── twitter_validation.csv    # Tập dữ liệu kiểm định cấu trúc
├── my_fast_roberta_model/ # Trọng số mô hình Custom RoBERTa (Quản lý qua Git LFS)
│   ├── model.safetensors         # Tệp trọng số mô hình cốt lõi (~500MB)
│   ├── config.json               # Cấu hình kiến trúc mạng mạng Transformer
│   └── [Các tệp Tokenizer định nghĩa từ điển mã từ...]
├── Đánh giá mô hình/       # Biểu đồ ma trận nhầm lẫn (Confusion Matrix) và đồ thị chỉ số
├── .gitattributes         # Cấu hình theo dõi tệp lớn của Git LFS
├── Mô hình phân tích cảm xúc từ dữ liệu Twitter-17.ipynb  # Mã nguồn thực nghiệm và phân tích dữ liệu
├── requirements.txt       # Danh sách các thư viện cần thiết phục vụ cài đặt môi trường
├── run_app.bat            # Tệp thực thi nhanh ứng dụng Web trên Windows
└── webapp.py              # Mã nguồn triển khai giao diện Web tương tác bằng Gradio

# Phân Tích Cảm Xúc Khía Cạnh Đa Phương Thức Trên Dữ Liệu Twitter-17
> **Hệ thống đối sánh hiệu năng đa phương pháp (Multi-Method Sentiment Analysis) và Triển khai Ứng dụng Web**

---

## 📑 Thông Tin Đề Tài
* **Môn học:** Xử lý ngôn ngữ tự nhiên (NLP)
* **Giảng viên hướng dẫn:** TS. Lê Quang Hùng
* **Sinh viên thực hiện:** Võ Nhật Luân
* **Mã số sinh viên:** 4651050152
* **Lớp:** Công nghệ Thông tin K46E
* **Đơn vị:** Khoa Công nghệ Thông tin - Trường Đại học Quy Nhơn (QNU)

---

## 🎯 Tổng Quan Đề Tài
Nghiên cứu này tập trung vào bài toán **Phân tích cảm xúc khía cạnh đa phương thức (Multimodal Aspect-Based Sentiment Analysis - MABSA)** trên không gian mạng xã hội Twitter. Hệ thống tiến hành thực nghiệm song song và đối sánh định lượng hiệu năng của **04 phương pháp tiếp cận chuyên sâu**:

1. **Custom RoBERTa:** Mô hình dựa trên kiến trúc Học sâu Transformer tiên tiến, được tinh chỉnh (Fine-tuning) chuyên biệt trên tập dữ liệu thực tế nhằm tối ưu hóa năng lực bóc tách ngữ cảnh hai chiều.
2. **DistilBERT:** Kiến trúc Transformer tinh gọn (kiểu tri thức BERT), tối ưu tốc độ tính toán nhưng vẫn giữ vững độ chính xác phân loại cao.
3. **VADER (Rule-based):** Giải thuật phân tích sắc thái dựa trên hệ thống luật ngôn ngữ và trọng số từ điển, tối ưu cho các thuật ngữ và tiếng lóng đặc thù của mạng xã hội.
4. **TextBlob (Lexicon-based):** Phương pháp tiếp cận truyền thống dựa trên kho từ điển ngôn ngữ học nhằm tính toán độ phân cực của văn bản.

---

## 📁 Cấu Trúc Thư Mục Dự Án
```text
├── .gradio/               # Thư mục lưu cấu hình bộ đệm của giao diện Web
├── Datasets/              # Thư mục chứa tập dữ liệu thực nghiệm Twitter-17
│   ├── twitter_training.csv      # Tập dữ liệu thô dùng để huấn luyện mô hình
│   └── twitter_validation.csv    # Tập dữ liệu kiểm định cấu trúc
├── my_fast_roberta_model/ # Trọng số mô hình Custom RoBERTa (Quản lý qua Git LFS)
│   ├── model.safetensors         # Tệp trọng số mô hình cốt lõi (~500MB)
│   ├── config.json               # Cấu hình kiến trúc mạng mạng Transformer
│   └── [Các tệp Tokenizer định nghĩa từ điển mã từ...]
├── Đánh giá mô hình/       # Biểu đồ ma trận nhầm lẫn (Confusion Matrix) và đồ thị chỉ số
├── .gitattributes         # Cấu hình theo dõi tệp lớn của Git LFS
├── Mô hình phân tích cảm xúc từ dữ liệu Twitter-17.ipynb  # Mã nguồn thực nghiệm và phân tích dữ liệu
├── requirements.txt       # Danh sách các thư viện cần thiết phục vụ cài đặt môi trường
├── run_app.bat            # Tệp thực thi nhanh ứng dụng Web trên Windows
└── webapp.py              # Mã nguồn triển khai giao diện Web tương tác bằng Gradio
