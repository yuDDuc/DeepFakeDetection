# 🛡️ DeepFake Detection System (MTCNN + Xception)

Hệ thống phát hiện video giả mạo (DeepFake) sử dụng kiến trúc mô hình Xception kết hợp với kỹ thuật trích xuất khuôn mặt dựa trên phân tích chuyển động bằng MTCNN. Dự án được tối ưu hóa cho tập dữ liệu FaceForensics++.

## 📋 Mục lục
- [Tổng quan quy trình](#tổng-quan-quy-trình)
- [Cài đặt](#cài-đặt)
- [Cấu trúc mô hình](#cấu-trúc-mô-hình)
- [Chiến lược huấn luyện](#chiến-lược-huấn-luyện)
- [Đánh giá](#đánh-giá)

## 🔄 Tổng quan quy trình (Pipeline)

1.  **Tiền xử lý (Preprocessing):**
    *   Sử dụng **MTCNN** để nhận diện khuôn mặt.
    *   **Motion-based selection:** Tính toán sự sai khác giữa các khung hình (`absdiff`) để chọn ra 10 khung hình có chuyển động mạnh nhất (nơi DeepFake thường lộ lỗi artifacts).
    *   Chuẩn hóa ảnh về kích thước $299 \times 299$ để phù hợp với Xception.
2.  **Huấn luyện (Training):**
    *   Sử dụng mạng **Xception** đã được huấn luyện trước trên ImageNet.
    *   Fine-tuning qua 3 giai đoạn để đạt độ chính xác tối ưu.
3.  **Dự đoán (Inference):**
    *   Dự đoán theo cấp độ Video (**Video-level**): Lấy trung bình xác suất của 10 khung hình để đưa ra kết quả cuối cùng.

## 🚀 Cài đặt

1. **Clone project:**
```bash
git clone <your-repo-url>
cd DFD
```

2. **Cài đặt thư viện:**
```bash
pip install -r requirements.txt
```

3. **Cấu hình dữ liệu:**
Chỉnh sửa đường dẫn `FF_REAL_PATH` và `FF_FAKE_PATH` trong file notebook/script để trỏ đến tập dữ liệu FaceForensics++ của bạn.

## 🧠 Cấu trúc mô hình

*   **Base:** Xception (Frozen/Unfrozen tùy giai đoạn).
*   **GlobalAveragePooling2D:** Giảm số lượng tham số và tránh overfitting.
*   **Dropout (0.5):** Tăng khả năng tổng quát hóa của mô hình.
*   **Dense (2, Softmax):** Phân loại REAL và FAKE.

## 📈 Chiến lược huấn luyện

| Giai đoạn | Mục tiêu | Learning Rate |
| :--- | :--- | :--- |
| **Stage 1** | Chỉ huấn luyện lớp phân loại (Head) | 1e-3 |
| **Stage 2** | Mở khóa 30 lớp cuối của Xception | 1e-4 |
| **Stage 3** | Tinh chỉnh toàn bộ mạng (Full Fine-tune) | 1e-5 |

## 📊 Kết quả & Đánh giá

Mô hình được đánh giá trên tập Test (15%) với các chỉ số:
*   **ROC-AUC Score:** Đánh giá khả năng phân loại tổng quát.
*   **Precision-Recall Curve:** Tìm ngưỡng (threshold) tối ưu để cân bằng giữa việc nhận dạng nhầm và bỏ sót.
*   **Confusion Matrix:** Trực quan hóa kết quả dự đoán trên từng video.

## 💻 Sử dụng thực tế

Bạn có thể sử dụng hàm `predict_video` để kiểm tra một video bất kỳ:

```python
score, verdict = predict_video("path/to/video.mp4", model, detector)
print(f"Kết quả: {verdict} với {score*100:.2f}% xác suất là FAKE")
```
