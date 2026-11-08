"""Gói ``src`` chứa toàn bộ logic của Practice 1 - phân loại ảnh Fashion-MNIST.

Các module chính:
    config   - Cấu hình đường dẫn và siêu tham số dùng chung
    data     - Tải / nạp dữ liệu, Dataset và DataLoader
    model    - Kiến trúc CNN (FashionCNN)
    trainer  - Vòng lặp huấn luyện, lưu checkpoint
    evaluate - Đánh giá mô hình (accuracy, báo cáo, ma trận nhầm lẫn)
    utils    - Tiện ích: seed, thiết bị, vẽ biểu đồ, lưu kết quả
"""

__version__ = "1.0.0"
