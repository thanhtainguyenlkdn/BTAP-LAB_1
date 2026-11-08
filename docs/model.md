# 🧠 Mô hình — Kiến trúc `FashionCNN`

> 🔗 **Liên kết:** [README](../README.md) · [data.md](data.md) · [training.md](training.md) · [evaluation.md](evaluation.md) · [demo.md](demo.md)

## 1. Định nghĩa
Mô hình được định nghĩa trong file `src/model.py` thông qua hàm `build_model(num_classes=10)`.
- **Tên class:** `FashionCNN`
- **Số lượng tham số:** **~0.87 triệu** (phù hợp để huấn luyện nhanh ngay cả trên CPU).

## 2. Chi tiết kiến trúc
Mạng bao gồm 2 khối Tích chập (Convolutional Blocks) và 1 khối Mạng truyền thẳng đa tầng (MLP) để phân loại.

| Khối | Các lớp (Layers) | Kích thước Đầu vào → Đầu ra |
|---|---|---|
| **Conv 1** | Conv2d(3×3) × 2 → BatchNorm → ReLU → MaxPool(2×2) → Dropout | `(1, 28, 28)` → `(32, 14, 14)` |
| **Conv 2** | Conv2d(3×3) × 2 → BatchNorm → ReLU → MaxPool(2×2) → Dropout | `(32, 14, 14)` → `(64, 7, 7)` |
| **MLP** | Flatten → Linear(3136→256) → BatchNorm → ReLU → Dropout → Linear(256→10) | `3136` → `256` → `10` |

## 3. Giải thích thiết kế
- **Hai khối Tích chập (Conv):** Đủ để trích xuất các đặc trưng từ cấp thấp (cạnh, góc) đến cấp cao (hình dạng quần áo) trên ảnh kích thước nhỏ 28×28.
- **Batch Normalization (BN):** Giúp quá trình huấn luyện hội tụ nhanh hơn, ổn định hơn và giảm sự phụ thuộc vào giá trị khởi tạo, cho phép dùng Learning Rate lớn hơn (1e-3).
- **Dropout & Weight Decay:** Là các kỹ thuật Regularization quan trọng, kết hợp với Data Augmentation giúp mô hình không bị Overfitting (học vẹt) khi huấn luyện quá dài (30 epochs).
- **MaxPooling (2×2):** Giảm kích thước không gian của ảnh, giúp giảm số lượng tham số và tăng khả năng bất biến với dịch chuyển nhỏ (translation invariance).

## 4. Luồng hoạt động (Forward Pass)
1. Ảnh xám 28×28 được đưa vào mạng dưới dạng Tensor `(Batch, 1, 28, 28)`.
2. Trích xuất đặc trưng qua **Khối Conv 1** và **Khối Conv 2**. Kết quả thu được một Tensor đặc trưng có kích thước `(Batch, 64, 7, 7)`.
3. Tensor này được làm phẳng (Flatten) thành vector có kích thước `3136`.
4. Đi qua lớp **Linear ẩn (256 chiều)**.
5. Đi qua lớp **Linear đầu ra (10 chiều)** tương ứng với 10 lớp.
6. Hàm Loss `CrossEntropyLoss` sẽ tự động áp dụng Softmax để tính xác suất khi huấn luyện.

## 5. Liên kết
- Quá trình tối ưu hóa mô hình này (Loss, Optimizer, Early Stopping) được mô tả tại: ➡️ [training.md](training.md).
- Cách nạp lại trọng số (`best_model.pth`) để đánh giá và suy luận: ➡️ [evaluation.md](evaluation.md).