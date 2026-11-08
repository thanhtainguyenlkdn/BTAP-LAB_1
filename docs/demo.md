# 🎯 Demo — Suy luận 1 ảnh (Inference)

> 🔗 **Liên kết:** [README](../README.md) · [data.md](data.md) · [model.md](model.md) · [training.md](training.md) · [evaluation.md](evaluation.md)

## 1. Yêu cầu
Trước khi chạy demo, bạn cần đảm bảo đã có file trọng số tốt nhất tại `models/best_model.pth`. 
*(File này được tạo ra khi bạn chạy `python main.py` hoặc chạy xong Notebook 02).*

## 2. Hàm suy luận chính
Hàm `predict_single(model, image01, device)` nằm trong `src/evaluate.py` được thiết kế để nhận đầu vào là một bức ảnh thô và tự động thực hiện các bước tiền xử lý.
- **Đầu vào:** Mảng numpy của ảnh (giá trị pixel từ 0–255 hoặc 0–1).
- **Tiền xử lý tự động:** Đưa về khoảng `[0, 1]` (chia 255 nếu cần) và chuẩn hoá bằng công thức `(x - 0.2860) / 0.3530`.
- **Đầu ra:** Trả về một tuple gồm 3 phần tử:
  1. `pred_idx`: Chỉ số lớp dự đoán (0-9).
  2. `pred_name`: Tên lớp dự đoán (VD: "Sneaker").
  3. `probs`: Mảng xác suất (10 phần tử) của cả 10 lớp.

## 3. Code Demo cơ bản
Dưới đây là đoạn code mẫu để dự đoán một ảnh lấy ngẫu nhiên từ tập Test:

```python
import numpy as np
from src import config, data, evaluate, utils
from src.trainer import load_trained_model

# 1. Khởi tạo thiết bị và nạp mô hình
device = utils.get_device()
model, _ = load_trained_model(config.BEST_MODEL_PATH, device=device)

# 2. Lấy ngẫu nhiên 1 ảnh từ tập test
X_train, y_train, X_test, y_test = data.load_raw_arrays()
i = np.random.randint(len(X_test))
image01 = X_test[i] / 255.0  # Đưa về khoảng [0, 1] (hàm predict_single cũng tự làm việc này)

# 3. Dự đoán
pred_idx, pred_name, probs = evaluate.predict_single(model, image01, device)

print("Nhãn thật :", config.CLASS_NAMES[y_test[i]])
print("Dự đoán   :", pred_name, f"({probs[pred_idx]:.2%})")
## 4. Trực quan hoá phân phối xác suất

Để xem mô hình "tự tin" như thế nào về dự đoán của mình, ta có thể vẽ biểu đồ cột cho 10 lớp. Notebook 03 đã tích hợp sẵn đoạn code sau:

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 4))
ax.bar(config.CLASS_NAMES, probs, color="tab:blue")
ax.set_xticks(range(len(config.CLASS_NAMES)))
ax.set_xticklabels(config.CLASS_NAMES, rotation=30, ha="right")
ax.set_ylabel("Xác suất")
ax.set_title(f"Kết quả: {pred_name}")

# Lưu biểu đồ vào outputs/figures/
utils.save_figure(fig, "03_prediction_example")

plt.show()
```

## 5. Suy luận với ảnh bên ngoài

Nếu bạn có một bức ảnh chụp quần áo bên ngoài và muốn mô hình dự đoán, bạn cần làm theo các bước sau:

1. Mở ảnh bằng thư viện (ví dụ: `PIL.Image` hoặc `cv2`).
2. Chuyển ảnh sang **Chế độ xám** (Grayscale, 1 kênh màu).
3. **Resize** ảnh về kích thước chuẩn `28 × 28`.
4. Chuyển thành mảng NumPy và đưa giá trị về khoảng `[0, 1]`.
5. Gọi hàm `predict_single(model, image01, device)`.

## 6. Liên kết

- Muốn xem mô hình hay đoán sai ở đâu? ➡️ Xem [evaluation.md](evaluation.md).
- Muốn cải thiện độ chính xác của mô hình? ➡️ Xem mục **"Hướng cải thiện"** trong [training.md](training.md).
