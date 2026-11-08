# 📈 Đánh giá — Tập Test & Phân tích lỗi

> 🔗 **Liên kết:** [README](../README.md) · [data.md](data.md) · [model.md](model.md) · [training.md](training.md) · [demo.md](demo.md)

## 1. Nạp mô hình tốt nhất
Sau khi quá trình huấn luyện (30 epochs + Early stopping) hoàn tất, trọng số tối ưu nhất đã được lưu tại `models/best_model.pth`. 
Hàm `load_trained_model()` trong `src/trainer.py` sẽ giúp khôi phục lại mô hình kèm theo metadata:

```python
from src.trainer import load_trained_model
from src import config, utils

device = utils.get_device()
model, checkpoint = load_trained_model(config.BEST_MODEL_PATH, device=device)

print("Epoch tốt nhất:", checkpoint["epoch"])
print("Val Loss      :", checkpoint["val_loss"])
print("Val Accuracy  :", checkpoint["val_acc"])
## 2. Các chỉ số đánh giá (`src/evaluate.py`)

Hàm `evaluate_classification(y_true, y_pred)` sẽ chạy dự đoán trên **10.000 ảnh của tập Test** và trả về:

- **Accuracy** (Độ chính xác tổng thể): Chỉ số chính do dữ liệu Fashion-MNIST cân bằng tuyệt đối.
- **Precision / Recall / F1-Score**: Điểm số chi tiết của từng lớp (Macro & Weighted).
- **Confusion Matrix** (Ma trận nhầm lẫn): Mảng 10×10 thể hiện số lượng ảnh dự đoán đúng/sai.

Hàm `save_evaluation_results(results, prefix="test")` sẽ tự động lưu các kết quả này ra file:

- `outputs/tables/test_metrics.csv` (Bảng điểm tổng hợp)
- `outputs/tables/test_classification_report.csv` (Báo cáo chi tiết từng lớp)
- `outputs/tables/test_confusion_matrix.csv` (Số liệu ma trận nhầm lẫn)
- `outputs/figures/test_confusion_matrix.png` (Hình ảnh ma trận)

## 3. Ma trận nhầm lẫn (Confusion Matrix)

Trong Notebook `03_evaluation_and_inference.ipynb`, ta vẽ 2 phiên bản ma trận nhầm lẫn:

```python
# 1. Bản thô (số lượng ảnh)
fig = utils.plot_confusion_matrix(
    results["confusion_matrix"],
    save_name="03_confusion_matrix_test"
)

# 2. Bản chuẩn hoá theo dòng (tỉ lệ %)
fig = utils.plot_confusion_matrix(
    results["confusion_matrix"],
    normalize=True
)
```

**Cách đọc:** Bản chuẩn hoá theo dòng cho biết **tỉ lệ %** ảnh của một lớp thật bị mô hình dự đoán nhầm sang lớp khác (tổng mỗi dòng = 100%).

## 4. Kết quả kỳ vọng & Phân tích

- **Accuracy:** Thường đạt mức **92% – 93%** với kiến trúc `FashionCNN` hiện tại.
- **Điểm yếu của mô hình:** Lớp **Shirt** (Áo sơ mi) thường có F1-score thấp nhất. Mô hình rất dễ nhầm lẫn nó với *T-shirt/top* (Áo thun), *Pullover* (Áo len chui đầu) và *Coat* (Áo khoác). Nguyên nhân do hình dáng và các nếp gấp của các loại áo này trên ảnh xám 28×28 gần như tương đồng.

## 5. Phân tích các ảnh bị phân loại sai

Để hiểu rõ hơn điểm yếu của mô hình, ta trích xuất 16 ảnh dự đoán sai ngẫu nhiên:

```python
images, trues, preds = evaluate.get_misclassified(
    model,
    test_loader,
    device,
    max_samples=16
)
```

Hình ảnh xuất ra (lưu tại `outputs/figures/03_misclassified.png`) sẽ có tiêu đề:

- `T:` (Truth) - Nhãn thật
- `P:` (Prediction) - Nhãn mô hình dự đoán

## 6. Hướng cải thiện trong tương lai

Nếu muốn nâng Accuracy lên mức cao hơn (95%+), có thể thử các phương pháp:

- Sử dụng Data Augmentation mạnh hơn (Crop, ColorJitter, Cutout).
- Thay đổi Scheduler (Cosine Annealing) hoặc dùng Warmup.
- Sử dụng kiến trúc sâu hơn như ResNet (cần resize ảnh lên 224×224).
