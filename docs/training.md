# 🏋️ Huấn luyện — 30 Epochs & Early Stopping (Patience = 5)

> 🔗 **Liên kết:** [README](../README.md) · [data.md](data.md) · [model.md](model.md) · [evaluation.md](evaluation.md) · [demo.md](demo.md)

## 1. Cấu hình mới
Để mô hình đạt kết quả tốt nhất và tránh overfitting, dự án đã được cập nhật lên cấu hình mở rộng:
- **Số epoch tối đa:** `30` (`config.NUM_EPOCHS_EXTENDED`)
- **Early stopping patience:** `5` (`config.EARLY_STOP_PATIENCE_EXTENDED`)
- **Vẽ đường cong học tập:** Vẽ **mỗi epoch** (`config.PLOT_EPOCH_STEP = 1`).

## 2. Cách chạy huấn luyện
Bạn có thể huấn luyện bằng 1 trong 2 cách sau:

**Cách 1: Chạy qua file CLI `main.py` (Khuyến nghị)**
```bash
# Chạy với cấu hình mặc định (30 epochs, early stop 5)
python main.py

# Hoặc chỉ định rõ các tham số
python main.py --epochs 30 --patience 5 --batch-size 128 --lr 1e-3
## Cách 2: Chạy qua Notebook `02_model_training.ipynb`

Di chuyển đến cell huấn luyện và chạy:

```python
trainer = Trainer(model, criterion, optimizer, scheduler, device)

history = trainer.fit(
    train_loader,
    val_loader,
    epochs=config.NUM_EPOCHS_EXTENDED,  # 30
    patience=config.EARLY_STOP_PATIENCE_EXTENDED,  # 5
    save_path=config.BEST_MODEL_PATH,
)
```

## 3. Siêu tham số (Hyperparameters)

| **Tham số**      | **Giá trị**       | **Ghi chú**                                           |
| :--------------- | :---------------- | :---------------------------------------------------- |
| Batch size       | 64                | Kích thước mỗi lô dữ liệu                             |
| Learning Rate    | 1e-3              | Tốc độ học ban đầu của AdamW                          |
| Weight Decay     | 1e-4              | Hệ số phạt L2 (chống overfitting)                     |
| Validation Split | 10%               | Trích 6.000 ảnh từ tập train để kiểm thử              |
| Scheduler        | ReduceLROnPlateau | Giảm LR đi 1 nửa nếu val_loss không giảm sau 2 epochs |

## 4. Cơ chế Early Stopping & Checkpoint

Quy trình hoạt động của `Trainer` (`src/trainer.py`) qua mỗi epoch:

1. **Train:** Cập nhật trọng số và tính `train_loss`, `train_acc`.
2. **Validation:** Đánh giá trên tập val (không tính gradient), lấy `val_loss`, `val_acc`.
3. **Scheduler:** Gọi `scheduler.step(val_loss)` để tự động điều chỉnh Learning Rate.
4. **Lưu Checkpoint:** Nếu `val_loss` giảm xuống mức thấp nhất (tốt nhất từ trước tới giờ) → Lưu đè file `models/best_model.pth`.
5. **Early Stopping:** Nếu sau **5 epochs** liên tiếp mà `val_loss` không cải thiện → Dừng huấn luyện sớm để tiết kiệm thời gian và tránh overfitting.

## 5. Các file kết quả đầu ra

Sau khi quá trình huấn luyện kết thúc, các file sau sẽ được tạo tự động:

- `models/best_model.pth` — Trọng số của mô hình có `val_loss` tốt nhất.
- `outputs/figures/training_history.png` — Biểu đồ thể hiện Loss và Accuracy qua từng epoch.
- `outputs/tables/training_history.csv` — Bảng số liệu chi tiết của từng epoch.

## 6. Cách đọc biểu đồ đường cong học tập

- **Trường hợp lý tưởng:** Cả `train_loss` và `val_loss` đều giảm dần qua các epoch.
- **Overfitting:** `train_loss` tiếp tục giảm nhưng `val_loss` lại bắt đầu tăng lên. (Đã được hạn chế nhờ Dropout, Weight Decay và Augmentation).
- Nếu biểu đồ đi ngang từ epoch 15-20 trở đi, chứng tỏ mô hình đã hội tụ.
