# Hướng dẫn chi tiết từng file — Practice 1 (Fashion-MNIST)

> Tài liệu mô tả **chức năng, các hàm/lớp chính và file đọc/ghi** của từng thành phần
> trong dự án. Để hiểu bài thực hành tổng thể, xem
> [chapter_1_fashion_mnist.md](chapter_1_fashion_mnist.md).

---

## Mục lục

- [1. `src/config.py`](#1-srcconfigpy)
- [2. `src/data.py`](#2-srcdatapypy)
- [3. `src/model.py`](#3-srcmodelpy)
- [4. `src/trainer.py`](#4-srctrainerpy)
- [5. `src/evaluate.py`](#5-srcevaluatepy)
- [6. `src/utils.py`](#6-srcutilspy)
- [7. `src/__init__.py`](#7-src__init__py)
- [8. `main.py`](#8-mainpy)
- [9. Các notebook](#9-các-notebook)
- [10. Các thư mục dữ liệu & kết quả](#10-các-thư-mục-dữ-liệu--kết-quả)
- [11. File cấu hình dự án](#11-file-cấu-hình-dự-án)

---

## 1. `src/config.py`

**Vai trò:** "bộ não cấu hình" — nơi **duy nhất** chứa đường dẫn và siêu tham số.
Khi muốn đổi batch size, số epoch... chỉ cần sửa file này (hoặc ghi đè qua `main.py`).

**Các hằng số:**

| Tên | Giá trị | Ý nghĩa |
|---|---|---|
| `PROJECT_ROOT` | – | Thư mục gốc dự án (tự dò, không phụ thuộc máy) |
| `DATA_DIR` / `MODELS_DIR` | `data/`, `models/` | Nơi lưu CSV Kaggle và trọng số `.pth` |
| `FIGURES_DIR` / `TABLES_DIR` | `outputs/figures/`, `outputs/tables/` | Nơi lưu PNG và CSV kết quả |
| `BEST_MODEL_PATH` | `models/best_model.pth` | File checkpoint tốt nhất |
| `CLASS_NAMES` | list 10 tên | Tên lớp theo chỉ số 0–9 |
| `NUM_CLASSES` | 10 | Số lớp phân loại |
| `MEAN`, `STD` | 0.2860, 0.3530 | Tham số chuẩn hoá pixel |
| `SEED`, `BATCH_SIZE`, `NUM_EPOCHS`, `LEARNING_RATE`, `WEIGHT_DECAY` | 42, 64, 15, 1e-3, 1e-4 | Siêu tham số huấn luyện |
| `VAL_SPLIT`, `EARLY_STOP_PATIENCE`, `NUM_WORKERS` | 0.1, 5, 0 | Chia validation, dừng sớm, worker |

**Hàm:**

- `ensure_dirs()` — tạo `data/`, `models/`, `outputs/figures/`, `outputs/tables/` nếu thiếu.
  Được gọi đầu mọi quy trình ghi kết quả nên không bao giờ lỗi "thư mục không tồn tại".

**Đọc/ghi:** không đọc ghi file nào (trừ việc tạo thư mục).

---

## 2. `src/data.py`

**Vai trò:** toàn bộ mọi thứ liên quan dữ liệu — tự tải Kaggle, đọc CSV, chuẩn hoá,
Dataset, DataLoader, chia train/validation, augmentation.

**Luồng nạp dữ liệu (hàm `_ensure_csv_data`):**

```
tìm CSV trong data/  ──có──►  dùng CSV
        │ không
tải bằng kagglehub, chép CSV vào data/  ──thành công──►  dùng CSV
        │ thất bại (chưa cài / lỗi mạng)
torchvision.datasets.FashionMNIST tự tải (dự phòng)
```

**Các hàm / lớp:**

| Thành phần | Chức năng |
|---|---|
| `KAGGLE_DATASET` | `"zalando-research/fashionmnist"` — id dataset trên Kaggle |
| `TRAIN_CSV_NAMES` / `TEST_CSV_NAMES` | Các tên file CSV chấp nhận được khi dò tìm |
| `_find_csv(names)` | Quét đệ quy `data/` tìm file CSV theo tên |
| `_download_kagglehub()` | Tải dataset bằng `kagglehub`, chép 2 CSV về `data/`; trả `True/False` |
| `_ensure_csv_data()` | Gộp 2 hàm trên: đảm bảo có CSV bằng mọi cách |
| `FashionMNISTCSVDataset` | Dataset đọc từ DataFrame: cột 0 là nhãn, 784 cột pixel → tensor chuẩn hoá; `augment=True` thì mỗi lần lấy ảnh sẽ ngẫu nhiên lật ngang + xoay ±10° |
| `load_raw_arrays()` | Trả về `(X_train, y_train, X_test, y_test)` numpy uint8 **chưa chuẩn hoá** — dùng cho notebook khám phá dữ liệu và vẽ ảnh |
| `get_dataloaders(...)` | Hàm chính cho huấn luyện: trả về `(train_loader, val_loader, test_loader)` |

**Tham số `get_dataloaders(batch_size, val_split, use_augmentation, num_workers, seed)`:**

- Chia tập train gốc theo hoán vị seed → **tái lập được** mỗi lần chạy.
- `use_augmentation=True` chỉ ảnh hưởng tập train; val/test luôn "sạch".
- Batch val/test gấp đôi batch train (vì không cần tính gradient → tốn ít bộ nhớ hơn).

**Đọc/ghi:** đọc CSV trong `data/`; ghi CSV khi tải kagglehub lần đầu.

---

## 3. `src/model.py`

**Vai trò:** định nghĩa kiến trúc mạng CNN. Tách riêng để dễ thử nghiệm model mới
(chỉ cần thêm lớp rồi sửa 1 dòng import).

| Thành phần | Chức năng |
|---|---|
| `FashionCNN(num_classes=10)` | Mạng CNN: `features` (2 khối conv → kích thước 28→14→7) + `classifier` (MLP 3.136→256→10) |
| `build_model(num_classes)` | Hàm factory tạo model mới — nơi duy nhất các module khác gọi tới |

**Chi tiết kiến trúc** (bảng layer đầy đủ xem ở [chapter_1 §6.1](chapter_1_fashion_mnist.md#6-mô-hình-cnn)):

- Khối conv 1: `Conv(1→32) + BN + ReLU + Conv(32→32) + BN + ReLU + MaxPool + Dropout(0.25)`
- Khối conv 2: `Conv(32→64) + BN + ReLU + Conv(64→64) + BN + ReLU + MaxPool + Dropout(0.25)`
- Đầu ra: `Flatten → Linear(3136→256) + BN + ReLU + Dropout(0.5) → Linear(256→10)`

`forward(x)` nhận tensor `(B, 1, 28, 28)` đã chuẩn hoá, trả **logits** `(B, 10)`
(softmax nằm trong CrossEntropyLoss).

**Đọc/ghi:** không.

---

## 4. `src/trainer.py`

**Vai trò:** quy trình huấn luyện đóng gói trong lớp `Trainer` + tiện ích checkpoint.

| Thành phần | Chức năng |
|---|---|
| `Trainer.__init__(model, criterion, optimizer, scheduler, device)` | Nếu không truyền loss/optimizer thì dùng mặc định CrossEntropy + AdamW theo cấu hình |
| `Trainer.fit(train_loader, val_loader, epochs, patience, save_path)` | Huấn luyện tối đa `epochs` epoch; mỗi epoch: train → validate → scheduler.step → so val_loss với kỷ lục để lưu checkpoint; dừng sớm khi hết patience. Trả về `history` dict |
| `Trainer.train_one_epoch(loader)` | Một lượt forward-backward-update trên tập train, có thanh tqdm; trả (loss, acc) |
| `Trainer.validate(loader)` | Đánh giá không gradient trên validation; trả (loss, acc) |
| `save_checkpoint(path, model, epoch, val_loss, val_acc)` | Lưu state_dict + metadata (epoch, val_loss, val_acc, tên lớp) |
| `load_trained_model(path, device, num_classes)` | Nạp checkpoint vào model mới, chuyển eval mode, in thông tin; trả `(model, checkpoint)` |

**`history` trả về gồm 4 khoá** (mỗi khoá là list số thực theo epoch):
`train_loss`, `train_acc`, `val_loss`, `val_acc` — truyền thẳng vào
`utils.plot_training_history()` và lưu CSV được.

**Đọc/ghi:** ghi `models/best_model.pth` (đọc lại ở notebook 03 / `main.py`).

---

## 5. `src/evaluate.py`

**Vai trò:** mọi thứ đánh giá sau huấn luyện — dùng ở notebook 03 và bước 4 của `main.py`.

| Thành phần | Chức năng |
|---|---|
| `collect_predictions(model, loader, device)` | Chạy model trên cả loader (không gradient), trả `(y_true, y_pred)` numpy |
| `evaluate_classification(y_true, y_pred)` | Tính accuracy, in classification report, trả dict `{"accuracy", "report", "report_text", "confusion_matrix"}` |
| `save_evaluation_results(results, prefix)` | Lưu 3 file: bảng P/R/F1 từng lớp (`classification_report_{prefix}.csv`), ma trận nhầm lẫn (`confusion_matrix_{prefix}.csv`), hình heatmap (`confusion_matrix_{prefix}.png`) |
| `get_misclassified(model, loader, device, max_samples)` | Gom tối đa `max_samples` ảnh dự đoán sai kèm nhãn thật/dự đoán — để vẽ lưới ảnh lỗi |
| `predict_single(model, image, device, class_names)` | Dự đoán 1 ảnh (numpy/tensor, `[0,1]`, tự chuẩn hoá bên trong); trả `(idx, tên_lớp, probs_10_lớp)` |

**Đọc/ghi:** ghi vào `outputs/tables/` và `outputs/figures/`.

---

## 6. `src/utils.py`

**Vai trò:** tiện ích dùng mọi nơi — seed, thiết bị, vẽ biểu đồ, lưu kết quả.
Tất cả hàm vẽ **trả về figure** (hiển thị ngay trong notebook) và **tự lưu PNG**
nếu truyền `save_name`.

| Thành phần | Chức năng |
|---|---|
| `set_seed(seed)` | Cố định random/np/torch seed — chạy lại cho đúng kết quả |
| `get_device()` | `cuda` nếu có GPU, ngược lại `cpu` |
| `count_parameters(model)` | Tổng tham số có thể học |
| `save_figure(fig, name)` | Lưu PNG vào `outputs/figures/`, in đường dẫn |
| `save_table(df, name)` | Lưu CSV vào `outputs/tables/` (utf-8-sig mở Excel không lỗi dấu) |
| `plot_class_distribution(y)` | Biểu đồ cột số ảnh mỗi lớp |
| `plot_image_grid(images, titles, ncols)` | Lưới ảnh xám (nhận numpy hoặc tensor) |
| `plot_training_history(history)` | 2 đồ thị cạnh nhau: Loss, Accuracy theo epoch |
| `plot_confusion_matrix(cm, normalize)` | Heatmap ma trận nhầm lẫn; `normalize=True` chuẩn hoá theo dòng |

---

## 7. `src/__init__.py`

Khai báo `src` là package Python kèm docstring liệt kê các module và `__version__`.
Không chứa logic.

---

## 8. `main.py`

**Vai trò:** chạy **toàn bộ pipeline bằng một lệnh** — ghép 3 notebook thành 1 script
tự động (hữu ích khi chạy đêm, chạy trên server, chạy nhiều cấu hình).

**Các bước trong `main(args)`:**

1. Set seed, tạo thư mục, chọn thiết bị.
2. `data.get_dataloaders(...)` theo tham số dòng lệnh.
3. Dựng `FashionCNN`, in số tham số.
4. Huấn luyện bằng `Trainer.fit()` → lưu `best_model.pth`, `training_history.png/.csv`.
5. Nạp model tốt nhất → đánh giá tập test → lưu report + ma trận nhầm lẫn.

**Tham số dòng lệnh:**

| Cờ | Mặc định | Ý nghĩa |
|---|---|---|
| `--epochs` | 15 | Số epoch tối đa |
| `--batch-size` | 64 | Kích thước batch |
| `--lr` | 1e-3 | Learning rate |
| `--val-split` | 0.1 | Tỉ lệ tập validation |
| `--seed` | 42 | Seed tái lập |
| `--no-augment` | tắt | Cờ: tắt augmentation |

```bash
python main.py --epochs 20 --batch-size 128
```

---

## 9. Các notebook

Chạy **đúng thứ tự 01 → 02 → 03** (03 cần file model của 02). Mỗi notebook đều có
cell đầu thêm thư mục gốc vào `sys.path` để `from src import ...` hoạt động.

### `notebooks/01_data_exploration.ipynb` — Khám phá dữ liệu

| Cell | Nội dung | Kết quả |
|:---:|---|---|
| 0 | Import, set seed | – |
| 1 | `data.load_raw_arrays()` (tự tải Kaggle lần đầu) | In shape 60k/10k |
| 2 | Đếm + vẽ phân bố 10 lớp | `01_class_distribution.png` |
| 3 | Lưới 16 ảnh mẫu ngẫu nhiên | `01_sample_images.png` |
| 4 | Thống kê pixel + ảnh trung bình mỗi lớp | `01_mean_images.png` |
| 5 | Markdown tổng kết nhận xét | – |

### `notebooks/02_model_training.ipynb` — Huấn luyện

| Cell | Nội dung | Kết quả |
|:---:|---|---|
| 0 | Import, device | – |
| 1 | `data.get_dataloaders()` ( augmentation bật) | In số ảnh/batch |
| 2 | `build_model()`, in kiến trúc + số tham số | – |
| 3 | CrossEntropy + AdamW + ReduceLROnPlateau → `Trainer.fit()` | `models/best_model.pth` |
| 4 | Vẽ đường cong học tập + lưu CSV | `02_training_history.png/.csv` |
| 5 | Markdown ghi chú kết quả (điền sau khi chạy) | – |

### `notebooks/03_evaluation_and_inference.ipynb` — Đánh giá & suy luận

| Cell | Nội dung | Kết quả |
|:---:|---|---|
| 0 | Import, device | – |
| 1 | `load_trained_model()` | In epoch/val_loss tốt nhất |
| 2 | Dự đoán tập test + classification report | In accuracy, P/R/F1 |
| 3 | Ma trận nhầm lẫn (thô + chuẩn hoá) + lưu | `03_confusion_matrix_test.png`, 2 bảng CSV |
| 4 | Lưới 16 ảnh dự đoán sai | `03_misclassified.png` |
| 5 | Dự đoán 1 ảnh + biểu đồ xác suất | `03_prediction_example.png` |
| 6 | Markdown tổng kết | – |

---

## 10. Các thư mục dữ liệu & kết quả

| Thư mục | Ai ghi | Nội dung |
|---|---|---|
| `data/` | `src/data.py` (kagglehub/torchvision) | 2 CSV Kaggle: `fashion-mnist_train.csv`, `fashion-mnist_test.csv` |
| `models/` | `Trainer.fit()` | `best_model.pth` — checkpoint tốt nhất + metadata |
| `outputs/figures/` | `utils.save_figure()` | Toàn bộ PNG (xem bảng §9.4 trong chapter_1) |
| `outputs/tables/` | `utils.save_table()` | CSV: history, classification report, confusion matrix |

Các thư mục đều có file `.gitkeep` mô tả — để git giữ cấu trúc khi nội dung bị ignore.

---

## 11. File cấu hình dự án

| File | Vai trò |
|---|---|
| `requirements.txt` | Danh sách thư viện: `pip install -r requirements.txt`. Ghi chú từng nhóm (deep learning, vẽ biểu đồ, Kaggle...). |
| `.gitignore` | Bỏ qua `data/`, `models/`, `outputs/`, `__pycache__/`, `.ipynb_checkpoints/`... khi dùng git. |
| `README.md` | Giới thiệu tổng quan + cài đặt + cách chạy (bản rút gọn của 2 file docs này). |
| `docs/chapter_1_fashion_mnist.md` | Bài làm Chương 1 chi tiết: lý thuyết, công thức, bảng chỉ số, hướng dẫn từng bước. |
| `docs/file_guide.md` | File này — tra cứu nhanh từng hàm, từng file. |

---

*Sau khi đổi/mở rộng code, hãy cập nhật file này để tài liệu luôn khớp với mã nguồn.*
