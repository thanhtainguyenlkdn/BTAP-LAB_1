# CHƯƠNG 1 — PHÂN LOẠI ẢNH FASHION-MNIST BẰNG MẠNG CNN

> Bài thực hành số 1 — Xây dựng pipeline học sâu end-to-end bằng PyTorch
> Dataset: [Fashion-MNIST (Kaggle — zalando-research)](https://www.kaggle.com/datasets/zalando-research/fashionmnist)

---

## Mục lục

1. [Giới thiệu chung](#1-giới-thiệu-chung)
2. [Dữ liệu Fashion-MNIST](#2-dữ-liệu-fashion-mnist)
3. [Tải dữ liệu từ Kaggle](#3-tải-dữ-liệu-từ-kaggle)
4. [Kiến trúc dự án](#4-kiến-trúc-dự-án)
5. [Tiền xử lý dữ liệu](#5-tiền-xử-lý-dữ-liệu)
6. [Mô hình CNN](#6-mô-hình-cnn)
7. [Huấn luyện mô hình](#7-huấn-luyện-mô-hình)
8. [Đánh giá mô hình](#8-đánh-giá-mô-hình)
9. [Hướng dẫn chạy từng bước](#9-hướng-dẫn-chạy-từng-bước)
10. [Kết quả mong đợi và nhận xét](#10-kết-quả-mong-đợi-và-nhận-xét)

---

## 1. Giới thiệu chung

### 1.1. Bối cảnh và mục tiêu

Phân loại ảnh là bài toán nền tảng của thị giác máy tính. Chương này xây dựng
một **pipeline hoàn chỉnh từ đầu đến cuối** trên bộ dữ liệu Fashion-MNIST để nắm:

- Cách tổ chức một dự án machine learning chuẩn (code tách module, dữ liệu/model/kết
  quả lưu riêng từng thư mục).
- Cách nạp và tự động tải dữ liệu công khai từ **Kaggle**.
- Cách tự cài đặt một mạng **CNN (Convolutional Neural Network)** bằng PyTorch.
- Quy trình huấn luyện: loss, optimizer, scheduler, early stopping, checkpoint.
- Quy trình đánh giá: accuracy, precision/recall/F1, ma trận nhầm lẫn, phân tích lỗi.

### 1.2. Công nghệ sử dụng

| Thành phần | Thư viện | Vai trò |
|---|---|---|
| Deep learning | `torch`, `torchvision` | Mô hình CNN, Dataset, augmentation |
| Dữ liệu | `pandas`, `numpy` | Đọc CSV Kaggle, xử lý mảng |
| Đánh giá | `scikit-learn` | Accuracy, classification report, confusion matrix |
| Trực quan hoá | `matplotlib`, `seaborn` | Biểu đồ, heatmap |
| Tải dữ liệu | `kagglehub` | Tự động tải dataset từ Kaggle |
| Tiến trình | `tqdm` | Thanh tiến trình khi huấn luyện/dự đoán |

---

## 2. Dữ liệu Fashion-MNIST

### 2.1. Giới thiệu

Fashion-MNIST do Zalando Research công bố (2017) làm **bộ chuẩn thay thế MNIST chữ số**:
cùng định dạng 28×28 xám, cùng số lượng ảnh, nhưng nhiệm vụ khó hơn và sát ứng dụng thực tế
hơn (nhận diện mặt hàng thời trang). Đây là dataset kinh điển để luyện CNN cơ bản.

### 2.2. Bảng 10 lớp

| Nhãn | Tên lớp | Diễn giải |
|:---:|---|---|
| 0 | T-shirt/top | Áo thun |
| 1 | Trouser | Quần dài |
| 2 | Pullover | Áo len không mũ |
| 3 | Dress | Váy |
| 4 | Coat | Áo khoác |
| 5 | Sandal | Dép xăng-đan |
| 6 | Shirt | Áo sơ mi |
| 7 | Sneaker | Giày thể thao |
| 8 | Bag | Túi xách |
| 9 | Ankle boot | Bốt cổ ngắn |

### 2.3. Thống kê tóm tắt

| Thuộc tính | Giá trị |
|---|---|
| Tổng số ảnh | 70.000 (60.000 train + 10.000 test) |
| Kích thước ảnh | 28×28 pixel, 1 kênh (xám) |
| Giá trị pixel | 0–255 |
| Số lớp | 10, **cân bằng tuyệt đối** (6.000 ảnh/lớp ở train, 1.000 ảnh/lớp ở test) |
| Định dạng Kaggle | 2 file CSV: cột `label` + 784 cột pixel |

### 2.4. Cấu trúc file CSV của Kaggle

Mỗi dòng là 1 ảnh; cột đầu là nhãn, 784 cột còn lại là pixel bị duỗi phẳng
(theo thứ tự hàng, từ trái sang phải, trên xuống dưới):

```
label, pixel1, pixel2, ..., pixel784
9,     0,      0,      ..., 23
0,     1,      0,      ..., 56
```

Muốn vẽ lại ảnh: `image = row[1:].reshape(28, 28)`.

---

## 3. Tải dữ liệu từ Kaggle

> Nguồn: <https://www.kaggle.com/datasets/zalando-research/fashionmnist>

### 3.1. Cách 1 — Tự động bằng kagglehub (dự án dùng cách này)

Code trong `src/data.py` tự thực hiện khi chạy lần đầu:

1. Tìm `fashion-mnist_train.csv` / `fashion-mnist_test.csv` trong `data/`.
2. Nếu chưa có → gọi `kagglehub.dataset_download("zalando-research/fashionmnist")`.
3. Sao chép 2 file CSV về `data/` → các lần chạy sau chỉ đọc local, không tải lại.

```python
import kagglehub
path = kagglehub.dataset_download("zalando-research/fashionmnist")
```

### 3.2. Cách 2 — Kaggle CLI

```bash
pip install kaggle
# Lấy token: kaggle.com -> Settings -> API -> Create New Token -> tải kaggle.json
# Đặt file vào: C:\Users\<tên_bạn>\.kaggle\kaggle.json
kaggle datasets download -d zalando-research/fashionmnist -p data --unzip
```

### 3.3. Cách 3 — Tải tay trên web

Vào trang dataset → bấm **Download** → giải nén 2 file CSV vào thư mục `data/`.

### 3.4. Dự phòng khi không dùng được Kaggle

`src/data.py` có lớp dự phòng: nếu không có CSV và kagglehub thất bại, pipeline
tự tải Fashion-MNIST qua `torchvision.datasets.FashionMNIST`. Kết quả huấn luyện
tương đương vì đây là cùng một bộ dữ liệu.

> 💡 **Chạy trên Kaggle Notebooks (tuỳ chọn):** tạo notebook mới trên kaggle.com,
> bấm *Add Input* → chọn dataset *Fashion-MNIST*, rồi chạy các cell của dự án.
> Lưu ý phải nén/clone toàn bộ mã nguồn `src/` lên hoặc gộp code vào một file.

---

## 4. Kiến trúc dự án

### 4.1. Sơ đồ thư mục

```
Practice_1/
├── data/                  # CSV Kaggle (tự tải) — KHÔNG đưa lên git
├── models/                # best_model.pth sau huấn luyện
├── outputs/
│   ├── figures/           # PNG: biểu đồ, ảnh dự đoán
│   └── tables/            # CSV: báo cáo đánh giá, history
├── src/                   # Logic chính (7 file Python)
├── notebooks/             # 3 notebook chạy theo trình tự
├── docs/                  # Tài liệu (file này + file_guide.md)
├── main.py                # Pipeline 1 lệnh
└── requirements.txt
```

### 4.2. Luồng xử lý (pipeline)

```
CSV Kaggle (data/)
        │  src/data.py — đọc, /255, chuẩn hoá, chia train/val, augmentation
        ▼
DataLoader (train 54.000 / val 6.000 / test 10.000)
        │  src/model.py — FashionCNN
        ▼
Huấn luyện (src/trainer.py)
        │  CrossEntropy + AdamW + ReduceLROnPlateau + early stopping
        ├──► models/best_model.pth   (checkpoint tốt nhất theo val_loss)
        ▼
Đánh giá (src/evaluate.py)
        ├──► outputs/figures/*.png   (ma trận nhầm lẫn, ảnh sai, ...)
        └──► outputs/tables/*.csv    (classification report, ...)
```

---

## 5. Tiền xử lý dữ liệu

### 5.1. Chuẩn hoá pixel

```
x_chuẩn_hoá = (x/255 − 0.2860) / 0.3530
```

- `0.2860` / `0.3530` là mean/std của toàn bộ ảnh Fashion-MNIST (hằng số công bố
  sẵn, định nghĩa trong `src/config.py`).
- Mục đích: đưa phân phối input về quanh 0, đơn vị gần 1 → mạng hội tụ nhanh và ổn định hơn.

### 5.2. Chia tập dữ liệu

| Tập | Số ảnh | Nguồn | Augmentation |
|---|---|---|---|
| Train | 54.000 | 90% tập train gốc | ✅ Có |
| Validation | 6.000 | 10% tập train gốc | ❌ Không |
| Test | 10.000 | tập test gốc | ❌ Không |

- **Validation** dùng để chọn model tốt nhất và điều chỉnh learning rate trong lúc huấn luyện.
- **Test** chỉ chạm ở bước cuối để đánh giá khách quan.
- Chia theo `seed=42` (numpy RandomState) nên mọi lần chạy cho cùng một kết quả chia.

### 5.3. Data augmentation (chỉ cho tập train)

- **Lật ngang ngẫu nhiên** (p = 0.5): quần áo lật trái–phải vẫn hợp lệ.
- **Xoay nhẹ ±10°**: mô phỏng ảnh chụp nghiêng nhẹ.

Augmentation làm dữ liệu "khó học thuộc" hơn → giảm overfitting, tăng khả năng khái quát.

---

## 6. Mô hình CNN

### 6.1. Kiến trúc chi tiết (FashionCNN — `src/model.py`)

| # | Lớp | Kích thước vào → ra | Tham số |
|:---:|---|---|---:|
| 1 | Conv2d(1→32, 3×3, pad 1) + BN + ReLU | (1, 28, 28) → (32, 28, 28) | 320 |
| 2 | Conv2d(32→32, 3×3, pad 1) + BN + ReLU | (32, 28, 28) → (32, 28, 28) | 9.248 |
| 3 | MaxPool 2×2 + Dropout2d(0.25) | (32, 28, 28) → (32, 14, 14) | 0 |
| 4 | Conv2d(32→64, 3×3, pad 1) + BN + ReLU | (32, 14, 14) → (64, 14, 14) | 18.496 |
| 5 | Conv2d(64→64, 3×3, pad 1) + BN + ReLU | (64, 14, 14) → (64, 14, 14) | 36.928 |
| 6 | MaxPool 2×2 + Dropout2d(0.25) | (64, 14, 14) → (64, 7, 7) | 0 |
| 7 | Flatten | (64, 7, 7) → 3.136 | 0 |
| 8 | Linear(3.136→256) + BN + ReLU + Dropout(0.5) | 3.136 → 256 | 803.072 |
| 9 | Linear(256→10) | 256 → 10 (logits) | 2.570 |

**Tổng: ≈ 0.87 triệu tham số.**

### 6.2. Vai trò từng thành phần

- **Conv2d 3×3:** quét ảnh bằng bộ lọc học được → nhận biết đặc trưng cục bộ
  (đường nét, góc, viền áo). Hai lớp Conv xếp chồng tạo đặc trưng từ đơn giản đến phức tạp.
- **BatchNorm:** chuẩn hoá theo batch giúp loss mượt, cho phép dùng learning rate lớn hơn.
- **ReLU:** hàm kích hoạt phi tuyến `max(0, x)` — điều kiện để mạng học được quan hệ phi tuyến.
- **MaxPool 2×2:** giảm một nửa kích thước không gian mỗi chiều → giảm tính toán,
  đồng thời tạo bất biến với dịch chuyển nhỏ của đặc trưng.
- **Dropout (0.25/0.5):** ngẫu nhiên tắt neuron khi huấn luyện → chống overfitting.
- **Linear cuối 10 node:** xuất logits cho 10 lớp (softamx nằm trong CrossEntropyLoss).

---

## 7. Huấn luyện mô hình

### 7.1. Hàm loss và optimizer

- **Loss:** `nn.CrossEntropyLoss` — chuẩn cho bài toán phân loại đa lớp
  (gồm sẵn softmax + negative log-likelihood).
- **Optimizer:** `AdamW` với `lr=1e-3`, `weight_decay=1e-4`
  (Adam có weight decay tách riêng — regularization tốt hơn Adam gốc).

### 7.2. Scheduler và early stopping

- **ReduceLROnPlateau (mode=min, factor=0.5, patience=2):** nếu val_loss không giảm
  sau 2 epoch thì giảm một nửa learning rate → tinh chỉnh bước nhở hơn khi gần hội tụ.
- **Early stopping (patience=5):** nếu val_loss không lập kỷ lục mới sau 5 epoch liền
  thì dừng huấn luyện — tiết kiệm thời gian và tránh overfitting thêm.

### 7.3. Checkpoint

Sau mỗi epoch, nếu val_loss tốt nhất từ trước tới nay → lưu `models/best_model.pth`:

```python
{
  "model_state": ...,   # state_dict của CNN
  "epoch": 12,          # epoch tốt nhất
  "val_loss": 0.2134,
  "val_acc": 0.9285,
  "class_names": [...], # 10 tên lớp
}
```

### 7.4. Siêu tham số mặc định (`src/config.py`)

| Tham số | Giá trị | Ý nghĩa |
|---|---|---|
| `SEED` | 42 | Tái lập kết quả |
| `BATCH_SIZE` | 64 | Số ảnh mỗi bước cập nhật |
| `NUM_EPOCHS` | 15 | Số epoch tối đa (có thể dừng sớm) |
| `LEARNING_RATE` | 1e-3 | Learning rate AdamW |
| `WEIGHT_DECAY` | 1e-4 | Regularization L2 |
| `VAL_SPLIT` | 0.1 | 10% train làm validation |
| `EARLY_STOP_PATIENCE` | 5 | Ngưỡng dừng sớm |

⏱️ **Thời gian chạy ước tính:** GPU ≈ 15–30 giây/epoch; CPU ≈ 2–5 phút/epoch.
Nếu chỉ có CPU, có thể giảm `NUM_EPOCHS` xuống 5–8 vẫn đạt accuracy ~90–92%.

---

## 8. Đánh giá mô hình

### 8.1. Các chỉ số

- **Accuracy:** tỉ lệ dự đoán đúng toàn bộ — chỉ số chính (dữ liệu cân bằng lớp).
- **Precision (lớp i):** trong số ảnh *được dự đoán* là lớp i, bao nhiêu phần trăm đúng.
- **Recall (lớp i):** trong số ảnh *thực sự* là lớp i, mô hình bắt được bao nhiêu phần trăm.
- **F1-score:** trung bình điều hoà của Precision và Recall.
- **Macro avg / Weighted avg:** F1 trung bình không cân bằng / có trọng số theo số ảnh.

Công thức:

```
Accuracy   = (TP + TN) / Tổng
Precision  = TP / (TP + FP)
Recall     = TP / (TP + FN)
F1         = 2·Precision·Recall / (Precision + Recall)
```

### 8.2. Đọc ma trận nhầm lẫn

- Hàng = nhãn thật, cột = nhãn dự đoán; đường chéo là số ảnh đúng.
- Ô (i, j) sáng ngoài đường chéo → lớp i hay bị nhầm thành lớp j.
- Bản **chuẩn hoá theo dòng** (mỗi hàng tổng = 1) cho biết *tỉ lệ* nhầm lẫn của từng lớp.

### 8.3. Kết quả minh hoạ sau khi chạy

`outputs/tables/classification_report_test.csv` (số liệu minh hoạ, kết quả thực
tùy máy/epoch):

```
Accuracy tập test: ~92.5%

class         precision  recall  f1-score  support
T-shirt/top      0.87     0.88     0.87      1000
Trouser          0.99     0.98     0.99      1000
...
Shirt            0.76     0.79     0.77      1000   <-- thấp nhất
...
```

---

## 9. Hướng dẫn chạy từng bước

### 9.1. Cài môi trường (một lần)

```bash
cd Practice_1
python -m venv .venv
source .venv/Scripts/activate        # Git Bash trên Windows
pip install -r requirements.txt
```

### 9.2. Chạy theo notebook (khuyến nghị khi học)

| Thứ tự | Notebook | Việc làm chính | Sản phẩm |
|:---:|---|---|---|
| 1 | `01_data_exploration.ipynb` | Tự tải dữ liệu Kaggle; xem phân bố lớp, ảnh mẫu, ảnh trung bình | `outputs/figures/01_*.png` |
| 2 | `02_model_training.ipynb` | Tạo DataLoader, huấn luyện CNN, vẽ đường cong học tập | `models/best_model.pth`, `outputs/figures/02_*.png`, `outputs/tables/02_training_history.csv` |
| 3 | `03_evaluation_and_inference.ipynb` | Đánh giá tập test, ma trận nhầm lẫn, ảnh sai, suy luận 1 ảnh | `outputs/figures/03_*.png`, `outputs/tables/*_test.csv` |

Chạy từng notebook theo đúng thứ tự (notebook 3 cần file model của notebook 2).

### 9.3. Chạy bằng một lệnh

```bash
python main.py                       # mặc định: 15 epoch, batch 64
python main.py --epochs 20 --batch-size 128
python main.py --no-augment --seed 0
```

`main.py` thực hiện tự động cả 4 bước: dữ liệu → dựng model → huấn luyện → đánh giá,
và lưu toàn bộ kết quả vào `models/` + `outputs/`.

### 9.4. Danh sách file kết quả

| File | Sinh ra từ | Nội dung |
|---|---|---|
| `models/best_model.pth` | `Trainer.fit()` | Trọng số tốt nhất + metadata |
| `outputs/figures/01_class_distribution.png` | Notebook 01 | Biểu đồ cột số ảnh mỗi lớp |
| `outputs/figures/01_sample_images.png` | Notebook 01 | Lưới 16 ảnh mẫu |
| `outputs/figures/01_mean_images.png` | Notebook 01 | Ảnh trung bình mỗi lớp |
| `outputs/figures/02_training_history.png` | Notebook 02 / main.py | Đồ thị loss & accuracy |
| `outputs/tables/02_training_history.csv` | Notebook 02 / main.py | Số liệu từng epoch |
| `outputs/figures/03_confusion_matrix_test.png` | Notebook 03 / main.py | Heatmap ma trận nhầm lẫn |
| `outputs/tables/classification_report_test.csv` | Notebook 03 / main.py | P/R/F1 từng lớp |
| `outputs/tables/confusion_matrix_test.csv` | Notebook 03 / main.py | Ma trận nhầm lẫn số thô |
| `outputs/figures/03_misclassified.png` | Notebook 03 | Lưới ảnh dự đoán sai |
| `outputs/figures/03_prediction_example.png` | Notebook 03 | Xác suất 10 lớp cho 1 ảnh |

---

## 10. Kết quả mong đợi và nhận xét

### 10.1. Kết quả

| Chỉ số | Giá trị tham khảo |
|---|---|
| Accuracy validation | ~93% |
| Accuracy test | **~92–93%** |
| F1 thấp nhất | lớp **Shirt** (~0.75–0.80) |
| F1 cao nhất | Trouser, Bag, Ankle boot (~0.97–0.99) |

### 10.2. Nhận xét

- Các cặp lớp hay nhầm lẫn nhất: **Shirt ↔ T-shirt/top, Shirt ↔ Pullover, Shirt ↔ Coat,
  Pullover ↔ Coat** — nhóm áo có hình dạng gần giống nhau trong ảnh xám 28×28.
- Ngược lại, Trouser / Bag / footwear có hình dạng đặc trưng nên F1 rất cao.
- Augmentation + Dropout + weight decay + early stopping giúp validation bám sát train
  → overfitting được kiểm soát tốt.

### 10.3. Hướng phát triển (bài tập mở rộng)

1. **Tăng dữ liệu aug:** dịch chuyển (RandomCrop với padding), thay đổi độ sáng.
2. **Scheduler cosine + warmup** thay ReduceLROnPlateau.
3. **Kiến trúc sâu hơn:** thêm khối conv (3×3×3) hoặc dùng ResNet-18 (cần resize 28→224).
4. **Tăng epoch khi có GPU** (30–50 epoch) → accuracy ~93.5–94%.
5. **Phân tích lỗi sâu hơn:** vẽ Grad-CAM xem mạng "nhìn" vùng nào khi phân loại sai.
6. **So sánh model:** logistic regression / MLP / CNN để thấy lợi ích của tích chập (convolution).

---

*Tài liệu này thuộc dự án Practice 1. Chi tiết từng hàm code xem tại [file_guide.md](file_guide.md).*
