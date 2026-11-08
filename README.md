# Practice 1 — Phân loại ảnh Fashion-MNIST bằng CNN (PyTorch)

Dự án xây dựng **pipeline học sâu end-to-end** hoàn chỉnh: tải dữ liệu → khám phá →
huấn luyện mạng CNN → đánh giá → suy luận, áp dụng lên bộ dữ liệu
**[Fashion-MNIST](https://www.kaggle.com/datasets/zalando-research/fashionmnist)**
của Zalando trên Kaggle.

- **Bài toán:** phân loại 70.000 ảnh xám 28×28 của các mặt hàng thời trang vào 10 lớp.
- **Mô hình:** CNN tự cài đặt (~0.87 triệu tham số), PyTorch.
- **Kết quả mong đợi:** accuracy tập test khoảng **92–93%**.
- **Ngôn ngữ / thư viện:** Python 3.10+, PyTorch, torchvision, pandas, scikit-learn, matplotlib, seaborn.

---

## 1. Cấu trúc thư mục

```
Practice_1/
│
├── data/                  # Dữ liệu (tự động tải về khi chạy code lần đầu)
│   ├── fashion-mnist_train.csv   # 60.000 ảnh train (label + 784 cột pixel)
│   └── fashion-mnist_test.csv    # 10.000 ảnh test
│
├── models/                # Lưu file trọng số (.pth)
│   └── best_model.pth     # Model tốt nhất theo val_loss (sinh ra sau huấn luyện)
│
├── outputs/
│   ├── figures/           # Biểu đồ, ảnh dự đoán (PNG)
│   └── tables/            # File CSV kết quả (báo cáo, ma trận nhầm lẫn, history)
│
├── src/                   # Logic code (Python files)
│   ├── __init__.py        # Khai báo package
│   ├── config.py          # Toàn bộ cấu hình: đường dẫn, siêu tham số, tên lớp
│   ├── data.py            # Tải/nạp dữ liệu Kaggle, Dataset, DataLoader
│   ├── model.py           # Kiến trúc CNN (FashionCNN)
│   ├── trainer.py         # Vòng lặp huấn luyện/validation, checkpoint, early stopping
│   ├── evaluate.py        # Đánh giá: accuracy, classification report, ma trận nhầm lẫn
│   └── utils.py           # Seed, thiết bị, vẽ biểu đồ, lưu file
│
├── notebooks/             # Các file Jupyter để chạy và báo cáo
│   ├── 01_data_exploration.ipynb         # Khám phá dữ liệu
│   ├── 02_model_training.ipynb           # Huấn luyện CNN
│   └── 03_evaluation_and_inference.ipynb # Đánh giá & suy luận
│
├── docs/                  # Tài liệu chi tiết
│   ├── chapter_1_fashion_mnist.md  # Bài làm Chương 1 (chi tiết toàn bộ bài thực hành)
│   └── file_guide.md               # Giải thích chi tiết từng file, từng hàm
│
├── main.py                # Chạy toàn bộ pipeline bằng 1 lệnh duy nhất
├── requirements.txt       # Danh sách thư viện cần cài
├── README.md              # File này
└── .gitignore
```

## 2. Ý nghĩa từng file

| File | Vai trò |
|---|---|
| `src/config.py` | Nơi duy nhất chứa cấu hình: đường dẫn thư mục, 10 tên lớp, siêu tham số (batch size, epochs, learning rate...), hàm `ensure_dirs()` tạo thư mục. |
| `src/data.py` | Tự tìm CSV Kaggle trong `data/`, nếu chưa có thì tải qua `kagglehub` (dự phòng torchvision); tạo `Dataset`/`DataLoader`, chia train/validation, augmentation. |
| `src/model.py` | Lớp `FashionCNN`: 2 khối Conv+BN+ReLU+MaxPool+Dropout rồi MLP phân loại 10 lớp. |
| `src/trainer.py` | Lớp `Trainer`: huấn luyện từng epoch, validate, lưu checkpoint tốt nhất, early stopping; hàm `load_trained_model()` nạp lại model. |
| `src/evaluate.py` | Thu dự đoán, tính accuracy/Precision/Recall/F1, ma trận nhầm lẫn, tìm ảnh sai, dự đoán 1 ảnh kèm xác suất. |
| `src/utils.py` | `set_seed`, `get_device`, đếm tham số, vẽ biểu đồ (phân bố lớp, lưới ảnh, đường cong học tập, heatmap ma trận nhầm lẫn), lưu PNG/CSV. |
| `notebooks/01_*.ipynb` | Khám phá dữ liệu: kích thước, phân bố lớp, ảnh mẫu, ảnh trung bình mỗi lớp. |
| `notebooks/02_*.ipynb` | Huấn luyện: tạo DataLoader, xây model, chạy `Trainer.fit()`, vẽ đường cong học tập. |
| `notebooks/03_*.ipynb` | Đánh giá & suy luận: nạp `best_model.pth`, báo cáo trên tập test, ảnh sai, dự đoán 1 ảnh. |
| `main.py` | Gộp cả 4 bước (dữ liệu → model → huấn luyện → đánh giá) chạy bằng dòng lệnh. |
| `requirements.txt` | Danh sách thư viện: `pip install -r requirements.txt`. |
| `docs/chapter_1_fashion_mnist.md` | **Bài làm Chương 1** — giải thích chi tiết lý thuyết + từng bước thực hành. |
| `docs/file_guide.md` | Mô tả chi tiết từng file, từng hàm: đầu vào/đầu ra, khi nào dùng. |

## 3. Cài đặt môi trường

```bash
# 1. Tạo và kích hoạt môi trường ảo (khuyến nghị)
python -m venv .venv
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# Windows (Git Bash):
source .venv/Scripts/activate

# 2. Cài thư viện
pip install -r requirements.txt
```

## 4. Tải dữ liệu từ Kaggle

Dự án hỗ trợ **3 cách**, code tự động áp dụng cách phù hợp:

**Cách 1 — Tự động (khuyến nghị, mặc định):** khi chạy notebook hoặc `main.py`
lần đầu mà `data/` chưa có CSV, code tự tải qua `kagglehub` (đã có trong
`requirements.txt`) và sao chép 2 file CSV vào `data/`. Các lần sau chỉ đọc file cũ.

**Cách 2 — Kaggle CLI (tải tay bằng lệnh):**

```bash
pip install kaggle
# Cần đặt file kaggle.json (từ trang Kaggle -> Settings -> API -> Create New Token)
# vào C:\Users\<tên_bạn>\.kaggle\kaggle.json
kaggle datasets download -d zalando-research/fashionmnist -p data --unzip
```

**Cách 3 — Tải tay trên web:** vào
[kaggle.com/datasets/zalando-research/fashionmnist](https://www.kaggle.com/datasets/zalando-research/fashionmnist),
bấm **Download**, giải nén 2 file CSV vào thư mục `data/`.

> Nếu cả Kaggle đều không khả dụng, code sẽ **tự dự phòng** tải Fashion-MNIST qua
> `torchvision` — pipeline vẫn chạy bình thường.

## 5. Cách chạy

**Cách A — theo từng notebook (khuyến nghị khi học):**

```bash
cd Practice_1
jupyter notebook
# Mở lần lượt theo thứ tự:
# notebooks/01_data_exploration.ipynb
# notebooks/02_model_training.ipynb
# notebooks/03_evaluation_and_inference.ipynb
```

**Cách B — chạy toàn bộ bằng 1 lệnh:**

```bash
python main.py                      # cấu hình mặc định (15 epoch, batch 64)
python main.py --epochs 20 --batch-size 128 --lr 5e-4
python main.py --no-augment         # tắt augmentation
```

## 6. Kết quả sau khi chạy

| Vị trí | Nội dung |
|---|---|
| `models/best_model.pth` | Trọng số CNN tốt nhất (val_loss thấp nhất) kèm metadata. |
| `outputs/figures/*.png` | Phân bố lớp, ảnh mẫu, ảnh trung bình, đường cong học tập, ma trận nhầm lẫn, ảnh sai, ví dụ suy luận. |
| `outputs/tables/*.csv` | `classification_report_test.csv`, `confusion_matrix_test.csv`, `training_history.csv`. |

## 7. Tài liệu chi tiết

- 📘 [docs/chapter_1_fashion_mnist.md](docs/chapter_1_fashion_mnist.md) — bài làm Chương 1 đầy đủ.
- 📄 [docs/file_guide.md](docs/file_guide.md) — giải thích chi tiết từng file, từng hàm.

## 8. Nguồn tham khảo

- Dataset: [Fashion-MNIST — Kaggle (zalando-research)](https://www.kaggle.com/datasets/zalando-research/fashionmnist)
- Bài báo gốc: Xiao, Rasul & Vollgraf, *Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms*, arXiv:1708.07747 (2017)
- Tài liệu PyTorch: <https://pytorch.org/docs/stable/index.html>
