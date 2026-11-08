"""Cấu hình toàn cục của dự án Practice 1 - Fashion-MNIST.

Tất cả đường dẫn, siêu tham số và hằng số dùng chung đều nằm ở đây
để các module khác chỉ cần import từ một nơi duy nhất.
"""

from pathlib import Path

# --------------------------------------------------------------------------- #
# Đường dẫn (tự động gắn với thư mục gốc của dự án)
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"                      # Dữ liệu (CSV Kaggle / torchvision)
MODELS_DIR = PROJECT_ROOT / "models"                  # Trọng số mô hình (.pth)
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"    # Biểu đồ, ảnh dự đoán (.png)
TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"      # Bảng kết quả (.csv)

# File trọng số tốt nhất được lưu trong quá trình huấn luyện
BEST_MODEL_PATH = MODELS_DIR / "best_model.pth"

# --------------------------------------------------------------------------- #
# Tên 10 lớp của Fashion-MNIST (chỉ số 0 -> 9)
# --------------------------------------------------------------------------- #
CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

NUM_CLASSES = len(CLASS_NAMES)  # 10

# Thống kê pixel của tập Fashion-MNIST (dùng để chuẩn hoá dữ liệu)
MEAN = 0.2860
STD = 0.3530

# --------------------------------------------------------------------------- #
# Siêu tham số huấn luyện (cấu hình duy nhất: 30 epoch + early stopping 5)
# --------------------------------------------------------------------------- #
SEED = 42                 # Seed để tái lập kết quả
BATCH_SIZE = 64           # Kích thước batch
NUM_EPOCHS = 30           # Số epoch tối đa (dùng 30 epoch theo yêu cầu)
LEARNING_RATE = 1e-3      # Learning rate khởi đầu (AdamW)
WEIGHT_DECAY = 1e-4       # Regularization L2 của AdamW
VAL_SPLIT = 0.1           # 10% tập train dùng làm tập validation
EARLY_STOP_PATIENCE = 5   # Dừng sớm nếu val_loss không giảm sau 5 epoch
NUM_WORKERS = 0           # Windows: để 0 cho ổn định, tăng lên nếu chạy Linux/macOS

# Alias tương thích ngược: các file cũ (main.py, notebook 02) đang gọi
# config.NUM_EPOCHS_EXTENDED và config.EARLY_STOP_PATIENCE_EXTENDED.
# Giờ đây hai cấu hình đã gộp làm một, alias chỉ trỏ về biến chính.
NUM_EPOCHS_EXTENDED = NUM_EPOCHS
EARLY_STOP_PATIENCE_EXTENDED = EARLY_STOP_PATIENCE

# Bước vẽ đường cong học tập: lưu epoch -> loss/acc
PLOT_EPOCH_STEP = 1       # vẽ mỗi epoch

# Đường dẫn tài liệu (dùng cho các file .md)
DOCS_DIR = PROJECT_ROOT / "docs"


def ensure_dirs() -> None:
    """Tạo các thư mục output (data, models, outputs/...) nếu chưa tồn tại."""
    for directory in (DATA_DIR, MODELS_DIR, FIGURES_DIR, TABLES_DIR, DOCS_DIR):
        directory.mkdir(parents=True, exist_ok=True)