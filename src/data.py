"""Tải và chuẩn bị dữ liệu Fashion-MNIST.

Nguồn dữ liệu: https://www.kaggle.com/datasets/zalando-research/fashionmnist

Thứ tự ưu tiên khi nạp dữ liệu:
    1. File CSV của Kaggle đã nằm sẵn trong ``data/``
       (``fashion-mnist_train.csv`` / ``fashion-mnist_test.csv``).
    2. Tự động tải qua thư viện ``kagglehub`` rồi sao chép CSV vào ``data/``.
    3. Dự phòng: tải qua ``torchvision.datasets.FashionMNIST``.

Dữ liệu gồm 70.000 ảnh xám 28x28 (60.000 train + 10.000 test),
mỗi ảnh là một mặt hàng thời trang thuộc 1 trong 10 lớp.
"""

from __future__ import annotations

import shutil

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.transforms import functional as TF

from . import config

KAGGLE_DATASET = "zalando-research/fashionmnist"

# Các tên file CSV có thể gặp tuỳ theo nơi tải (Kaggle gốc / mirror)
TRAIN_CSV_NAMES = {"fashion-mnist_train.csv", "train.csv"}
TEST_CSV_NAMES = {"fashion-mnist_test.csv", "test.csv"}


# --------------------------------------------------------------------------- #
# Tìm kiếm / tự tải file CSV từ Kaggle
# --------------------------------------------------------------------------- #
def _find_csv(names: set[str]):
    """Tìm file CSV có tên thuộc ``names`` bên trong thư mục ``data/``."""
    if not config.DATA_DIR.exists():
        return None
    for path in config.DATA_DIR.rglob("*.csv"):
        if path.name.lower() in names:
            return path
    return None


def _download_kagglehub():
    """Cố gắng tải dataset từ Kaggle bằng ``kagglehub`` rồi chép CSV vào ``data/``.

    Trả về True nếu thành công, False nếu thất bại (chưa cài / không mạng...).
    """
    try:
        import kagglehub
    except ImportError:
        print("[data] Chưa cài `kagglehub` -> bỏ qua bước tải tự động từ Kaggle.")
        return False

    try:
        print(f"[data] Đang tải dataset '{KAGGLE_DATASET}' từ Kaggle (lần đầu có thể lâu)...")
        root = __import__("pathlib").Path(kagglehub.dataset_download(KAGGLE_DATASET))
    except Exception as exc:  # noqa: BLE001 - mọi lỗi mạng/token đều cho dự phòng
        print(f"[data] Tải bằng kagglehub thất bại: {exc}")
        return False

    # Sao chép 2 file CSV cần thiết vào data/ để các lần chạy sau không phải tải lại
    copied = 0
    for src in root.rglob("*.csv"):
        if src.name.lower() in (TRAIN_CSV_NAMES | TEST_CSV_NAMES):
            config.ensure_dirs()
            shutil.copy2(src, config.DATA_DIR / src.name)
            copied += 1
    print(f"[data] Đã sao chép {copied} file CSV vào thư mục {config.DATA_DIR}")
    return copied > 0


def _ensure_csv_data():
    """Trả về (train_csv, test_csv). Tự tải bằng kagglehub nếu chưa có."""
    train_csv = _find_csv(TRAIN_CSV_NAMES)
    test_csv = _find_csv(TEST_CSV_NAMES)
    if train_csv and test_csv:
        return train_csv, test_csv

    _download_kagglehub()
    return _find_csv(TRAIN_CSV_NAMES), _find_csv(TEST_CSV_NAMES)


# --------------------------------------------------------------------------- #
# Dataset đọc từ CSV của Kaggle
# --------------------------------------------------------------------------- #
class FashionMNISTCSVDataset(Dataset):
    """Dataset đọc từ DataFrame của file CSV Kaggle.

    Format CSV: cột đầu tiên là ``label`` (0-9), 784 cột còn lại là
    giá trị pixel (0-255) của ảnh 28x28 bị duỗi phẳng.

    Dữ liệu được chuẩn hoá sẵn: (pixel/255 - MEAN) / STD.
    Nếu ``augment=True`` (chỉ dùng cho tập train), mỗi lần lấy ảnh sẽ
    ngẫu nhiên lật ngang và xoay nhẹ ±10 độ để tăng cường dữ liệu.
    """

    def __init__(self, dataframe: pd.DataFrame, augment: bool = False):
        self.labels = torch.from_numpy(dataframe.iloc[:, 0].to_numpy(dtype=np.int64))
        images = dataframe.iloc[:, 1:].to_numpy(dtype=np.float32).reshape(-1, 1, 28, 28)
        images = images / 255.0
        images = (images - config.MEAN) / config.STD
        self.images = torch.from_numpy(images)
        self.augment = augment

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        image = self.images[idx]
        if self.augment:
            if torch.rand(1).item() < 0.5:
                image = torch.flip(image, dims=[-1])  # lật ngang
            # Dùng torch.rand thay random.uniform để seed được chia sẻ
            # đúng giữa các DataLoader worker -> kết quả tái lập hoàn toàn
            angle = (torch.rand(1).item() * 20.0) - 10.0  # [-10, +10] độ
            image = TF.rotate(image, angle)
        return image, self.labels[idx]


# --------------------------------------------------------------------------- #
# Nạp dữ liệu thô để khám phá (notebook 01)
# --------------------------------------------------------------------------- #
def load_raw_arrays():
    """Trả về (X_train, y_train, X_test, y_test).

    X có dtype uint8, shape (N, 28, 28), giá trị 0-255 (chưa chuẩn hoá) -
    dùng cho phần khám phá dữ liệu và vẽ ảnh.
    """
    train_csv, test_csv = _ensure_csv_data()

    if train_csv and test_csv:
        print(f"[data] Nạp từ CSV:\n  - train: {train_csv.name}\n  - test : {test_csv.name}")

        def _to_arrays(csv_path):
            df = pd.read_csv(csv_path)
            y = df.iloc[:, 0].to_numpy(dtype=np.int64)
            x = df.iloc[:, 1:].to_numpy(dtype=np.uint8).reshape(-1, 28, 28)
            return x, y

        x_train, y_train = _to_arrays(train_csv)
        x_test, y_test = _to_arrays(test_csv)
        return x_train, y_train, x_test, y_test

    # Dự phòng: không có CSV -> dùng dữ liệu torchvision đã tải
    print("[data] Không tìm thấy CSV -> dùng dữ liệu torchvision trong data/")
    from torchvision import datasets

    train = datasets.FashionMNIST(root=config.DATA_DIR, train=True, download=True)
    test = datasets.FashionMNIST(root=config.DATA_DIR, train=False, download=True)
    return (
        train.data.numpy(),
        train.targets.numpy(),
        test.data.numpy(),
        test.targets.numpy(),
    )


# --------------------------------------------------------------------------- #
# DataLoader cho huấn luyện / đánh giá
# --------------------------------------------------------------------------- #
def get_dataloaders(
    batch_size: int = config.BATCH_SIZE,
    val_split: float = config.VAL_SPLIT,
    use_augmentation: bool = True,
    num_workers: int = config.NUM_WORKERS,
    seed: int = config.SEED,
):
    """Tạo (train_loader, val_loader, test_loader).

    - Tập train gốc (60.000 ảnh) được chia thành train (90%)
      và validation (10%) theo ``seed`` để kết quả tái lập được.
    - ``use_augmentation=True`` chỉ áp dụng augmentation cho tập train.
    """
    config.ensure_dirs()
    pin_memory = torch.cuda.is_available()
    train_csv, test_csv = _ensure_csv_data()

    if train_csv and test_csv:
        # ------------------- Nguồn 1/2: CSV Kaggle ------------------- #
        print(f"[data] Nạp từ CSV:\n  - train: {train_csv.name}\n  - test : {test_csv.name}")
        full_df = pd.read_csv(train_csv)
        test_df = pd.read_csv(test_csv)

        n_val = int(len(full_df) * val_split)
        perm = np.random.RandomState(seed).permutation(len(full_df))
        val_idx, train_idx = perm[:n_val], perm[n_val:]

        train_ds = FashionMNISTCSVDataset(full_df.iloc[train_idx], augment=use_augmentation)
        val_ds = FashionMNISTCSVDataset(full_df.iloc[val_idx], augment=False)
        test_ds = FashionMNISTCSVDataset(test_df, augment=False)
    else:
        # ------------- Nguồn 3 (dự phòng): torchvision ------------- #
        print("[data] Không có CSV Kaggle -> tải Fashion-MNIST qua torchvision.")
        from torchvision import datasets, transforms

        eval_tf = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([config.MEAN], [config.STD]),
            ]
        )
        aug_tf = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                eval_tf,
            ]
        )
        train_full_aug = datasets.FashionMNIST(config.DATA_DIR, train=True, download=True, transform=aug_tf)
        train_full_eval = datasets.FashionMNIST(config.DATA_DIR, train=True, download=False, transform=eval_tf)
        test_ds = datasets.FashionMNIST(config.DATA_DIR, train=False, download=True, transform=eval_tf)

        n_val = int(len(train_full_aug) * val_split)
        perm = np.random.RandomState(seed).permutation(len(train_full_aug))
        val_idx, train_idx = perm[:n_val], perm[n_val:]

        # Tôn trọng flag use_augmentation: chỉ dùng aug_tf khi được yêu cầu
        src_dataset = train_full_aug if use_augmentation else train_full_eval
        train_ds = Subset(src_dataset, train_idx.tolist())
        val_ds = Subset(train_full_eval, val_idx.tolist())

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size * 2, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size * 2, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    print(
        f"[data] Số ảnh: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)} "
        f"(batch_size={batch_size}, augmentation={'bật' if use_augmentation else 'tắt'})"
    )
    return train_loader, val_loader, test_loader
