"""Các hàm tiện ích dùng chung cho toàn bộ dự án Practice 1 — Fashion-MNIST."""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Sequence

import numpy as np
import matplotlib.pyplot as plt
import torch

from . import config


# =====================================================================
# 1. SEED & DEVICE
# =====================================================================
def set_seed(seed: int = 42) -> None:
    """Cố định seed cho toàn bộ random / numpy / torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Trả về device tốt nhất hiện có."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# =====================================================================
# 2. ĐẾM THAM SỐ
# =====================================================================
def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# =====================================================================
# 3. LƯU FIGURE / BẢNG
# =====================================================================
def _ensure_dir(path: Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_figure(fig: plt.Figure, name: str, dpi: int = 150) -> Path:
    """Lưu figure vào outputs/figures/<name>.png."""
    fig_dir = _ensure_dir(config.PROJECT_ROOT / "outputs" / "figures")
    out_path = fig_dir / f"{name}.png"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    print(f"[utils] Đã lưu hình: {out_path}")
    return out_path


def save_table(df, name: str) -> Path:
    """Lưu DataFrame vào outputs/tables/<name>.csv (utf-8-sig để mở Excel)."""
    import pandas as pd

    table_dir = _ensure_dir(config.PROJECT_ROOT / "outputs" / "tables")
    out_path = table_dir / f"{name}.csv"
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"[utils] Đã lưu bảng: {out_path}")
    return out_path


# =====================================================================
# 4. VẼ BIỂU ĐỒ DỮ LIỆU
# =====================================================================
def plot_class_distribution(y_train: np.ndarray, save_name: str | None = None) -> plt.Figure:
    """Biểu đồ cột phân bố số ảnh mỗi lớp."""
    counts = np.bincount(y_train, minlength=config.NUM_CLASSES)
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(config.CLASS_NAMES, counts, color="tab:blue", alpha=0.85)
    for bar, c in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
                f"{c:,}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(range(len(config.CLASS_NAMES)))
    ax.set_xticklabels(config.CLASS_NAMES, rotation=30, ha="right")
    ax.set_ylabel("Số ảnh")
    ax.set_title("Phân bố số ảnh theo lớp (tập train)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    if save_name:
        save_figure(fig, save_name)
    return fig


def plot_image_grid(
    images: Sequence[np.ndarray],
    titles: Sequence[str],
    ncols: int = 8,
    save_name: str | None = None,
    cmap: str = "gray",
) -> plt.Figure:
    """Vẽ lưới ảnh với tiêu đề tương ứng. Tự động chuyển tensor sang numpy và xử lý shape."""
    n = len(images)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 1.6, nrows * 1.7))
    axes = np.array(axes).reshape(-1)

    for ax, img, title in zip(axes, images, titles):
        # 1. Nếu là tensor PyTorch, chuyển sang numpy
        if torch.is_tensor(img):
            img = img.cpu().detach().numpy()
            
        # 2. Xử lý shape (C, H, W) hoặc (1, H, W) -> (H, W)
        if img.ndim == 3:
            if img.shape[0] == 1: # (1, 28, 28) -> (28, 28)
                img = img.squeeze(0)
            elif img.shape[0] == 3: # (3, H, W) -> (H, W, 3)
                img = np.transpose(img, (1, 2, 0))

        ax.imshow(img, cmap=cmap)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
        
    for ax in axes[n:]:
        ax.axis("off")

    fig.tight_layout()
    if save_name:
        save_figure(fig, save_name)
    return fig


def plot_samples_per_class(X: np.ndarray, y: np.ndarray, save_name: str | None = None) -> plt.Figure:
    """Vẽ 1 ảnh mẫu cho mỗi lớp (10 lớp → 10 ảnh)."""
    imgs, titles = [], []
    for c in sorted(np.unique(y).tolist()):
        idx = np.where(y == c)[0][0]
        imgs.append(X[idx])
        titles.append(config.CLASS_NAMES[c])
    return plot_image_grid(imgs, titles, ncols=5, save_name=save_name)


# =====================================================================
# 5. VẼ LỊCH SỬ HUẤN LUYỆN
# =====================================================================
def plot_training_history(history: dict, save_name: str | None = None) -> plt.Figure:
    """Vẽ loss & accuracy — Train/Val tách riêng (chưa có Test)."""
    epochs = list(range(1, len(history["train_loss"]) + 1))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    ax = axes[0]
    ax.plot(epochs, history["train_loss"], label="Train", color="tab:blue", marker="o", ms=3, lw=1.8)
    ax.plot(epochs, history["val_loss"], label="Val", color="tab:orange", marker="s", ms=3, lw=1.8)
    ax.set_title("Loss — Train / Val")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(epochs, history["train_acc"], label="Train", color="tab:blue", marker="o", ms=3, lw=1.8)
    ax.plot(epochs, history["val_acc"], label="Val", color="tab:orange", marker="s", ms=3, lw=1.8)
    ax.set_title("Accuracy — Train / Val")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy"); ax.legend(); ax.grid(alpha=0.3)

    fig.tight_layout()
    if save_name:
        save_figure(fig, save_name)
    return fig


def plot_training_history_advanced(
    history: dict,
    test_metrics: dict | None = None,
    stage_marks: list[int] | None = None,
    save_name: str | None = None,
) -> plt.Figure:
    """Vẽ loss & accuracy với Train / Val / Test TÁCH RIÊNG.

    Test vẽ dạng đường ngang vì chỉ có 1 giá trị cuối.
    stage_marks: các epoch chia giai đoạn (vẽ đường dọc mờ).
    """
    epochs = list(range(1, len(history["train_loss"]) + 1))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ---------- LOSS ----------
    ax = axes[0]
    ax.plot(epochs, history["train_loss"], label="Train", color="tab:blue", marker="o", ms=3, lw=1.8)
    ax.plot(epochs, history["val_loss"], label="Val", color="tab:orange", marker="s", ms=3, lw=1.8)
    if test_metrics and "test_loss" in test_metrics:
        ax.axhline(test_metrics["test_loss"], color="tab:green", ls="--", lw=1.6,
                   label=f"Test ({test_metrics['test_loss']:.4f})")
    if stage_marks:
        for m in stage_marks:
            if 1 <= m <= len(epochs):
                ax.axvline(m, color="gray", ls=":", alpha=0.6)
    ax.set_title("Loss — Train / Val / Test (tách riêng)")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.legend(); ax.grid(alpha=0.3)

    # ---------- ACCURACY ----------
    ax = axes[1]
    ax.plot(epochs, history["train_acc"], label="Train", color="tab:blue", marker="o", ms=3, lw=1.8)
    ax.plot(epochs, history["val_acc"], label="Val", color="tab:orange", marker="s", ms=3, lw=1.8)
    if test_metrics and "test_acc" in test_metrics:
        ax.axhline(test_metrics["test_acc"], color="tab:green", ls="--", lw=1.6,
                   label=f"Test ({test_metrics['test_acc']:.4f})")
    if stage_marks:
        for m in stage_marks:
            if 1 <= m <= len(epochs):
                ax.axvline(m, color="gray", ls=":", alpha=0.6)
    ax.set_title("Accuracy — Train / Val / Test (tách riêng)")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy"); ax.legend(); ax.grid(alpha=0.3)

    fig.tight_layout()
    if save_name:
        save_figure(fig, save_name)
    return fig


# =====================================================================
# 6. VẼ CONFUSION MATRIX
# =====================================================================
def plot_confusion_matrix(
    cm: np.ndarray,
    normalize: bool = False,
    save_name: str | None = None,
    cmap: str = "Blues",
    figsize: tuple[float, float] = (9, 7),
) -> plt.Figure:
    """Vẽ confusion matrix.

    normalize=True  → chuẩn hoá theo dòng (tỉ lệ %).
    normalize=False → số lượng tuyệt đối.
    """
    cm = np.asarray(cm, dtype=float)
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        cm_display = cm / row_sums
        fmt = ".2f"
        title = "Confusion Matrix — Chuẩn hoá theo dòng (tỉ lệ)"
        cbar_label = "Tỉ lệ"
    else:
        cm_display = cm
        fmt = ".0f"
        title = "Confusion Matrix — Số lượng tuyệt đối"
        cbar_label = "Số ảnh"

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm_display, cmap=cmap, aspect="auto")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label)

    ax.set_xticks(range(config.NUM_CLASSES))
    ax.set_yticks(range(config.NUM_CLASSES))
    ax.set_xticklabels(config.CLASS_NAMES, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(config.CLASS_NAMES, fontsize=9)
    ax.set_xlabel("Dự đoán"); ax.set_ylabel("Thật"); ax.set_title(title)

    thresh = cm_display.max() / 2.0 if cm_display.max() > 0 else 0.5
    for i in range(config.NUM_CLASSES):
        for j in range(config.NUM_CLASSES):
            val = cm_display[i, j]
            color = "white" if val > thresh else "black"
            ax.text(j, i, format(val, fmt), ha="center", va="center",
                    color=color, fontsize=8)

    fig.tight_layout()
    if save_name:
        save_figure(fig, save_name)
    return fig