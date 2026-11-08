"""Điểm vào dòng lệnh: chạy toàn bộ pipeline từ đầu đến cuối.

Pipeline: nạp dữ liệu -> huấn luyện CNN -> đánh giá trên tập test -> lưu kết quả.

Ví dụ chạy:
    python main.py                       # dùng cấu hình mặc định trong src/config.py
    python main.py --epochs 20 --batch-size 128
    python main.py --lr 5e-4 --no-augment
    python main.py --epochs 30           # 30 epoch + early stopping 5 (cấu hình mở rộng)
    python main.py --epochs 30 --patience 5
"""

from __future__ import annotations

import argparse

import pandas as pd
import torch
import torch.nn as nn

from src import config, data, evaluate, utils
from src.model import build_model
from src.trainer import Trainer, load_trained_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Practice 1 - Phân loại ảnh Fashion-MNIST bằng CNN")
    # Mặc định dùng cấu hình mở rộng 30 epoch (vẫn override được bằng --epochs)
    parser.add_argument("--epochs", type=int, default=config.NUM_EPOCHS_EXTENDED, help="Số epoch tối đa")
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE, help="Kích thước batch")
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE, help="Learning rate")
    parser.add_argument("--val-split", type=float, default=config.VAL_SPLIT, help="Tỉ lệ tập validation")
    parser.add_argument("--seed", type=int, default=config.SEED, help="Seed tái lập kết quả")
    parser.add_argument("--no-augment", action="store_true", help="Tắt data augmentation")
    # BỔ SUNG: cho phép chỉnh early-stopping patience từ CLI (mặc định = 5)
    parser.add_argument("--patience", type=int, default=config.EARLY_STOP_PATIENCE,
                        help="Early stopping: số epoch chờ cải thiện val_loss (mặc định = 5)")
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    utils.set_seed(args.seed)
    config.ensure_dirs()
    device = utils.get_device()
    print("=" * 70)
    print("PRACTICE 1 - PHÂN LOẠI ẢNH FASHION-MNIST BẰNG CNN (PyTorch)")
    print("=" * 70)
    print(
        f"Thiết bị  : {device}\n"
        f"Seed      : {args.seed}  |  Epochs: {args.epochs}  |  Batch: {args.batch_size}\n"
        f"LR        : {args.lr}  |  Val split: {args.val_split:.0%}  |  "
        f"Augmentation: {'tắt' if args.no_augment else 'bật'}\n"
        f"Early stop: patience = {args.patience} epoch"
    )

    # ----------------------- Bước 1: Dữ liệu ----------------------- #
    print("\n[Bước 1/4] Chuẩn bị dữ liệu...")
    train_loader, val_loader, test_loader = data.get_dataloaders(
        batch_size=args.batch_size,
        val_split=args.val_split,
        use_augmentation=not args.no_augment,
        seed=args.seed,
    )

    # ----------------------- Bước 2: Mô hình ----------------------- #
    print("\n[Bước 2/4] Xây dựng mô hình...")
    model = build_model(num_classes=config.NUM_CLASSES)
    print(f"Kiến trúc: {model.__class__.__name__}")
    print(f"Tổng số tham số: {utils.count_parameters(model):,}")

    # ---------------------- Bước 3: Huấn luyện ---------------------- #
    print("\n[Bước 3/4] Huấn luyện...")
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    trainer = Trainer(model, criterion, optimizer, scheduler, device)
    history = trainer.fit(
        train_loader,
        val_loader,
        epochs=args.epochs,
        patience=args.patience,
        save_path=config.BEST_MODEL_PATH,
    )

    # Lưu bảng loss/acc từng epoch
    utils.save_table(pd.DataFrame(history), "training_history")

    # Vẽ đường cong học tập nâng cao: đánh dấu best point + chia giai đoạn
    utils.plot_training_history_advanced(
        history,
        save_name="training_history_advanced",
        stage_marks=[10, 20],  # chia 3 giai đoạn: 1-10, 11-20, 21-30
    )

    # ---------------------- Bước 4: Đánh giá ---------------------- #
    print("\n[Bước 4/4] Đánh giá trên tập test với model tốt nhất...")
    best_model, checkpoint = load_trained_model(config.BEST_MODEL_PATH, device=device)
    y_true, y_pred = evaluate.collect_predictions(best_model, test_loader, device)
    results = evaluate.evaluate_classification(y_true, y_pred)
    evaluate.save_evaluation_results(results, prefix="test")

    print("\n" + "=" * 70)
    print("HOÀN TẤT! Các file kết quả:")
    print(f"  - Trọng số mô hình : {config.BEST_MODEL_PATH}")
    print(f"  - Biểu đồ          : {config.FIGURES_DIR}")
    print(f"  - Bảng kết quả CSV : {config.TABLES_DIR}")
    print(f"  - Tài liệu (md)    : {config.DOCS_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main(parse_args())