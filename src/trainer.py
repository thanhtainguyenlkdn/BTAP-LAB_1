"""Vòng lặp huấn luyện, validation, lưu model tốt nhất và lịch sử huấn luyện.

Lớp ``Trainer`` chịu trách nhiệm:
    - Chạy từng epoch: huấn luyện trên train_loader, đánh giá trên val_loader.
    - Theo dõi history (loss/accuracy từng epoch) để vẽ đường cong học tập.
    - Lưu checkpoint tốt nhất (val_loss thấp nhất) vào ``models/best_model.pth``.
    - Dừng sớm (early stopping) khi val_loss không giảm sau `patience` epoch.
"""

from __future__ import annotations

import time
from pathlib import Path

import torch
import torch.nn as nn
from tqdm.auto import tqdm

from . import config, utils


class Trainer:
    """Đối tượng huấn luyện gói model, loss, optimizer và scheduler."""

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module | None = None,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler=None,
        device: torch.device | None = None,
    ):
        self.device = device or utils.get_device()
        self.model = model.to(self.device)
        self.criterion = criterion or nn.CrossEntropyLoss()
        self.optimizer = optimizer or torch.optim.AdamW(
            model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY,
        )
        self.scheduler = scheduler  # có thể để None

    # ------------------------------------------------------------------ #
    # Huấn luyện đầy đủ
    # ------------------------------------------------------------------ #
    def fit(
        self,
        train_loader,
        val_loader,
        epochs: int = config.NUM_EPOCHS,
        patience: int = config.EARLY_STOP_PATIENCE,
        save_path=config.BEST_MODEL_PATH,
    ) -> dict:
        """Huấn luyện tối đa ``epochs`` epoch, trả về history dạng dict.

        History gồm các khoá: train_loss, train_acc, val_loss, val_acc.
        Model tốt nhất (val_loss thấp nhất) được lưu vào ``save_path``.
        """
        history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
        best_val_loss = float("inf")
        epochs_no_improve = 0

        for epoch in range(1, epochs + 1):
            t0 = time.time()
            train_loss, train_acc = self.train_one_epoch(train_loader, epoch, epochs)
            val_loss, val_acc = self.validate(val_loader)
            if self.scheduler is not None:
                self.scheduler.step(val_loss)

            elapsed = time.time() - t0
            print(
                f"Epoch {epoch:03d}/{epochs} | "
                f"train loss {train_loss:.4f}, acc {train_acc:.4f} | "
                f"val loss {val_loss:.4f}, acc {val_acc:.4f} | {elapsed:.1f}s",
                flush=True,
            )
            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            if val_loss < best_val_loss - 1e-4:
                best_val_loss = val_loss
                epochs_no_improve = 0
                save_checkpoint(save_path, self.model, epoch, val_loss, val_acc)
                print(f"  -> Lưu model tốt nhất (val_loss={val_loss:.4f})")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(
                        f"Dừng sớm tại epoch {epoch}: val_loss không giảm "
                        f"sau {patience} epoch liền."
                    )
                    break

        return history

    # ------------------------------------------------------------------ #
    # Các bước con
    # ------------------------------------------------------------------ #
    def train_one_epoch(self, loader, epoch=None, epochs=None):
        """Huấn luyện 1 epoch trên ``loader``, trả về (loss trung bình, accuracy)."""
        self.model.train()
        total_loss = total_correct = total = 0
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{epochs} [train]", leave=False)
        for images, labels in pbar:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += batch_size
            pbar.set_postfix(loss=f"{total_loss / total:.4f}", acc=f"{total_correct / total:.4f}")
        return total_loss / total, total_correct / total

    @torch.no_grad()
    def validate(self, loader):
        """Đánh giá 1 epoch trên tập validation (không tính gradient)."""
        self.model.eval()
        total_loss = total_correct = total = 0
        for images, labels in loader:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += batch_size
        return total_loss / total, total_correct / total


# --------------------------------------------------------------------------- #
# Lưu / nạp checkpoint
# --------------------------------------------------------------------------- #
def save_checkpoint(path, model: nn.Module, epoch: int, val_loss: float, val_acc: float) -> None:
    """Lưu state_dict kèm metadata vào ``path``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "epoch": epoch,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "class_names": config.CLASS_NAMES,
        },
        path,
    )


def load_trained_model(path=config.BEST_MODEL_PATH, device=None, num_classes=config.NUM_CLASSES):
    """Nạp model tốt nhất từ checkpoint, trả về (model ở chế độ eval, checkpoint).

    Lưu ý: cần huấn luyện (notebook 02 hoặc `python main.py`) trước khi gọi hàm này.
    """
    from .model import build_model  # import trong hàm để tránh phụ thuộc vòng

    device = device or utils.get_device()
    # weights_only=False: checkpoint chứa cả metadata (class_names, epoch...)
    # nên phải cho phép unpickle toàn bộ. Chỉ load file tự sinh ra từ dự án này.
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = build_model(num_classes=num_classes)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    print(
        f"[trainer] Đã nạp model từ {path} "
        f"(epoch {checkpoint['epoch']}, val_loss={checkpoint['val_loss']:.4f}, "
        f"val_acc={checkpoint['val_acc']:.4f})"
    )
    return model, checkpoint
