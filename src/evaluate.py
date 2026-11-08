"""Đánh giá mô hình: dự đoán, accuracy, classification report, confusion matrix.

Các hàm chính:
    - collect_predictions       : chạy model trên 1 DataLoader, thu nhãn thật/dự đoán.
    - evaluate_classification   : tính accuracy, báo cáo từng lớp, confusion matrix.
    - save_evaluation_results   : lưu 2 bảng CSV (report + confusion matrix).
    - get_misclassified         : lấy các ảnh bị dự đoán sai.
    - get_correct_per_class     : lấy 1 ảnh dự đoán ĐÚNG cho mỗi lớp (đủ 10 ảnh, đúng thứ tự).
    - predict_single            : dự đoán 1 ảnh kèm phân phối xác suất.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tqdm.auto import tqdm

from . import config, utils


# --------------------------------------------------------------------------- #
# Thu thập dự đoán
# --------------------------------------------------------------------------- #
@torch.no_grad()
def collect_predictions(model, loader, device=None):
    """Chạy model trên toàn bộ ``loader``, trả về (y_true, y_pred) dạng numpy."""
    device = device or utils.get_device()
    model.to(device)
    model.eval()
    all_labels, all_preds = [], []
    for images, labels in tqdm(loader, desc="Dự đoán", leave=False):
        outputs = model(images.to(device))
        all_preds.append(outputs.argmax(dim=1).cpu().numpy())
        all_labels.append(labels.numpy() if torch.is_tensor(labels) else np.asarray(labels))
    return np.concatenate(all_labels), np.concatenate(all_preds)


# --------------------------------------------------------------------------- #
# Đánh giá tổng thể
# --------------------------------------------------------------------------- #
def evaluate_classification(y_true, y_pred) -> dict:
    """Tính các chỉ số và in báo cáo ra màn hình.

    Trả về dict: {"accuracy", "report", "report_text", "confusion_matrix"}.
    """
    accuracy = accuracy_score(y_true, y_pred)
    report_text = classification_report(y_true, y_pred, target_names=config.CLASS_NAMES, digits=4)
    report = classification_report(
        y_true, y_pred, target_names=config.CLASS_NAMES, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=list(range(config.NUM_CLASSES)))

    print(f"Accuracy trên tập đánh giá: {accuracy * 100:.2f}%\n")
    print(report_text)
    return {
        "accuracy": accuracy,
        "report": report,
        "report_text": report_text,
        "confusion_matrix": cm,
    }


def save_evaluation_results(results: dict, prefix: str = "test") -> dict:
    """Lưu 2 bảng CSV (report + confusion matrix). Không vẽ hình ở đây —
    notebook tự vẽ để tránh trùng lặp."""
    report = results["report"]

    # Bảng precision/recall/f1 từng lớp + các dòng tổng hợp
    rows = []
    for name in config.CLASS_NAMES:
        r = report[name]
        rows.append({
            "class": name,
            "precision": round(r["precision"], 4),
            "recall": round(r["recall"], 4),
            "f1_score": round(r["f1-score"], 4),
            "support": int(r["support"]),
        })
    for key in ("macro avg", "weighted avg"):
        r = report[key]
        rows.append({
            "class": key,
            "precision": round(r["precision"], 4),
            "recall": round(r["recall"], 4),
            "f1_score": round(r["f1-score"], 4),
            "support": int(r["support"]),
        })
    rows.append({
        "class": "accuracy",
        "precision": "",
        "recall": "",
        "f1_score": round(results["accuracy"], 4),
        "support": int(report["macro avg"]["support"]),
    })
    report_path = utils.save_table(pd.DataFrame(rows), f"classification_report_{prefix}")

    # Bảng confusion matrix
    cm_df = pd.DataFrame(
        results["confusion_matrix"],
        index=[f"true_{name}" for name in config.CLASS_NAMES],
        columns=[f"pred_{name}" for name in config.CLASS_NAMES],
    ).reset_index().rename(columns={"index": "actual"})
    cm_path = utils.save_table(cm_df, f"confusion_matrix_{prefix}")

    return {"report": report_path, "confusion_matrix": cm_path}


# --------------------------------------------------------------------------- #
# Trực quan hoá lỗi & suy luận
# --------------------------------------------------------------------------- #
@torch.no_grad()
def get_misclassified(model, loader, device=None, max_samples: int = 16):
    """Trả về (images, trues, preds) của các ảnh bị dự đoán sai."""
    device = device or utils.get_device()
    model.to(device)
    model.eval()
    images, trues, preds = [], [], []
    for batch_images, batch_labels in loader:
        outputs = model(batch_images.to(device))
        pred = outputs.argmax(dim=1).cpu()
        wrong = pred != batch_labels
        for i in torch.nonzero(wrong).ravel().tolist():
            images.append(batch_images[i].cpu())
            trues.append(int(batch_labels[i]))
            preds.append(int(pred[i]))
            if len(images) >= max_samples:
                return images, trues, preds
    return images, trues, preds


@torch.no_grad()
def get_correct_per_class(model, loader, device=None, class_names=None):
    """Lấy 1 ảnh dự đoán ĐÚNG cho mỗi lớp (đủ 10 ảnh, đúng thứ tự 0 → 9).

    Nếu không tìm đủ → báo lỗi rõ ràng (không lọc âm thầm).
    """
    device = device or utils.get_device()
    model.to(device)
    model.eval()
    class_names = class_names or config.CLASS_NAMES

    images = [None] * config.NUM_CLASSES
    trues = [None] * config.NUM_CLASSES
    preds = [None] * config.NUM_CLASSES
    found = 0

    for batch_images, batch_labels in loader:
        outputs = model(batch_images.to(device))
        pred = outputs.argmax(dim=1).cpu()
        for i in range(len(batch_labels)):
            t, p = int(batch_labels[i]), int(pred[i])
            if t == p and images[t] is None:
                images[t] = batch_images[i].cpu()
                trues[t] = t
                preds[t] = p
                found += 1
        if found == config.NUM_CLASSES:
            break

    if found < config.NUM_CLASSES:
        missing = [class_names[i] for i in range(config.NUM_CLASSES) if images[i] is None]
        raise RuntimeError(
            f"Không tìm đủ 10 ảnh dự đoán đúng — thiếu lớp: {missing}."
        )
    return images, trues, preds


def predict_single(model, image, device=None, class_names=None, already_normalized: bool = False):
    """Dự đoán nhãn cho 1 ảnh (28, 28) hoặc (1, 28, 28).

    ``already_normalized``: True nếu ảnh đã qua (x/255 - MEAN)/STD.
    Trả về (idx, name, probs_10_lớp).
    """
    device = device or utils.get_device()
    class_names = class_names or config.CLASS_NAMES

    if torch.is_tensor(image):
        x = image.detach().clone().float()
    else:
        x = torch.as_tensor(np.asarray(image), dtype=torch.float32)
    x = x.reshape(1, 28, 28)
    if not already_normalized:
        x = (x - config.MEAN) / config.STD
    x = x.unsqueeze(0).to(device)

    model.to(device)
    model.eval()
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0].cpu().numpy()
    idx = int(probs.argmax())
    return idx, class_names[idx], probs