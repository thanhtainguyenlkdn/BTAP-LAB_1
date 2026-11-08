"""Định nghĩa kiến trúc mô hình CNN cho bài toán phân loại 10 lớp Fashion-MNIST.

Kiến trúc ``FashionCNN``:
    - 2 khối Conv (mỗi khối gồm 2 cặp Conv3x3-BN-ReLU, tiếp theo là MaxPool và Dropout2d)
      trích xuất đặc trưng: 28×28 -> 14×14 -> 7×7, kênh 1 -> 32 -> 64.
    - Đầu phân loại (MLP): Flatten -> Linear(3136, 256) -> BN1d -> ReLU -> Dropout
      -> Linear(256, 10).
    Tổng cộng khoảng 0.87 triệu tham số.
"""

import torch.nn as nn


class FashionCNN(nn.Module):
    """Mạng CNN gọn nhẹ cho ảnh xám 28x28, đầu ra logits của 10 lớp."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            # ---- Khối 1: (1, 28, 28) -> (32, 14, 14) ---- #
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), 
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),          # 28 -> 14
            nn.Dropout2d(0.25),
            # ---- Khối 2: (32, 14, 14) -> (64, 7, 7) ---- #
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),          # 14 -> 7
            nn.Dropout2d(0.25),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),                         # (64, 7, 7) -> 3136
            nn.Linear(64 * 7 * 7, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),          # logits 10 lớp
        )

    def forward(self, x):
        """Nhận tensor ảnh (B, 1, 28, 28) đã chuẩn hoá, trả về logits (B, num_classes)."""
        return self.classifier(self.features(x))


def build_model(num_classes: int = 10) -> FashionCNN:
    """Tạo model mới (chưa huấn luyện)."""
    return FashionCNN(num_classes=num_classes)
