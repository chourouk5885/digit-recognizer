"""A compact CNN for 28x28 grayscale digits (~470k parameters; typically 99.4-99.6% test accuracy)."""

import torch
from torch import nn


def conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    """3x3 conv -> batch norm -> ReLU. Padding keeps the spatial size."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class DigitCNN(nn.Module):
    """
    Input (N, 1, 28, 28)
      -> [conv 32, conv 32] -> max-pool -> 14x14
      -> [conv 64, conv 64] -> max-pool -> 7x7
      -> flatten -> dense 128 -> dropout -> dense 10 (logits)
    """

    def __init__(self, num_classes: int = 10, dropout: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            conv_block(1, 32),
            conv_block(32, 32),
            nn.MaxPool2d(2),
            nn.Dropout(0.1),
            conv_block(32, 64),
            conv_block(64, 64),
            nn.MaxPool2d(2),
            nn.Dropout(0.1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
