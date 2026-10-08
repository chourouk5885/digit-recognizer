"""Load a trained model and predict digits from raw drawings."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .model import DigitCNN
from .preprocess import normalize, to_mnist


def save_checkpoint(model: torch.nn.Module, path: str | Path, **meta) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), **meta}, path)


def load_model(path: str | Path, device: str | torch.device = "cpu") -> DigitCNN:
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model = DigitCNN()
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval()


class DigitPredictor:
    def __init__(self, weights_path: str | Path, device: str = "cpu"):
        self.device = torch.device(device)
        self.model = load_model(weights_path, self.device)

    @torch.no_grad()
    def predict(self, image) -> tuple[dict[str, float], np.ndarray | None]:
        """Raw drawing (numpy/PIL) -> ({"0": p0, ..., "9": p9}, the 28x28 image the model saw).

        Returns ({}, None) for an empty canvas.
        """
        img28 = to_mnist(image)
        if img28 is None:
            return {}, None
        x = torch.from_numpy(normalize(img28)).to(self.device)
        probs = torch.softmax(self.model(x), dim=1)[0].cpu().numpy()
        return {str(d): float(p) for d, p in enumerate(probs)}, img28
