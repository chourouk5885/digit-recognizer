"""Turn a drawing from the web canvas into an MNIST-style 28x28 image.

This step matters more than the model for the live demo. MNIST digits are
white-on-black, fit in a 20x20 box and are centred by their centre of mass in a
28x28 frame. A canvas drawing is black-on-white (or on a transparent
background), any size, anywhere on the canvas. If we just resized the canvas to
28x28 the model would see something it was never trained on. So we reproduce
the original MNIST pipeline:

    composite over white -> invert -> crop to the ink -> resize longest side to 20px
    -> paste into 28x28 -> shift so the centre of mass is in the middle
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from .constants import MNIST_MEAN, MNIST_STD

INK_THRESHOLD = 30   # pixel values (0-255) below this are treated as background
MIN_INK_PIXELS = 15  # fewer inked pixels than this = empty canvas


def to_ink(image) -> np.ndarray:
    """Return a float array (H, W) where 255 = ink and 0 = background.

    Accepts a PIL image or a numpy array that is grayscale (H, W), RGB (H, W, 3)
    or RGBA (H, W, 4). Transparent areas are treated as background. Works for dark
    ink on a light background and for light ink on a dark background.
    """
    if isinstance(image, Image.Image):
        image = np.asarray(image.convert("RGBA"))
    arr = np.asarray(image, dtype=np.float32)
    if arr.ndim == 3 and arr.shape[2] == 4:
        alpha = arr[..., 3:4] / 255.0
        arr = arr[..., :3] * alpha + 255.0 * (1.0 - alpha)  # composite over white
    if arr.ndim == 3:
        gray = arr[..., :3] @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    else:
        gray = arr
    # Mostly-light image -> dark ink on light paper -> invert.
    return 255.0 - gray if gray.mean() > 127 else gray


def _shift(img: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """Translate without wrap-around (pixels shifted out are dropped)."""
    h, w = img.shape
    out = np.zeros_like(img)
    ys, yd = (slice(0, h - dy), slice(dy, h)) if dy >= 0 else (slice(-dy, h), slice(0, h + dy))
    xs, xd = (slice(0, w - dx), slice(dx, w)) if dx >= 0 else (slice(-dx, w), slice(0, w + dx))
    out[yd, xd] = img[ys, xs]
    return out


def to_mnist(image) -> np.ndarray | None:
    """Canvas drawing -> uint8 28x28, white digit on black, MNIST-style. None if the canvas is empty."""
    ink = to_ink(image)
    mask = ink > INK_THRESHOLD
    if mask.sum() < MIN_INK_PIXELS:
        return None

    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    crop = ink[rows[0]: rows[-1] + 1, cols[0]: cols[-1] + 1]

    h, w = crop.shape
    scale = 20.0 / max(h, w)
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    small = Image.fromarray(np.clip(crop, 0, 255).astype(np.uint8)).resize((new_w, new_h), Image.LANCZOS)

    canvas = np.zeros((28, 28), dtype=np.float32)
    top, left = (28 - new_h) // 2, (28 - new_w) // 2
    canvas[top: top + new_h, left: left + new_w] = np.asarray(small, dtype=np.float32)

    total = canvas.sum()
    if total > 0:
        yy, xx = np.indices(canvas.shape)
        cy, cx = (yy * canvas).sum() / total, (xx * canvas).sum() / total
        canvas = _shift(canvas, int(round(13.5 - cy)), int(round(13.5 - cx)))
    return np.clip(canvas, 0, 255).astype(np.uint8)


def normalize(img28: np.ndarray) -> np.ndarray:
    """uint8 28x28 -> float32 (1, 1, 28, 28) with the same normalisation used in training."""
    x = img28.astype(np.float32) / 255.0
    x = (x - MNIST_MEAN) / MNIST_STD
    return x[None, None, :, :]
