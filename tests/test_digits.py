import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from digits.preprocess import normalize, to_ink, to_mnist  # noqa: E402


# ---------- helpers: fake canvas drawings ----------------------------------------
def draw_one(size=280, offset=(0, 0), mode="RGBA", thickness=20):
    """A '1'-like vertical stroke drawn in black, the way the Gradio Sketchpad sends it."""
    bg = (0, 0, 0, 0) if mode == "RGBA" else "white"
    img = Image.new(mode, (size, size), bg)
    d = ImageDraw.Draw(img)
    x = size // 2 + offset[0]
    y0, y1 = size // 4 + offset[1], 3 * size // 4 + offset[1]
    d.line([(x, y0), (x, y1)], fill="black", width=thickness)
    return np.asarray(img)


def draw_zero(size=280, box=(60, 40, 220, 240), mode="RGBA"):
    bg = (0, 0, 0, 0) if mode == "RGBA" else "white"
    img = Image.new(mode, (size, size), bg)
    ImageDraw.Draw(img).ellipse(box, outline="black", width=20)
    return np.asarray(img)


def centre_of_mass(img):
    yy, xx = np.indices(img.shape)
    t = img.sum()
    return (yy * img).sum() / t, (xx * img).sum() / t


# ---------- preprocessing (no PyTorch needed) ---------------------------------------
@pytest.mark.parametrize("mode", ["RGBA", "RGB", "L"])
def test_ink_is_high_where_drawn(mode):
    ink = to_ink(draw_one(mode=mode))
    assert ink.shape == (280, 280)
    assert ink[140, 140] > 200      # on the stroke
    assert ink[10, 10] < 5          # background


def test_white_on_black_input_not_inverted():
    img = np.zeros((100, 100), dtype=np.uint8)
    img[20:80, 45:55] = 255
    ink = to_ink(img)
    assert ink[50, 50] == 255 and ink[0, 0] == 0


def test_empty_canvas_returns_none():
    assert to_mnist(np.zeros((280, 280, 4), dtype=np.uint8)) is None
    assert to_mnist(np.full((280, 280, 3), 255, dtype=np.uint8)) is None


@pytest.mark.parametrize("offset", [(0, 0), (-90, -50), (80, 60)])
def test_output_is_mnist_style(offset):
    out = to_mnist(draw_one(offset=offset))
    assert out.shape == (28, 28) and out.dtype == np.uint8
    assert out.max() > 200                             # bright digit on dark background
    assert out[0].max() == 0 and out[:, 0].max() == 0  # empty border like MNIST
    cy, cx = centre_of_mass(out.astype(float))
    assert abs(cy - 13.5) <= 1 and abs(cx - 13.5) <= 1  # centred wherever it was drawn


def test_digit_fits_20px_box():
    out = to_mnist(draw_zero())
    rows = np.where((out > 30).any(axis=1))[0]
    cols = np.where((out > 30).any(axis=0))[0]
    assert rows[-1] - rows[0] + 1 <= 21 and cols[-1] - cols[0] + 1 <= 21
    assert max(rows[-1] - rows[0], cols[-1] - cols[0]) + 1 >= 18


def test_position_invariance():
    a = to_mnist(draw_zero(box=(20, 20, 120, 140))).astype(float)
    b = to_mnist(draw_zero(box=(150, 120, 250, 240))).astype(float)
    assert np.abs(a - b).mean() < 5  # same digit anywhere on the canvas -> ~same input


def test_normalize_shape_and_scale():
    x = normalize(np.zeros((28, 28), dtype=np.uint8))
    assert x.shape == (1, 1, 28, 28) and x.dtype == np.float32
    assert np.allclose(x, -0.1307 / 0.3081, atol=1e-5)


# ---------- model & inference (need PyTorch) -------------------------------------------
def test_model_shapes_and_size():
    torch = pytest.importorskip("torch")
    from digits.model import DigitCNN, count_parameters
    m = DigitCNN()
    assert m(torch.zeros(4, 1, 28, 28)).shape == (4, 10)
    assert count_parameters(m) < 1_000_000


def test_model_can_learn():
    torch = pytest.importorskip("torch")
    from digits.engine import train_one_epoch
    from digits.model import DigitCNN
    torch.manual_seed(0)
    x, y = torch.randn(64, 1, 28, 28), torch.randint(0, 10, (64,))
    loader = [(x, y)]
    m = DigitCNN()
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    first, _ = train_one_epoch(m, loader, opt, None, "cpu")
    for _ in range(30):
        last, _ = train_one_epoch(m, loader, opt, None, "cpu")
    assert last < first * 0.5  # can overfit a single batch


def test_predictor_roundtrip(tmp_path):
    pytest.importorskip("torch")
    from digits.inference import DigitPredictor, save_checkpoint
    from digits.model import DigitCNN
    path = tmp_path / "m.pt"
    save_checkpoint(DigitCNN(), path, epoch=0, val_acc=0.0)
    pred = DigitPredictor(path)
    probs, img = pred.predict(draw_one())
    assert set(probs) == {str(d) for d in range(10)}
    assert abs(sum(probs.values()) - 1) < 1e-4 and img.shape == (28, 28)
    assert pred.predict(np.zeros((280, 280, 4), dtype=np.uint8)) == ({}, None)
