"""Figures for the training report."""

from __future__ import annotations

from pathlib import Path

import matplotlib

try:  # keep the inline backend in Jupyter, use a headless one in scripts
    get_ipython  # type: ignore[name-defined]  # noqa: B018
except NameError:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BLUE, ORANGE, RED = "#2a78d6", "#eb6834", "#e34948"
BLUES = LinearSegmentedColormap.from_list("blues", ["#f4f8fd", "#9ec5f4", "#2a78d6", "#0d366b"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": GRID,
    "axes.labelcolor": INK_2, "axes.titlecolor": INK, "axes.titlesize": 12,
    "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "xtick.color": INK_2, "ytick.color": INK_2, "font.size": 10,
    "legend.frameon": False, "savefig.dpi": 150, "savefig.bbox": "tight",
})


def _save(fig, outdir: Path | None, name: str):
    if outdir is not None:
        outdir.mkdir(parents=True, exist_ok=True)
        fig.savefig(outdir / f"{name}.png")
    return fig


def _unnormalize(x: np.ndarray) -> np.ndarray:
    from .constants import MNIST_MEAN, MNIST_STD
    return np.clip(x * MNIST_STD + MNIST_MEAN, 0, 1)


def plot_samples(images: np.ndarray, labels: np.ndarray, outdir: Path | None = None, n: int = 32):
    """Grid of training images (already normalised tensors as numpy, shape (N, 1, 28, 28))."""
    cols = 8
    rows = int(np.ceil(min(n, len(images)) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.1, rows * 1.25))
    for ax, img, lab in zip(axes.flat, images[:n], labels[:n]):
        ax.imshow(_unnormalize(img[0]), cmap="gray")
        ax.set_title(str(int(lab)), fontsize=10, loc="center", fontweight="normal")
        ax.axis("off")
    for ax in list(axes.flat)[min(n, len(images)):]:
        ax.axis("off")
    fig.suptitle("Training samples (with augmentation)", x=0.01, ha="left", fontweight="bold", color=INK)
    fig.tight_layout()
    return _save(fig, outdir, "01_samples")


def plot_history(history: list[dict], outdir: Path | None = None):
    ep = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    ax = axes[0]
    ax.plot(ep, [h["train_loss"] for h in history], color=BLUE, lw=2, marker="o", ms=5, label="train")
    ax.plot(ep, [h["val_loss"] for h in history], color=ORANGE, lw=2, marker="s", ms=5, label="validation")
    ax.set_title("Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-entropy")
    ax.legend()
    ax = axes[1]
    ax.plot(ep, [100 * h["train_acc"] for h in history], color=BLUE, lw=2, marker="o", ms=5, label="train")
    ax.plot(ep, [100 * h["val_acc"] for h in history], color=ORANGE, lw=2, marker="s", ms=5, label="validation")
    ax.set_title("Accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.legend(loc="lower right")
    for a in axes:
        a.set_xticks(ep)
    fig.tight_layout()
    return _save(fig, outdir, "02_training_curves")


def plot_confusion(preds: np.ndarray, targets: np.ndarray, outdir: Path | None = None):
    cm = np.zeros((10, 10), dtype=int)
    np.add.at(cm, (targets, preds), 1)
    off = cm.copy()
    np.fill_diagonal(off, 0)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    ax.imshow(cm, cmap=BLUES)
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center", fontsize=8,
                color="white" if v > cm.max() * 0.6 else INK)
    ax.set_title(f"Confusion matrix (test, {len(targets):,} images)")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    ax = axes[1]
    ax.imshow(off, cmap=LinearSegmentedColormap.from_list("err", [SURFACE, "#f6c7b3", ORANGE, "#9c3b13"]))
    for (i, j), v in np.ndenumerate(off):
        if v:
            ax.text(j, i, str(v), ha="center", va="center", fontsize=9,
                    color="white" if v > max(off.max(), 1) * 0.6 else INK)
    ax.set_title("Errors only (where the model gets confused)")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for a in axes:
        a.set_xticks(range(10))
        a.set_yticks(range(10))
        a.grid(False)
    fig.tight_layout()
    return _save(fig, outdir, "03_confusion_matrix")


def plot_per_class(preds: np.ndarray, targets: np.ndarray, outdir: Path | None = None):
    acc = np.array([(preds[targets == d] == d).mean() * 100 for d in range(10)])
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(range(10), acc, color=BLUE, width=0.7)
    for d, a in enumerate(acc):
        ax.text(d, a + 0.02, f"{a:.2f}%", ha="center", va="bottom", fontsize=8, color=INK_2)
    lo = max(90.0, np.floor(acc.min()) - 1)
    ax.set_ylim(lo, 100.3)
    ax.set_xticks(range(10))
    ax.set_xlabel("Digit")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"Test accuracy per digit (y-axis starts at {lo:.0f}%)")
    ax.grid(axis="x", visible=False)
    return _save(fig, outdir, "04_per_class_accuracy")


def plot_mistakes(images: np.ndarray, preds: np.ndarray, targets: np.ndarray, probs: np.ndarray,
                  outdir: Path | None = None, n: int = 24):
    wrong = np.where(preds != targets)[0]
    # most confident mistakes first: the interesting ones
    wrong = wrong[np.argsort(-probs[wrong, preds[wrong]])][:n]
    if len(wrong) == 0:
        return None
    cols = 8
    rows = int(np.ceil(len(wrong) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.4, rows * 2.0), squeeze=False)
    for ax, i in zip(axes.flat, wrong):
        ax.imshow(_unnormalize(images[i, 0]), cmap="gray")
        ax.set_title(f"true {targets[i]} → {preds[i]}\n{probs[i, preds[i]]:.0%}", fontsize=8,
                     loc="center", fontweight="normal", color=INK)
        ax.axis("off")
    for ax in list(axes.flat)[len(wrong):]:
        ax.axis("off")
    fig.suptitle("Most confident mistakes on the test set (true → predicted, confidence)",
                 x=0.01, ha="left", fontweight="bold", color=INK)
    fig.tight_layout(h_pad=1.5)
    return _save(fig, outdir, "05_mistakes")
