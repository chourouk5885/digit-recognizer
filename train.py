"""Train the CNN on MNIST, evaluate on the test set and write a report.

    python train.py                 # 8 epochs, about 5-10 minutes on a laptop
    python train.py --epochs 3      # quicker run
    python train.py --device cpu    # force CPU (default: CUDA > Apple GPU (mps) > CPU)
    python train.py --eval-only     # skip training: re-evaluate the saved model and redraw the report

Outputs:
    models/mnist_cnn.pt             best checkpoint (by validation accuracy), used by app.py
    reports/metrics.json            test accuracy, per-class accuracy, training settings
    reports/history.csv             loss/accuracy per epoch
    reports/RESULTS.md              human-readable summary
    reports/figures/*.png           samples, curves, confusion matrix, per-class accuracy, mistakes
"""

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from digits import plots  # noqa: E402
from digits.data import get_loaders  # noqa: E402
from digits.engine import evaluate, pick_device, train_one_epoch  # noqa: E402
from digits.inference import load_model, save_checkpoint  # noqa: E402
from digits.model import DigitCNN, count_parameters  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="Train a CNN on MNIST")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-3, help="peak learning rate (one-cycle schedule)")
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--device", default="auto", help="auto | cpu | cuda | mps")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data-dir", default=str(ROOT / "data"))
    p.add_argument("--out", default=str(ROOT / "models" / "mnist_cnn.pt"))
    p.add_argument("--eval-only", action="store_true",
                   help="don't train; evaluate the saved checkpoint and regenerate reports/")
    return p.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main():
    args = parse_args()
    set_seed(args.seed)
    device = pick_device(args.device)
    reports = ROOT / "reports"
    figs = reports / "figures"

    print(f"Device: {device}")
    print("Loading MNIST (downloads ~12 MB the first time)...")
    train_loader, val_loader, test_loader = get_loaders(
        args.data_dir, args.batch_size, num_workers=args.num_workers, seed=args.seed,
        pin_memory=device.type == "cuda")
    print(f"train {len(train_loader.dataset):,} | val {len(val_loader.dataset):,} | test {len(test_loader.dataset):,}")

    if args.eval_only:
        if not Path(args.out).exists():
            sys.exit(f"No checkpoint at {args.out}. Train first with `python train.py`.")
        history = load_history(reports / "history.csv")
        _, best_acc = evaluate(load_model(args.out, device), val_loader, device)
        print("Eval-only: skipping training, using the saved checkpoint.")
    else:
        history, best_acc = train(args, device, train_loader, val_loader, figs)

    # ---- final evaluation on the untouched test set, with the best checkpoint ----
    model = load_model(args.out, device)
    te_loss, te_acc, preds, targets, probs = evaluate(model, test_loader, device, return_predictions=True)
    preds, targets, probs = preds.numpy(), targets.numpy(), probs.numpy()
    test_images = torch.cat([xb for xb, _ in test_loader]).numpy()
    per_class = {str(d): float((preds[targets == d] == d).mean()) for d in range(10)}
    n_err = int((preds != targets).sum())

    cm = np.zeros((10, 10), dtype=int)
    np.add.at(cm, (targets, preds), 1)
    np.fill_diagonal(cm, 0)
    top_pairs = sorted(((int(cm[i, j]), i, j) for i in range(10) for j in range(10) if cm[i, j]), reverse=True)[:5]

    plots.plot_history(history, figs)
    plots.plot_confusion(preds, targets, figs)
    plots.plot_per_class(preds, targets, figs)
    plots.plot_mistakes(test_images, preds, targets, probs, figs)

    metrics = {
        "test_accuracy": te_acc, "test_loss": te_loss, "test_errors": n_err, "test_size": int(len(targets)),
        "best_val_accuracy": best_acc, "per_class_accuracy": per_class,
        "most_confused": [{"true": i, "predicted": j, "count": c} for c, i, j in top_pairs],
        **vars(args), "parameters": count_parameters(model), "device": str(device),
    }
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "metrics.json").write_text(json.dumps(metrics, indent=2))
    with open(reports / "history.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(history[0]))
        w.writeheader()
        w.writerows(history)
    write_results_md(reports / "RESULTS.md", metrics, history)

    print(f"\nTest accuracy: {te_acc:.2%} ({n_err} errors out of {len(targets):,})")
    worst = min(per_class, key=per_class.get)
    print(f"Hardest digit: {worst} ({per_class[worst]:.2%})")
    print("Most confused pairs: " + ", ".join(f"{i}→{j} ({c})" for c, i, j in top_pairs))
    print(f"\nModel saved to {args.out}\nReport written to {reports}/\nNext: python app.py")


def train(args, device, train_loader, val_loader, figs):
    x, y = next(iter(train_loader))
    plots.plot_samples(x.numpy(), y.numpy(), figs)

    model = DigitCNN().to(device)
    print(f"Model parameters: {count_parameters(model):,}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, epochs=args.epochs, steps_per_epoch=len(train_loader))

    history, best_acc = [], -1.0
    print(f"\n{'epoch':>5} {'train_loss':>10} {'train_acc':>9} {'val_loss':>9} {'val_acc':>8} {'time':>6}")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, scheduler, device)
        va_loss, va_acc = evaluate(model, val_loader, device)
        history.append({"epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc,
                        "val_loss": va_loss, "val_acc": va_acc})
        mark = ""
        if va_acc > best_acc:
            best_acc = va_acc
            save_checkpoint(model, args.out, epoch=epoch, val_acc=va_acc)
            mark = "  * saved"
        print(f"{epoch:>5} {tr_loss:>10.4f} {tr_acc:>9.2%} {va_loss:>9.4f} {va_acc:>8.2%} "
              f"{time.time() - t0:>5.0f}s{mark}")

    return history, best_acc


def load_history(path: Path) -> list[dict]:
    if not path.exists():
        sys.exit(f"{path} not found - run a full training first.")
    with open(path) as f:
        return [{k: (int(v) if k == "epoch" else float(v)) for k, v in row.items()} for row in csv.DictReader(f)]


def write_results_md(path: Path, m: dict, history: list[dict]):
    pc = m["per_class_accuracy"]
    lines = [
        "# Results", "", "_Auto-generated by `python train.py`._", "",
        f"- **Test accuracy: {m['test_accuracy']:.2%}** ({m['test_errors']} errors out of {m['test_size']:,} images)",
        f"- Best validation accuracy: {m['best_val_accuracy']:.2%}",
        f"- Parameters: {m['parameters']:,} · epochs: {m['epochs']} · batch size: {m['batch_size']} · device: {m['device']}",
        "", "## Accuracy per digit", "", "| Digit | " + " | ".join(str(d) for d in range(10)) + " |",
        "|---|" + "---:|" * 10, "| Accuracy | " + " | ".join(f"{100 * pc[str(d)]:.2f}%" for d in range(10)) + " |",
        "", "## Most confused pairs", "",
    ]
    lines += [f"- true **{p['true']}** predicted as **{p['predicted']}**: {p['count']} times" for p in m["most_confused"]]
    lines += ["", "## Training history", "", "| Epoch | Train loss | Train acc | Val loss | Val acc |",
              "|---:|---:|---:|---:|---:|"]
    lines += [f"| {h['epoch']} | {h['train_loss']:.4f} | {h['train_acc']:.2%} | {h['val_loss']:.4f} | {h['val_acc']:.2%} |"
              for h in history]
    lines += ["", "## Figures", "", "![Curves](figures/02_training_curves.png)",
              "![Confusion](figures/03_confusion_matrix.png)", "![Mistakes](figures/05_mistakes.png)", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
