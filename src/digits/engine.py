"""Training and evaluation loops."""

import torch
from torch import nn


def pick_device(preferred: str = "auto") -> torch.device:
    if preferred != "auto":
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")  # Apple Silicon GPU
    return torch.device("cpu")


def train_one_epoch(model, loader, optimizer, scheduler, device) -> tuple[float, float]:
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    total_loss, correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        total_loss += loss.item() * len(y)
        correct += (logits.argmax(1) == y).sum().item()
        n += len(y)
    return total_loss / n, correct / n


@torch.no_grad()
def evaluate(model, loader, device, return_predictions: bool = False):
    """Return (loss, accuracy) or, with return_predictions, (loss, accuracy, preds, targets, probs)."""
    model.eval()
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    total_loss, correct, n = 0.0, 0, 0
    preds, targets, probs = [], [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total_loss += loss_fn(logits, y).item()
        p = logits.argmax(1)
        correct += (p == y).sum().item()
        n += len(y)
        if return_predictions:
            preds.append(p.cpu())
            targets.append(y.cpu())
            probs.append(torch.softmax(logits, 1).cpu())
    if return_predictions:
        return total_loss / n, correct / n, torch.cat(preds), torch.cat(targets), torch.cat(probs)
    return total_loss / n, correct / n
