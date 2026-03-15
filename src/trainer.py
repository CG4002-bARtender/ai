from __future__ import annotations
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torch.utils.data import DataLoader

from config import TrainConfig


class Trainer:
    def __init__(self, model: nn.Module, device: torch.device, cfg: TrainConfig):
        self._model = model
        self._device = device
        self._cfg = cfg

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        checkpoint_path: Path,
        lr: float,
        epochs: int,
        freeze_backbone: bool = False,
    ) -> dict:
        checkpoint_path = Path(checkpoint_path)
        model = self._model
        cfg = self._cfg

        if freeze_backbone:
            for name, param in model.named_parameters():
                param.requires_grad = name.startswith("fc.")
        else:
            for param in model.parameters():
                param.requires_grad = True

        criterion = nn.CrossEntropyLoss()
        optimiser = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr,
            weight_decay=cfg.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=epochs)

        history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(1, epochs + 1):
            t_loss, t_acc = self._run_epoch(train_loader, criterion, optimiser, train=True)
            v_loss, v_acc = self._run_epoch(val_loader, criterion, None, train=False)
            scheduler.step()

            history["train_loss"].append(t_loss)
            history["val_loss"].append(v_loss)
            history["train_acc"].append(t_acc)
            history["val_acc"].append(v_acc)

            print(
                f"Epoch {epoch:3d}/{epochs}  "
                f"train loss={t_loss:.4f} acc={t_acc:.3f}  "
                f"val loss={v_loss:.4f} acc={v_acc:.3f}"
            )

            if v_loss < best_val_loss:
                best_val_loss = v_loss
                patience_counter = 0
                torch.save({"state_dict": model.state_dict()}, checkpoint_path)
                print(f"  -> New best, saved {checkpoint_path}")
            else:
                patience_counter += 1
                if patience_counter >= cfg.patience:
                    print(f"  Early stopping at epoch {epoch}")
                    break

        return history

    def _run_epoch(
        self,
        loader: DataLoader,
        criterion: nn.Module,
        optimiser,
        train: bool,
    ) -> tuple[float, float]:
        self._model.train(train)
        total_loss, correct, total = 0.0, 0, 0
        with torch.set_grad_enabled(train):
            for x, y in loader:
                x, y = x.to(self._device), y.to(self._device)
                logits = self._model(x)
                loss = criterion(logits, y)
                if train:
                    optimiser.zero_grad()
                    loss.backward()
                    optimiser.step()
                total_loss += loss.item() * len(y)
                correct += (logits.argmax(1) == y).sum().item()
                total += len(y)
        return total_loss / total, correct / total

    def load_checkpoint(self, path: Path) -> nn.Module:
        path = Path(path)
        ckpt = torch.load(path, map_location=self._device)
        state = ckpt.get("state_dict", ckpt.get("model", ckpt))
        self._model.load_state_dict(state)
        return self._model

    def save_curve(self, history: dict, path: Path):
        path = Path(path)
        n = len(history["train_loss"])
        epochs = range(1, n + 1)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        ax1.plot(epochs, history["train_loss"], label="train")
        ax1.plot(epochs, history["val_loss"], label="val")
        ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss"); ax1.set_title("Loss"); ax1.legend()
        ax2.plot(epochs, history["train_acc"], label="train")
        ax2.plot(epochs, history["val_acc"], label="val")
        ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy"); ax2.set_title("Accuracy"); ax2.legend()
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"Saved curve -> {path}")

    def save_confusion_matrix(self, val_loader: DataLoader, class_names: list[str], path: Path):
        path = Path(path)
        self._model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for x, y in val_loader:
                preds = self._model(x.to(self._device)).argmax(1).cpu()
                all_preds.extend(preds.tolist())
                all_labels.extend(y.tolist())
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        overall_acc = (all_preds == all_labels).mean()

        for i, cls in enumerate(class_names):
            mask = all_labels == i
            if mask.sum() == 0:
                continue
            cls_acc = (all_preds[mask] == all_labels[mask]).mean()
            print(f"  {cls:<16s}  {cls_acc:.2%}  ({mask.sum()} samples)")
        print(f"  Overall val accuracy: {overall_acc:.2%}")

        cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(class_names))))
        fig, ax = plt.subplots(figsize=(9, 8))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        disp.plot(ax=ax, xticks_rotation=45, colorbar=False)
        ax.set_title("Confusion Matrix (val)")
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"Saved confusion matrix -> {path}")
