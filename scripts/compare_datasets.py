"""
Compares old dataset vs new dataset using the SAME model (small/original,
channels=[16,32,64,64]) so that model is the fixed variable and dataset
is the only thing that changes.

New dataset structure:
  final_8k_without_noise/<class>/<variant>/<file>.wav
  e.g. aviation/clean/aviation_clean_1.wav

Old dataset structure:
  recordings/<classNN>.wav  (flat, class name = stem with trailing digits stripped)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np

from config import AudioConfig, ModelConfig, TrainConfig
from audio_loader import AudioLoader
from spectrogram import SpectrogramConverter
from augmentation import Augmenter
from drink_dataset import DrinkDataset
from model import SmallResNet
from trainer import Trainer


# Fixed: use original small model
SMALL_CHANNELS = [16, 32, 64, 64]

NEW_DATASET_PATH = Path("/new_dataset")


class SubdirDrinkDataset(Dataset):
    """Loads the new dataset where files live in class/variant/ subfolders."""

    _COPIES = 5

    def __init__(self, files, labels, audio_loader, spec_converter, augmenter, augment=False):
        self._files = files
        self._labels = labels
        self._loader = audio_loader
        self._converter = spec_converter
        self._augmenter = augmenter
        self._augment = augment

    def __len__(self):
        return len(self._files) * self._COPIES if self._augment else len(self._files)

    def __getitem__(self, idx):
        if self._augment:
            file_idx, aug_idx = divmod(idx, self._COPIES)
        else:
            file_idx, aug_idx = idx, 0

        pcm = self._loader.load(self._files[file_idx])
        y = pcm.astype(np.float32) / 32768.0

        if aug_idx > 0:
            y = self._augmenter.augment_waveform(y)

        pcm_out = (y * 32767).clip(-32768, 32767).astype(np.int16)
        mel = self._converter.convert(pcm_out)
        mel_t = torch.from_numpy(mel)

        if aug_idx > 0:
            mel_t = self._augmenter.augment_spectrogram(mel_t)

        return mel_t, self._labels[file_idx]

    @staticmethod
    def make_splits(root: Path, audio_loader, spec_converter, augmenter):
        # class name = top-level subfolder name
        all_classes: dict[str, list[Path]] = {}
        for class_dir in sorted(root.iterdir()):
            if not class_dir.is_dir():
                continue
            files = sorted(class_dir.rglob("*.wav"))
            if files:
                all_classes[class_dir.name] = files

        class_names = sorted(all_classes)
        class_to_idx = {c: i for i, c in enumerate(class_names)}

        train_files, train_labels = [], []
        val_files, val_labels = [], []

        for cls, files in all_classes.items():
            n_val = max(1, len(files) // 5)
            for i, f in enumerate(files):
                if i < n_val:
                    val_files.append(f)
                    val_labels.append(class_to_idx[cls])
                else:
                    train_files.append(f)
                    train_labels.append(class_to_idx[cls])

        train_ds = SubdirDrinkDataset(train_files, train_labels, audio_loader,
                                      spec_converter, augmenter, augment=True)
        val_ds = SubdirDrinkDataset(val_files, val_labels, audio_loader,
                                    spec_converter, augmenter, augment=False)
        return train_ds, val_ds, class_names


def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total if total > 0 else 0.0


def run(label, train_loader, val_loader, n_classes, train_cfg, model_cfg, device):
    print(f"\n{'='*55}")
    print(f"  Dataset: {label}")
    print(f"{'='*55}")

    # We don't load base.pt here because it was saved with larger channels.
    # Both runs start from the same random init so the comparison is still fair.
    model = SmallResNet(n_classes=n_classes,
                        channels=SMALL_CHANNELS).to(device)
    trainer = Trainer(model, device, train_cfg)

    model.fc = nn.Linear(model.n_features, n_classes)
    nn.init.xavier_uniform_(model.fc.weight)
    nn.init.zeros_(model.fc.bias)
    model = model.to(device)

    tmp_ckpt = train_cfg.artifacts_dir / f"tmp_{label.replace(' ', '_')}.pt"

    print("  Stage A: head warmup...")
    trainer.train(train_loader, val_loader, checkpoint_path=tmp_ckpt,
                  lr=train_cfg.lr_finetune_head,
                  epochs=train_cfg.finetune_warmup_epochs,
                  freeze_backbone=True)

    print("  Stage B: full fine-tune...")
    history = trainer.train(train_loader, val_loader, checkpoint_path=tmp_ckpt,
                            lr=train_cfg.lr_finetune_full,
                            epochs=train_cfg.finetune_epochs,
                            freeze_backbone=False)

    trainer.load_checkpoint(tmp_ckpt)
    final_acc = evaluate(model, val_loader, device)
    best_acc = max(history["val_acc"])
    tmp_ckpt.unlink(missing_ok=True)

    print(f"\n  >> Best val acc : {best_acc:.1%}")
    print(f"  >> Final val acc: {final_acc:.1%}")
    return best_acc, final_acc


def main():
    audio_cfg = AudioConfig()
    model_cfg = ModelConfig()
    train_cfg = TrainConfig()

    audio_loader = AudioLoader(audio_cfg)
    spec_converter = SpectrogramConverter(audio_cfg)
    augmenter = Augmenter(audio_cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Model: SmallResNet channels={SMALL_CHANNELS}  (fixed)")

    # --- Old dataset ---
    print("\nLoading OLD dataset (recordings/)...")
    old_train, old_val, old_classes = DrinkDataset.make_splits(
        train_cfg.data_dir, audio_loader, spec_converter, augmenter
    )
    print(f"  Classes: {old_classes}")
    print(f"  Train: {len(old_train)}  Val: {len(old_val)}")
    old_train_loader = DataLoader(old_train, batch_size=16, shuffle=True, num_workers=2)
    old_val_loader = DataLoader(old_val, batch_size=16, shuffle=False, num_workers=2)

    # --- New dataset ---
    print("\nLoading NEW dataset (final_8k_without_noise)...")
    new_train, new_val, new_classes = SubdirDrinkDataset.make_splits(
        NEW_DATASET_PATH, audio_loader, spec_converter, augmenter
    )
    print(f"  Classes: {new_classes}")
    print(f"  Train: {len(new_train)}  Val: {len(new_val)}")
    new_train_loader = DataLoader(new_train, batch_size=16, shuffle=True, num_workers=2)
    new_val_loader = DataLoader(new_val, batch_size=16, shuffle=False, num_workers=2)

    # Run both
    old_best, old_final = run("old dataset", old_train_loader, old_val_loader,
                               len(old_classes), train_cfg, model_cfg, device)
    new_best, new_final = run("new dataset", new_train_loader, new_val_loader,
                               len(new_classes), train_cfg, model_cfg, device)

    # Summary
    print(f"\n\n{'='*55}")
    print("  DATASET COMPARISON SUMMARY  (model fixed: small)")
    print(f"{'='*55}")
    print(f"  {'Dataset':<20} {'Files/class':>11}  {'Best val':>9}  {'Final val':>9}")
    print(f"  {'-'*52}")
    print(f"  {'Old (flat ~25/cls)':<20} {'~25':>11}  {old_best:>9.1%}  {old_final:>9.1%}")
    print(f"  {'New (60/cls)':<20} {'60':>11}  {new_best:>9.1%}  {new_final:>9.1%}")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
