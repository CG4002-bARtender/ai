"""
Cross-dataset evaluation — answers: does training on the new dataset
generalise better than training on the old one?

Two experiments, same small model (fixed variable):

  Exp A: train on OLD  → evaluate on NEW (held-out)
  Exp B: train on NEW  → evaluate on OLD (held-out)

If Exp B val acc > Exp A val acc, the new dataset produces a more
generalisable model.

Class name mapping between datasets:
  old flat names  : irishcoffee, midorisour, oldfashioned, scotchneat,
                    vodkaneat, whiskeyneat
  new folder names: irish_coffee, midori_sour, old_fashioned, scotch_neat,
                    vodka_neat, whiskey_neat

We normalise both to a canonical set so labels line up.
"""
import sys
from pathlib import Path
import re

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

SMALL_CHANNELS = [16, 32, 64, 64]

NEW_DATASET_PATH = Path("/new_dataset")

# Map both naming conventions to a single canonical name
CANONICAL = {
    # old flat names
    "aviation": "aviation",
    "godfather": "godfather",
    "irishcoffee": "irishcoffee",
    "martini": "martini",
    "midorisour": "midorisour",
    "oldfashioned": "oldfashioned",
    "scotchneat": "scotchneat",
    "tuxedo": "tuxedo",
    "vodkaneat": "vodkaneat",
    "whiskeyneat": "whiskeyneat",
    # new folder names → same canonical
    "irish_coffee": "irishcoffee",
    "midori_sour": "midorisour",
    "old_fashioned": "oldfashioned",
    "scotch_neat": "scotchneat",
    "vodka_neat": "vodkaneat",
    "whiskey_neat": "whiskeyneat",
}

CANONICAL_CLASSES = sorted(set(CANONICAL.values()))
CLASS_TO_IDX = {c: i for i, c in enumerate(CANONICAL_CLASSES)}


class FlatDataset(Dataset):
    """Old dataset: recordings/<classNN>.wav"""
    def __init__(self, files, labels, audio_loader, spec_converter, augmenter, augment=False):
        self._files = files
        self._labels = labels
        self._loader = audio_loader
        self._converter = spec_converter
        self._augmenter = augmenter
        self._augment = augment
        self._copies = 5

    def __len__(self):
        return len(self._files) * self._copies if self._augment else len(self._files)

    def __getitem__(self, idx):
        file_idx, aug_idx = divmod(idx, self._copies) if self._augment else (idx, 0)
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
    def load_all(data_dir, audio_loader, spec_converter, augmenter, augment=False):
        files, labels = [], []
        for f in sorted(data_dir.glob("*.wav")):
            raw_cls = re.sub(r"\d+$", "", f.stem)
            canon = CANONICAL.get(raw_cls)
            if canon is None:
                print(f"  WARNING: unknown class '{raw_cls}' in {f.name}, skipping")
                continue
            files.append(f)
            labels.append(CLASS_TO_IDX[canon])
        return FlatDataset(files, labels, audio_loader, spec_converter, augmenter, augment)


class SubdirDataset(Dataset):
    """New dataset: <class>/<variant>/<file>.wav"""
    def __init__(self, files, labels, audio_loader, spec_converter, augmenter, augment=False):
        self._files = files
        self._labels = labels
        self._loader = audio_loader
        self._converter = spec_converter
        self._augmenter = augmenter
        self._augment = augment
        self._copies = 5

    def __len__(self):
        return len(self._files) * self._copies if self._augment else len(self._files)

    def __getitem__(self, idx):
        file_idx, aug_idx = divmod(idx, self._copies) if self._augment else (idx, 0)
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
    def load_all(root, audio_loader, spec_converter, augmenter, augment=False):
        files, labels = [], []
        for class_dir in sorted(root.iterdir()):
            if not class_dir.is_dir():
                continue
            canon = CANONICAL.get(class_dir.name)
            if canon is None:
                print(f"  WARNING: unknown class '{class_dir.name}', skipping")
                continue
            for f in sorted(class_dir.rglob("*.wav")):
                files.append(f)
                labels.append(CLASS_TO_IDX[canon])
        return SubdirDataset(files, labels, audio_loader, spec_converter, augmenter, augment)


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


def train_model(train_loader, val_loader, n_classes, train_cfg, device, label):
    print(f"\n  Training on {label}...")
    model = SmallResNet(n_classes=n_classes, channels=SMALL_CHANNELS).to(device)
    trainer = Trainer(model, device, train_cfg)
    tmp = train_cfg.artifacts_dir / f"tmp_cross_{label.replace(' ', '_')}.pt"

    trainer.train(train_loader, val_loader, checkpoint_path=tmp,
                  lr=train_cfg.lr_finetune_head,
                  epochs=train_cfg.finetune_warmup_epochs,
                  freeze_backbone=True)

    history = trainer.train(train_loader, val_loader, checkpoint_path=tmp,
                            lr=train_cfg.lr_finetune_full,
                            epochs=train_cfg.finetune_epochs,
                            freeze_backbone=False)

    trainer.load_checkpoint(tmp)
    tmp.unlink(missing_ok=True)
    best_own_val = max(history["val_acc"])
    return model, best_own_val


def main():
    audio_cfg = AudioConfig()
    train_cfg = TrainConfig()

    audio_loader = AudioLoader(audio_cfg)
    spec_converter = SpectrogramConverter(audio_cfg)
    augmenter = Augmenter(audio_cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_classes = len(CANONICAL_CLASSES)

    print(f"Device: {device}")
    print(f"Model: SmallResNet channels={SMALL_CHANNELS}  (fixed)")
    print(f"Classes ({n_classes}): {CANONICAL_CLASSES}")

    # Load all files from both datasets
    print("\nLoading OLD dataset...")
    old_ds = FlatDataset.load_all(train_cfg.data_dir, audio_loader, spec_converter, augmenter)
    old_ds_aug = FlatDataset.load_all(train_cfg.data_dir, audio_loader, spec_converter, augmenter, augment=True)
    print(f"  {len(old_ds)} files total")

    print("\nLoading NEW dataset...")
    new_ds = SubdirDataset.load_all(NEW_DATASET_PATH, audio_loader, spec_converter, augmenter)
    new_ds_aug = SubdirDataset.load_all(NEW_DATASET_PATH, audio_loader, spec_converter, augmenter, augment=True)
    print(f"  {len(new_ds)} files total")

    old_train_loader = DataLoader(old_ds_aug, batch_size=16, shuffle=True, num_workers=2)
    old_test_loader = DataLoader(old_ds, batch_size=16, shuffle=False, num_workers=2)
    new_train_loader = DataLoader(new_ds_aug, batch_size=16, shuffle=True, num_workers=2)
    new_test_loader = DataLoader(new_ds, batch_size=16, shuffle=False, num_workers=2)

    # Exp A: train on OLD, test on NEW
    print(f"\n{'='*55}")
    print("  Exp A: train on OLD → test on NEW")
    print(f"{'='*55}")
    model_a, own_val_a = train_model(old_train_loader, old_test_loader,
                                      n_classes, train_cfg, device, "OLD")
    cross_acc_a = evaluate(model_a, new_test_loader, device)
    print(f"  Own val acc (old→old): {own_val_a:.1%}")
    print(f"  Cross acc  (old→new): {cross_acc_a:.1%}")

    # Exp B: train on NEW, test on OLD
    print(f"\n{'='*55}")
    print("  Exp B: train on NEW → test on OLD")
    print(f"{'='*55}")
    model_b, own_val_b = train_model(new_train_loader, new_test_loader,
                                      n_classes, train_cfg, device, "NEW")
    cross_acc_b = evaluate(model_b, old_test_loader, device)
    print(f"  Own val acc (new→new): {own_val_b:.1%}")
    print(f"  Cross acc  (new→old): {cross_acc_b:.1%}")

    # Summary
    print(f"\n\n{'='*55}")
    print("  CROSS-DATASET EVALUATION SUMMARY")
    print(f"{'='*55}")
    print(f"  {'Experiment':<30} {'Own val':>8}  {'Cross acc':>10}")
    print(f"  {'-'*52}")
    print(f"  {'Train OLD → Test NEW':<30} {own_val_a:>8.1%}  {cross_acc_a:>10.1%}")
    print(f"  {'Train NEW → Test OLD':<30} {own_val_b:>8.1%}  {cross_acc_b:>10.1%}")
    print(f"{'='*55}")
    print()
    if cross_acc_b > cross_acc_a:
        print("  => Training on NEW dataset generalises BETTER.")
        print("     Switch to the new dataset for the final model.")
    elif cross_acc_b < cross_acc_a:
        print("  => Training on OLD dataset generalises BETTER.")
        print("     The new dataset may be too clean / not representative.")
    else:
        print("  => Both datasets generalise equally well.")
    print()


if __name__ == "__main__":
    main()
