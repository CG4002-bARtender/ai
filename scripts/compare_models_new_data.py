"""
Compares small vs large model trained on combined new datasets
(without_noise + with_noise), cross-evaluated on old dataset.

Fixed variable : dataset (combined new)
Changing variable: model size (small vs large)
Cross-eval on old dataset to measure real-world generalisation.
"""
import sys
import re
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import numpy as np

from config import AudioConfig, ModelConfig, TrainConfig
from audio_loader import AudioLoader
from spectrogram import SpectrogramConverter
from augmentation import Augmenter
from model import SmallResNet
from trainer import Trainer

WITHOUT_NOISE_PATH = Path("/new_dataset")
WITH_NOISE_PATH    = Path("/new_dataset_noise")
OLD_DATA_PATH      = Path("recordings")  # inside /workspace

CANONICAL = {
    "aviation": "aviation", "godfather": "godfather",
    "irishcoffee": "irishcoffee", "martini": "martini",
    "midorisour": "midorisour", "oldfashioned": "oldfashioned",
    "scotchneat": "scotchneat", "tuxedo": "tuxedo",
    "vodkaneat": "vodkaneat", "whiskeyneat": "whiskeyneat",
    "irish_coffee": "irishcoffee", "midori_sour": "midorisour",
    "old_fashioned": "oldfashioned", "scotch_neat": "scotchneat",
    "vodka_neat": "vodkaneat", "whiskey_neat": "whiskeyneat",
}
CANONICAL_CLASSES = sorted(set(CANONICAL.values()))
CLASS_TO_IDX = {c: i for i, c in enumerate(CANONICAL_CLASSES)}
N_CLASSES = len(CANONICAL_CLASSES)

MODELS = {
    "small  [16,32,64,64]  ": [16, 32, 64, 64],
    "larger [32,64,128,128]": [32, 64, 128, 128],
}


class AudioDataset(Dataset):
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
        file_idx, aug_idx = divmod(idx, self._COPIES) if self._augment else (idx, 0)
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


def load_subdir(root, audio_loader, spec_converter, augmenter, augment=False):
    """Load dataset where files live in class/variant/ subfolders."""
    files, labels = [], []
    for class_dir in sorted(root.iterdir()):
        if not class_dir.is_dir():
            continue
        canon = CANONICAL.get(class_dir.name)
        if canon is None:
            continue
        for f in sorted(class_dir.rglob("*.wav")):
            files.append(f)
            labels.append(CLASS_TO_IDX[canon])
    return AudioDataset(files, labels, audio_loader, spec_converter, augmenter, augment)


def load_flat(data_dir, audio_loader, spec_converter, augmenter, augment=False):
    """Load old flat dataset: recordings/<classNN>.wav"""
    files, labels = [], []
    for f in sorted(data_dir.glob("*.wav")):
        raw_cls = re.sub(r"\d+$", "", f.stem)
        canon = CANONICAL.get(raw_cls)
        if canon is None:
            continue
        files.append(f)
        labels.append(CLASS_TO_IDX[canon])
    return AudioDataset(files, labels, audio_loader, spec_converter, augmenter, augment)


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


def train_and_eval(name, channels, train_loader, val_loader, cross_loader,
                   train_cfg, device):
    print(f"\n{'='*58}")
    print(f"  Model: {name}  params={sum(p.numel() for p in SmallResNet(N_CLASSES, channels).parameters()):,}")
    print(f"{'='*58}")

    model = SmallResNet(n_classes=N_CLASSES, channels=channels).to(device)
    trainer = Trainer(model, device, train_cfg)
    tmp = train_cfg.artifacts_dir / f"tmp_{name.strip().replace(' ','_')}.pt"

    print("  Stage A: head warmup...")
    trainer.train(train_loader, val_loader, checkpoint_path=tmp,
                  lr=train_cfg.lr_finetune_head,
                  epochs=train_cfg.finetune_warmup_epochs,
                  freeze_backbone=True)

    print("  Stage B: full fine-tune...")
    history = trainer.train(train_loader, val_loader, checkpoint_path=tmp,
                            lr=train_cfg.lr_finetune_full,
                            epochs=train_cfg.finetune_epochs,
                            freeze_backbone=False)

    trainer.load_checkpoint(tmp)
    tmp.unlink(missing_ok=True)

    own_val  = evaluate(model, val_loader, device)
    cross_acc = evaluate(model, cross_loader, device)
    best_val = max(history["val_acc"])

    print(f"\n  >> Best val acc (new data)  : {best_val:.1%}")
    print(f"  >> Final val acc (new data) : {own_val:.1%}")
    print(f"  >> Cross acc (old data)     : {cross_acc:.1%}")
    return best_val, own_val, cross_acc


def main():
    audio_cfg  = AudioConfig()
    train_cfg  = TrainConfig()
    audio_loader   = AudioLoader(audio_cfg)
    spec_converter = SpectrogramConverter(audio_cfg)
    augmenter      = Augmenter(audio_cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Classes ({N_CLASSES}): {CANONICAL_CLASSES}")

    # --- Combined new training data ---
    print("\nLoading without_noise dataset...")
    ds_clean     = load_subdir(WITHOUT_NOISE_PATH, audio_loader, spec_converter, augmenter, augment=True)
    ds_clean_val = load_subdir(WITHOUT_NOISE_PATH, audio_loader, spec_converter, augmenter, augment=False)

    print("Loading with_noise dataset...")
    ds_noise     = load_subdir(WITH_NOISE_PATH, audio_loader, spec_converter, augmenter, augment=True)
    ds_noise_val = load_subdir(WITH_NOISE_PATH, audio_loader, spec_converter, augmenter, augment=False)

    combined_train = ConcatDataset([ds_clean, ds_noise])
    combined_val   = ConcatDataset([ds_clean_val, ds_noise_val])
    print(f"Combined train: {len(combined_train)} samples  Val: {len(combined_val)} samples")

    train_loader = DataLoader(combined_train, batch_size=32, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(combined_val,   batch_size=32, shuffle=False, num_workers=2)

    # --- Old dataset for cross-eval ---
    print("Loading OLD dataset for cross-eval...")
    old_ds = load_flat(OLD_DATA_PATH, audio_loader, spec_converter, augmenter, augment=False)
    cross_loader = DataLoader(old_ds, batch_size=16, shuffle=False, num_workers=2)
    print(f"Old dataset: {len(old_ds)} samples")

    # --- Run both models ---
    results = {}
    for name, channels in MODELS.items():
        results[name] = train_and_eval(
            name, channels, train_loader, val_loader, cross_loader, train_cfg, device
        )

    # --- Summary ---
    print(f"\n\n{'='*58}")
    print("  FINAL SUMMARY")
    print("  Train: without_noise + with_noise combined")
    print("  Cross-eval: old dataset (real-world generalisation)")
    print(f"{'='*58}")
    print(f"  {'Model':<28} {'Best val':>9}  {'Own val':>8}  {'Cross acc':>10}")
    print(f"  {'-'*55}")
    for name, (best, own, cross) in results.items():
        print(f"  {name:<28} {best:>9.1%}  {own:>8.1%}  {cross:>10.1%}")
    print(f"{'='*58}\n")

    best_model = max(results, key=lambda n: results[n][2])
    print(f"  => Best generalisation: {best_model.strip()}")
    print(f"     Cross acc: {results[best_model][2]:.1%}\n")


if __name__ == "__main__":
    main()
