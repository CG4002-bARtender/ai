"""
Loads drink audio from the new subdirectory datasets (without_noise + with_noise)
combined into a single train/val split.

New dataset structure: <root>/<class>/<variant>/<file>.wav
Class names use underscores (irish_coffee) — normalised to canonical names.
"""
from __future__ import annotations
import re
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, ConcatDataset

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


class SubdirDrinkDataset(Dataset):
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

    @staticmethod
    def from_root(root: Path, audio_loader, spec_converter, augmenter, augment=False):
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
        return SubdirDrinkDataset(files, labels, audio_loader, spec_converter, augmenter, augment)


def make_combined_splits(roots: list[Path], audio_loader, spec_converter, augmenter):
    """
    Given a list of dataset roots, combine all files and return
    (train_dataset, val_dataset, class_names).

    Uses 80/20 split per class per root.
    """
    train_files, train_labels = [], []
    val_files,   val_labels   = [], []

    for root in roots:
        if not root.exists():
            print(f"  WARNING: dataset path not found, skipping: {root}")
            continue
        all_classes: dict[str, list[Path]] = {}
        for class_dir in sorted(root.iterdir()):
            if not class_dir.is_dir():
                continue
            canon = CANONICAL.get(class_dir.name)
            if canon is None:
                continue
            files = sorted(class_dir.rglob("*.wav"))
            all_classes.setdefault(canon, []).extend(files)

        for cls, files in all_classes.items():
            n_val = max(1, len(files) // 5)
            for i, f in enumerate(files):
                if i < n_val:
                    val_files.append(f);  val_labels.append(CLASS_TO_IDX[cls])
                else:
                    train_files.append(f); train_labels.append(CLASS_TO_IDX[cls])

    train_ds = SubdirDrinkDataset(train_files, train_labels,
                                   audio_loader, spec_converter, augmenter, augment=True)
    val_ds   = SubdirDrinkDataset(val_files,   val_labels,
                                   audio_loader, spec_converter, augmenter, augment=False)
    return train_ds, val_ds, CANONICAL_CLASSES
