from __future__ import annotations
import re
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from audio_loader import AudioLoader
from spectrogram import SpectrogramConverter
from augmentation import Augmenter


class DrinkDataset(Dataset):
    """10 drink-name classes from data/audio/, with optional augmentation."""

    _COPIES = 5

    def __init__(
        self,
        files: list[Path],
        labels: list[int],
        audio_loader: AudioLoader,
        spectrogram_converter: SpectrogramConverter,
        augmenter: Augmenter,
        augment: bool = False,
    ):
        self._files = files
        self._labels = labels
        self._loader = audio_loader
        self._converter = spectrogram_converter
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
    def make_splits(
        data_dir: Path,
        audio_loader: AudioLoader,
        spectrogram_converter: SpectrogramConverter,
        augmenter: Augmenter,
        classes: list[str] | None = None,
    ) -> tuple["DrinkDataset", "DrinkDataset", list[str]]:
        all_classes: dict[str, list[Path]] = {}
        for f in sorted(data_dir.glob("*.wav")):
            cls = re.sub(r"\d+$", "", f.stem)
            all_classes.setdefault(cls, []).append(f)

        if classes is not None:
            all_classes = {c: all_classes[c] for c in classes if c in all_classes}

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

        train_ds = DrinkDataset(
            train_files, train_labels, audio_loader, spectrogram_converter, augmenter, augment=True
        )
        val_ds = DrinkDataset(
            val_files, val_labels, audio_loader, spectrogram_converter, augmenter, augment=False
        )
        return train_ds, val_ds, class_names
