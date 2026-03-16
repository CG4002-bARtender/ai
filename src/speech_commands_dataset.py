from __future__ import annotations
from pathlib import Path

import numpy as np
import torch
import librosa
import torchaudio
from torch.utils.data import Dataset

from config import AudioConfig
from spectrogram import SpectrogramConverter


class SpeechCommandsDataset(Dataset):
    """Google Speech Commands v2 — 35 keyword classes."""

    def __init__(
        self,
        root: str | Path,
        subset: str,
        audio_cfg: AudioConfig,
        spectrogram_converter: SpectrogramConverter,
    ):
        sc_path = Path(root) / "SpeechCommands" / "speech_commands_v0.02"
        sentinel_files = ["validation_list.txt", "testing_list.txt"]
        if sc_path.exists() and not all((sc_path / f).exists() for f in sentinel_files):
            import shutil
            shutil.rmtree(sc_path.parent)
        self._ds = torchaudio.datasets.SPEECHCOMMANDS(str(root), download=True, subset=subset)
        self._cfg = audio_cfg
        self._converter = spectrogram_converter

        root = Path(root) / "SpeechCommands" / "speech_commands_v0.02"
        self.classes = sorted(
            d.name for d in root.iterdir()
            if d.is_dir() and not d.name.startswith("_")
        )
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

    def __len__(self):
        return len(self._ds)

    def __getitem__(self, idx):
        waveform, sample_rate, label, *_ = self._ds[idx]
        y = waveform.squeeze(0).numpy()
        y_8k = librosa.resample(y, orig_sr=sample_rate, target_sr=self._cfg.target_sr)
        y_padded = np.pad(y_8k, (0, max(0, self._cfg.target_samples - len(y_8k))))[:self._cfg.target_samples]
        pcm = (y_padded * 32767).astype(np.int16)
        mel = self._converter.convert(pcm)
        return torch.from_numpy(mel), self.class_to_idx[label]
