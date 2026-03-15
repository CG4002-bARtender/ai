import re
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

import librosa
import torchaudio

from pipeline import load_wav, wav_to_mel, TARGET_SR, TARGET_SAMPLES


class SpeechCommandsDataset(Dataset):
    """Google Speech Commands v2 — 35 keyword classes."""

    def __init__(self, root: str | Path, subset: str = "training"):
        self._ds = torchaudio.datasets.SPEECHCOMMANDS(str(root), download=True, subset=subset)
        # Build label map from the 35 class folders
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
        # waveform: [1, T] float32 in range [-1, 1], sample_rate: 16000
        y = waveform.squeeze(0).numpy()  # [T]
        # Convert to int16-range for load_wav compat, then resample + pad
        y_8k = librosa.resample(y, orig_sr=sample_rate, target_sr=TARGET_SR)
        y_padded = np.pad(y_8k, (0, max(0, TARGET_SAMPLES - len(y_8k))))[:TARGET_SAMPLES]
        # wav_to_mel expects int16 range
        pcm = (y_padded * 32767).astype(np.int16)
        mel = wav_to_mel(pcm)  # [1, 40, 123]
        return torch.from_numpy(mel), self.class_to_idx[label]


class DrinkDataset(Dataset):
    """10 drink-name classes from data/, with optional augmentation."""

    def __init__(self, files: list[Path], labels: list[int], augment: bool = False):
        self.files = files
        self.labels = labels
        self.augment = augment

    # When augment=True each file yields 5 samples: 1 clean + 4 augmented
    _COPIES = 5

    def __len__(self):
        return len(self.files) * self._COPIES if self.augment else len(self.files)

    def __getitem__(self, idx):
        if self.augment:
            file_idx, aug_idx = divmod(idx, self._COPIES)
        else:
            file_idx, aug_idx = idx, 0

        pcm = load_wav(self.files[file_idx])  # int16 [16000]
        y = pcm.astype(np.float32) / 32768.0

        if aug_idx > 0:  # aug_idx 0 = clean original
            y = _maybe_time_shift(y)
            y = _maybe_add_noise(y)
            y = _maybe_pitch_shift(y)

        pcm_out = (y * 32767).clip(-32768, 32767).astype(np.int16)
        mel = wav_to_mel(pcm_out)  # [1, 40, 123]
        mel_t = torch.from_numpy(mel)
        if aug_idx > 0:
            mel_t = _spec_augment(mel_t)
        return mel_t, self.labels[file_idx]


def make_drink_splits(data_dir: Path, classes: list[str] | None = None):
    """Return (train_dataset, val_dataset, class_names) with 80/20 stratified split."""
    all_classes: dict[str, list[Path]] = {}
    for f in sorted(data_dir.glob("*.wav")):
        cls = re.sub(r"\d+$", "", f.stem)
        all_classes.setdefault(cls, []).append(f)

    if classes is not None:
        all_classes = {c: all_classes[c] for c in classes if c in all_classes}
    classes = all_classes

    class_names = sorted(classes)
    class_to_idx = {c: i for i, c in enumerate(class_names)}

    train_files, train_labels = [], []
    val_files, val_labels = [], []

    for cls, files in classes.items():
        n_val = max(1, len(files) // 5)
        for i, f in enumerate(files):
            if i < n_val:
                val_files.append(f)
                val_labels.append(class_to_idx[cls])
            else:
                train_files.append(f)
                train_labels.append(class_to_idx[cls])

    train_ds = DrinkDataset(train_files, train_labels, augment=True)
    val_ds = DrinkDataset(val_files, val_labels, augment=False)
    return train_ds, val_ds, class_names


# ---------- augmentation helpers ----------

def _maybe_time_shift(y: np.ndarray, p: float = 0.5, max_frac: float = 0.40) -> np.ndarray:
    if np.random.rand() >= p:
        return y
    max_shift = int(max_frac * len(y))
    shift = np.random.randint(-max_shift, max_shift + 1)
    out = np.zeros_like(y)
    if shift >= 0:
        out[shift:] = y[:len(y) - shift]
    else:
        out[:len(y) + shift] = y[-shift:]
    return out


def _maybe_add_noise(y: np.ndarray, p: float = 0.5) -> np.ndarray:
    if np.random.rand() >= p:
        return y
    snr_db = np.random.uniform(20, 40)
    signal_power = np.mean(y ** 2) + 1e-10
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = np.random.randn(len(y)) * np.sqrt(noise_power)
    return (y + noise).astype(np.float32)


def _maybe_pitch_shift(y: np.ndarray, p: float = 0.5) -> np.ndarray:
    if np.random.rand() >= p:
        return y
    steps = np.random.uniform(-1.0, 1.0)
    return librosa.effects.pitch_shift(y, sr=TARGET_SR, n_steps=steps)


def _spec_augment(mel: torch.Tensor, freq_mask_f: int = 8, time_mask_t: int = 20) -> torch.Tensor:
    """Apply one frequency mask and one time mask to a [1, F, T] mel tensor."""
    mel = mel.clone()
    _, F, T = mel.shape

    # Frequency masking
    f = np.random.randint(0, freq_mask_f + 1)
    f0 = np.random.randint(0, F - f + 1)
    mel[:, f0:f0 + f, :] = 0

    # Time masking
    t = np.random.randint(0, time_mask_t + 1)
    t0 = np.random.randint(0, T - t + 1)
    mel[:, :, t0:t0 + t] = 0

    return mel
