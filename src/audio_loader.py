import numpy as np
import scipy.io.wavfile as wav
import librosa
from pathlib import Path

from config import AudioConfig


class AudioLoader:
    def __init__(self, cfg: AudioConfig):
        self._cfg = cfg

    def load(self, path: Path) -> np.ndarray:
        sr, data = wav.read(path)
        if data.ndim > 1:
            data = data[:, 0]
        data = data.astype(np.float32)
        if sr != self._cfg.target_sr:
            data = librosa.resample(data, orig_sr=sr, target_sr=self._cfg.target_sr)
        n = self._cfg.target_samples
        if len(data) < n:
            data = np.pad(data, (0, n - len(data)))
        else:
            data = data[:n]
        return data.astype(np.int16)
