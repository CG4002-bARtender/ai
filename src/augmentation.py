import numpy as np
import torch
import librosa

from config import AudioConfig


class Augmenter:
    def __init__(self, cfg: AudioConfig, freq_mask_f: int = 8, time_mask_t: int = 20):
        self._cfg = cfg
        self._freq_mask_f = freq_mask_f
        self._time_mask_t = time_mask_t

    def augment_waveform(self, y: np.ndarray) -> np.ndarray:
        y = self._maybe_time_shift(y)
        y = self._maybe_add_noise(y)
        y = self._maybe_pitch_shift(y)
        return y

    def augment_spectrogram(self, mel: torch.Tensor) -> torch.Tensor:
        mel = mel.clone()
        _, F, T = mel.shape

        f = np.random.randint(0, self._freq_mask_f + 1)
        f0 = np.random.randint(0, F - f + 1)
        mel[:, f0:f0 + f, :] = 0

        t = np.random.randint(0, self._time_mask_t + 1)
        t0 = np.random.randint(0, T - t + 1)
        mel[:, :, t0:t0 + t] = 0

        return mel

    def _maybe_time_shift(self, y: np.ndarray, p: float = 0.5, max_frac: float = 0.40) -> np.ndarray:
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

    def _maybe_add_noise(self, y: np.ndarray, p: float = 0.5) -> np.ndarray:
        if np.random.rand() >= p:
            return y
        snr_db = np.random.uniform(20, 40)
        signal_power = np.mean(y ** 2) + 1e-10
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = np.random.randn(len(y)) * np.sqrt(noise_power)
        return (y + noise).astype(np.float32)

    def _maybe_pitch_shift(self, y: np.ndarray, p: float = 0.5) -> np.ndarray:
        if np.random.rand() >= p:
            return y
        steps = np.random.uniform(-1.0, 1.0)
        return librosa.effects.pitch_shift(y, sr=self._cfg.target_sr, n_steps=steps)
