import numpy as np
import librosa

from config import AudioConfig


class SpectrogramConverter:
    def __init__(self, cfg: AudioConfig):
        self._cfg = cfg

    def convert(self, pcm: np.ndarray) -> np.ndarray:
        y = pcm.astype(np.float32) / 32768.0
        S = librosa.feature.melspectrogram(
            y=y,
            sr=self._cfg.target_sr,
            n_fft=self._cfg.n_fft,
            hop_length=self._cfg.hop_length,
            n_mels=self._cfg.n_mels,
            fmin=self._cfg.fmin,
            fmax=self._cfg.fmax,
        )
        log_S = np.log(S + 1e-6)
        return log_S[np.newaxis].astype(np.float32)
