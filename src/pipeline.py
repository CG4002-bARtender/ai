import numpy as np
import librosa
import scipy.io.wavfile as wav

TARGET_SR = 8000
TARGET_SAMPLES = 16000


def load_wav(path) -> np.ndarray:
    sr, data = wav.read(path)
    if data.ndim > 1:
        data = data[:, 0]
    data = data.astype(np.float32)
    if sr != TARGET_SR:
        data = librosa.resample(data, orig_sr=sr, target_sr=TARGET_SR)
    if len(data) < TARGET_SAMPLES:
        data = np.pad(data, (0, TARGET_SAMPLES - len(data)))
    else:
        data = data[:TARGET_SAMPLES]
    return data.astype(np.int16)


def wav_to_mel(pcm: np.ndarray) -> np.ndarray:
    y = pcm.astype(np.float32) / 32768.0
    S = librosa.feature.melspectrogram(
        y=y, sr=TARGET_SR, n_fft=256, hop_length=128,
        n_mels=40, fmin=0, fmax=4000,
    )
    log_S = np.log(S + 1e-6)
    return log_S[np.newaxis].astype(np.float32)
