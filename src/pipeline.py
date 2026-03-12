"""
Bit-exact mel-spectrogram pipeline.

Stages 1–7 mirror the FPGA/HLS hardware exactly.
DO NOT change constants, formulas, or ordering without updating PLAN.md
and alerting the hardware team.
"""

import json
import os

import librosa
import numpy as np
import scipy.io.wavfile
import scipy.signal

# ---------------------------------------------------------------------------
# Locked parameters — must match PLAN.md and pipeline_params.json exactly
# ---------------------------------------------------------------------------
SAMPLE_RATE = 8000
NUM_SAMPLES = 16000
FFT_SIZE = 256
HOP_SIZE = 128
NUM_FRAMES = 123       # floor((16000 - 256) / 128) + 1
NUM_MELS = 40
FMIN = 0
FMAX = 4000
LOG_EPSILON = 1e-6


# ---------------------------------------------------------------------------
# Stage 1 — Frame Extraction
# ---------------------------------------------------------------------------
def extract_frames(pcm: np.ndarray) -> np.ndarray:
    """
    Input:  [16000] int16
    Output: [123, 256] int16
    """
    assert pcm.dtype == np.int16, "pcm must be int16"
    assert pcm.shape == (NUM_SAMPLES,), f"pcm must be shape ({NUM_SAMPLES},)"
    return np.stack([pcm[i * HOP_SIZE: i * HOP_SIZE + FFT_SIZE] for i in range(NUM_FRAMES)])


# ---------------------------------------------------------------------------
# Stage 2 — Hann Window
# ---------------------------------------------------------------------------
def apply_hann(frames: np.ndarray) -> np.ndarray:
    """
    Input:  [123, 256] int16
    Output: [123, 256] float64
    """
    window = np.hanning(FFT_SIZE)  # float64, shape [256]
    return frames.astype(np.float64) * window


# ---------------------------------------------------------------------------
# Stage 3 — Real FFT
# ---------------------------------------------------------------------------
def compute_rfft(windowed: np.ndarray) -> np.ndarray:
    """
    Input:  [123, 256] float64
    Output: [123, 129] complex128
    """
    return np.fft.rfft(windowed, n=FFT_SIZE, axis=1)


# ---------------------------------------------------------------------------
# Stage 4 — Power Spectrum
# ---------------------------------------------------------------------------
def compute_power(ffts: np.ndarray) -> np.ndarray:
    """
    Input:  [123, 129] complex128
    Output: [123, 129] float64  —  P[k] = re[k]^2 + im[k]^2
    """
    return ffts.real ** 2 + ffts.imag ** 2


# ---------------------------------------------------------------------------
# Stage 5 — Mel Filterbank
# ---------------------------------------------------------------------------
def apply_mel_filterbank(power: np.ndarray) -> np.ndarray:
    """
    Input:  [123, 129] float64
    Output: [123, 40] float64
    """
    mel_basis = _get_mel_basis()
    return power @ mel_basis.T


def _get_mel_basis() -> np.ndarray:
    """Returns float64 mel filterbank [40, 129]. norm=None is mandatory."""
    return librosa.filters.mel(
        sr=SAMPLE_RATE,
        n_fft=FFT_SIZE,
        n_mels=NUM_MELS,
        fmin=FMIN,
        fmax=FMAX,
        norm=None,
        dtype=np.float64,
    )


# ---------------------------------------------------------------------------
# Stage 6 — Log Compression
# ---------------------------------------------------------------------------
def apply_log(mel: np.ndarray) -> np.ndarray:
    """
    Input:  [123, 40] float64
    Output: [123, 40] float64  —  natural log, epsilon=1e-6
    """
    return np.log(mel + LOG_EPSILON)


# ---------------------------------------------------------------------------
# Stage 7 — Stack Frames
# ---------------------------------------------------------------------------
def stack_frames(log_mel: np.ndarray) -> np.ndarray:
    """
    Input:  [123, 40] float64
    Output: [1, 40, 123] float32
    """
    spectrogram = log_mel.T              # [40, 123]
    spectrogram = spectrogram[np.newaxis]  # [1, 40, 123]
    return spectrogram.astype(np.float32)


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------
def run_pipeline(pcm: np.ndarray) -> np.ndarray:
    """
    Input:  [16000] int16
    Output: [1, 40, 123] float32
    """
    frames = extract_frames(pcm)
    windowed = apply_hann(frames)
    ffts = compute_rfft(windowed)
    power = compute_power(ffts)
    mel = apply_mel_filterbank(power)
    log_mel = apply_log(mel)
    return stack_frames(log_mel)


# ---------------------------------------------------------------------------
# Audio loading
# ---------------------------------------------------------------------------
def load_wav(path: str) -> np.ndarray:
    """
    Load a WAV file and return a [16000] int16 array at SAMPLE_RATE.
    Resamples if needed, zero-pads if short, truncates if long.
    """
    sr, data = scipy.io.wavfile.read(path)

    # Convert stereo to mono
    if data.ndim > 1:
        data = data[:, 0]

    # Resample to SAMPLE_RATE if necessary
    if sr != SAMPLE_RATE:
        # resample_poly requires float; we convert, resample, convert back
        data_f = data.astype(np.float64)
        gcd = _gcd(SAMPLE_RATE, sr)
        data_f = scipy.signal.resample_poly(data_f, SAMPLE_RATE // gcd, sr // gcd)
        data = np.clip(data_f, -32768, 32767).astype(np.int16)

    # Ensure int16
    if data.dtype != np.int16:
        data = data.astype(np.int16)

    # Pad or truncate to NUM_SAMPLES
    if len(data) < NUM_SAMPLES:
        data = np.pad(data, (0, NUM_SAMPLES - len(data)))
    else:
        data = data[:NUM_SAMPLES]

    return data


def _gcd(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return a


# ---------------------------------------------------------------------------
# Artifact generation
# ---------------------------------------------------------------------------
def save_artifacts(artifacts_dir: str = "artifacts") -> None:
    """
    Write hann_window.npy, mel_weights.npy, and pipeline_params.json
    to artifacts_dir.
    """
    os.makedirs(artifacts_dir, exist_ok=True)

    hann = np.hanning(FFT_SIZE)
    np.save(os.path.join(artifacts_dir, "hann_window.npy"), hann)

    mel_basis = _get_mel_basis()
    np.save(os.path.join(artifacts_dir, "mel_weights.npy"), mel_basis)

    params = {
        "sample_rate": SAMPLE_RATE,
        "num_samples": NUM_SAMPLES,
        "fragment_size": HOP_SIZE,
        "fft_size": FFT_SIZE,
        "hop_size": HOP_SIZE,
        "num_frames": NUM_FRAMES,
        "num_mels": NUM_MELS,
        "fmin": FMIN,
        "fmax": FMAX,
        "log_epsilon": LOG_EPSILON,
        "cnn_input_shape": [1, NUM_MELS, NUM_FRAMES],
    }
    with open(os.path.join(artifacts_dir, "pipeline_params.json"), "w") as f:
        json.dump(params, f, indent=2)

    print(f"Artifacts written to {artifacts_dir}/")
    print(f"  hann_window.npy  shape={hann.shape}  dtype={hann.dtype}")
    print(f"  mel_weights.npy  shape={mel_basis.shape}  dtype={mel_basis.dtype}")
    print(f"  pipeline_params.json")
