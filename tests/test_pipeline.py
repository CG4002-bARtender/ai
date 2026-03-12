"""
Unit tests for the bit-exact preprocessing pipeline.

Each test targets one stage and verifies shape, dtype, and numerical
correctness. These tests are the primary guard against accidental
deviations from the FPGA/HLS spec.
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.pipeline import (
    FFT_SIZE,
    FMAX,
    HOP_SIZE,
    LOG_EPSILON,
    NUM_FRAMES,
    NUM_MELS,
    NUM_SAMPLES,
    SAMPLE_RATE,
    apply_hann,
    apply_log,
    apply_mel_filterbank,
    compute_power,
    compute_rfft,
    extract_frames,
    run_pipeline,
    stack_frames,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pcm(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(-32768, 32767, size=NUM_SAMPLES, dtype=np.int16)


# ---------------------------------------------------------------------------
# Stage 1 — Frame Extraction
# ---------------------------------------------------------------------------

def test_frame_extraction_shape():
    pcm = _make_pcm()
    frames = extract_frames(pcm)
    assert frames.shape == (NUM_FRAMES, FFT_SIZE), f"Expected ({NUM_FRAMES}, {FFT_SIZE}), got {frames.shape}"


def test_frame_extraction_dtype():
    pcm = _make_pcm()
    frames = extract_frames(pcm)
    assert frames.dtype == np.int16, f"Expected int16, got {frames.dtype}"


def test_frame_extraction_boundaries():
    pcm = np.arange(NUM_SAMPLES, dtype=np.int16)
    frames = extract_frames(pcm)
    # Frame 0: samples 0–255
    np.testing.assert_array_equal(frames[0], pcm[0:FFT_SIZE])
    # Frame 1: samples 128–383
    np.testing.assert_array_equal(frames[1], pcm[HOP_SIZE: HOP_SIZE + FFT_SIZE])
    # NUM_FRAMES=123 is locked; last frame starts at 122*128=15616, ends at 15871.
    # The tail 128 samples are intentionally unused to keep the frame count exact.
    last_start = (NUM_FRAMES - 1) * HOP_SIZE
    assert last_start + FFT_SIZE <= NUM_SAMPLES, "Last frame must not exceed pcm length"
    np.testing.assert_array_equal(frames[-1], pcm[last_start: last_start + FFT_SIZE])


# ---------------------------------------------------------------------------
# Stage 2 — Hann Window
# ---------------------------------------------------------------------------

def test_hann_window_shape():
    frames = np.zeros((NUM_FRAMES, FFT_SIZE), dtype=np.int16)
    windowed = apply_hann(frames)
    assert windowed.shape == (NUM_FRAMES, FFT_SIZE)


def test_hann_window_dtype():
    frames = np.zeros((NUM_FRAMES, FFT_SIZE), dtype=np.int16)
    windowed = apply_hann(frames)
    assert windowed.dtype == np.float64, f"Expected float64, got {windowed.dtype}"


def test_hann_window_endpoints_near_zero():
    frames = np.ones((NUM_FRAMES, FFT_SIZE), dtype=np.int16)
    windowed = apply_hann(frames)
    # w[0] and w[N-1] must be 0 (or very close — np.hanning uses cosine formula)
    assert abs(windowed[0, 0]) < 1e-10, f"w[0] should be ~0, got {windowed[0, 0]}"
    assert abs(windowed[0, FFT_SIZE - 1]) < 1e-10, f"w[N-1] should be ~0, got {windowed[0, FFT_SIZE - 1]}"


def test_hann_window_midpoints_near_one():
    frames = np.ones((NUM_FRAMES, FFT_SIZE), dtype=np.int16)
    windowed = apply_hann(frames)
    # w[127] and w[128] should be close to 1 for N=256
    # np.hanning(256) peaks between indices 127 and 128 (~0.99996), not exactly 1.
    assert abs(windowed[0, 127] - 1.0) < 1e-3, f"w[127] should be ~1, got {windowed[0, 127]}"
    assert abs(windowed[0, 128] - 1.0) < 1e-3, f"w[128] should be ~1, got {windowed[0, 128]}"


# ---------------------------------------------------------------------------
# Stage 3 — Real FFT
# ---------------------------------------------------------------------------

def test_rfft_shape():
    windowed = np.zeros((NUM_FRAMES, FFT_SIZE))
    ffts = compute_rfft(windowed)
    assert ffts.shape == (NUM_FRAMES, FFT_SIZE // 2 + 1), f"Expected ({NUM_FRAMES}, 129), got {ffts.shape}"


def test_rfft_dtype():
    windowed = np.zeros((NUM_FRAMES, FFT_SIZE))
    ffts = compute_rfft(windowed)
    assert ffts.dtype == np.complex128, f"Expected complex128, got {ffts.dtype}"


def test_rfft_dc_bin_is_real():
    rng = np.random.default_rng(42)
    windowed = rng.standard_normal((NUM_FRAMES, FFT_SIZE))
    ffts = compute_rfft(windowed)
    # DC bin of a real signal must be purely real
    assert np.allclose(ffts[:, 0].imag, 0, atol=1e-9), "DC bin should be real-valued"


def test_rfft_known_sine():
    # Pure sine at 1 kHz, 8 kHz sample rate → bin = round(1000 / (8000/256)) = 32
    t = np.arange(FFT_SIZE) / SAMPLE_RATE
    freq = 1000.0
    sine = np.sin(2 * np.pi * freq * t)
    windowed = np.tile(sine, (NUM_FRAMES, 1))
    ffts = compute_rfft(windowed)
    expected_bin = round(freq / (SAMPLE_RATE / FFT_SIZE))  # 32
    magnitudes = np.abs(ffts[0])
    assert magnitudes[expected_bin] == magnitudes.max(), (
        f"Peak should be at bin {expected_bin}, got bin {magnitudes.argmax()}"
    )


# ---------------------------------------------------------------------------
# Stage 4 — Power Spectrum
# ---------------------------------------------------------------------------

def test_power_spectrum_shape():
    ffts = np.zeros((NUM_FRAMES, FFT_SIZE // 2 + 1), dtype=np.complex128)
    power = compute_power(ffts)
    assert power.shape == (NUM_FRAMES, FFT_SIZE // 2 + 1)


def test_power_spectrum_non_negative():
    rng = np.random.default_rng(0)
    ffts = rng.standard_normal((NUM_FRAMES, FFT_SIZE // 2 + 1)) + 1j * rng.standard_normal((NUM_FRAMES, FFT_SIZE // 2 + 1))
    power = compute_power(ffts)
    assert np.all(power >= 0), "Power spectrum must be non-negative"


def test_power_spectrum_formula():
    # Spot-check: P = re^2 + im^2
    z = np.array([[3.0 + 4.0j]])
    power = compute_power(z)
    assert abs(power[0, 0] - 25.0) < 1e-12, f"Expected 25.0, got {power[0, 0]}"


def test_power_spectrum_dtype():
    ffts = np.zeros((NUM_FRAMES, FFT_SIZE // 2 + 1), dtype=np.complex128)
    power = compute_power(ffts)
    assert power.dtype == np.float64


# ---------------------------------------------------------------------------
# Stage 5 — Mel Filterbank
# ---------------------------------------------------------------------------

def test_mel_filterbank_shape():
    power = np.ones((NUM_FRAMES, FFT_SIZE // 2 + 1))
    mel = apply_mel_filterbank(power)
    assert mel.shape == (NUM_FRAMES, NUM_MELS), f"Expected ({NUM_FRAMES}, {NUM_MELS}), got {mel.shape}"


def test_mel_filterbank_non_negative():
    power = np.abs(np.random.randn(NUM_FRAMES, FFT_SIZE // 2 + 1))
    mel = apply_mel_filterbank(power)
    assert np.all(mel >= 0), "Mel filterbank output must be non-negative"


def test_mel_basis_shape():
    import librosa
    basis = librosa.filters.mel(
        sr=SAMPLE_RATE, n_fft=FFT_SIZE, n_mels=NUM_MELS,
        fmin=0, fmax=FMAX, norm=None, dtype=np.float64,
    )
    assert basis.shape == (NUM_MELS, FFT_SIZE // 2 + 1), (
        f"Mel basis should be ({NUM_MELS}, {FFT_SIZE // 2 + 1}), got {basis.shape}"
    )


# ---------------------------------------------------------------------------
# Stage 6 — Log Compression
# ---------------------------------------------------------------------------

def test_log_compression_shape():
    mel = np.ones((NUM_FRAMES, NUM_MELS))
    log_mel = apply_log(mel)
    assert log_mel.shape == (NUM_FRAMES, NUM_MELS)


def test_log_compression_no_inf():
    # All-zero input is the worst case; epsilon must prevent -inf
    mel = np.zeros((NUM_FRAMES, NUM_MELS))
    log_mel = apply_log(mel)
    assert not np.any(np.isinf(log_mel)), "Log compression must not produce -inf (check epsilon)"
    assert not np.any(np.isnan(log_mel)), "Log compression must not produce nan"


def test_log_compression_epsilon_spot_check():
    mel = np.array([[0.0]])
    log_mel = apply_log(mel)
    expected = np.log(LOG_EPSILON)
    assert abs(log_mel[0, 0] - expected) < 1e-12, (
        f"log(0 + 1e-6) should be {expected:.6f}, got {log_mel[0, 0]:.6f}"
    )


def test_log_compression_uses_natural_log():
    mel = np.array([[1.0]])
    log_mel = apply_log(mel)
    # log(1 + 1e-6) ≈ 1e-6 (natural log)
    expected = np.log(1.0 + LOG_EPSILON)
    assert abs(log_mel[0, 0] - expected) < 1e-12


# ---------------------------------------------------------------------------
# Stage 7 — Stack Frames
# ---------------------------------------------------------------------------

def test_stack_frames_shape():
    log_mel = np.zeros((NUM_FRAMES, NUM_MELS))
    spec = stack_frames(log_mel)
    assert spec.shape == (1, NUM_MELS, NUM_FRAMES), f"Expected (1, {NUM_MELS}, {NUM_FRAMES}), got {spec.shape}"


def test_stack_frames_dtype():
    log_mel = np.zeros((NUM_FRAMES, NUM_MELS))
    spec = stack_frames(log_mel)
    assert spec.dtype == np.float32, f"Expected float32, got {spec.dtype}"


def test_stack_frames_transpose():
    # Verify the transpose: log_mel[t, m] == spec[0, m, t]
    log_mel = np.arange(NUM_FRAMES * NUM_MELS, dtype=np.float64).reshape(NUM_FRAMES, NUM_MELS)
    spec = stack_frames(log_mel)
    for t in range(min(5, NUM_FRAMES)):
        for m in range(min(5, NUM_MELS)):
            assert abs(spec[0, m, t] - log_mel[t, m]) < 1e-6, (
                f"Transpose mismatch at t={t}, m={m}"
            )


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def test_full_pipeline_shape():
    pcm = _make_pcm()
    spec = run_pipeline(pcm)
    assert spec.shape == (1, NUM_MELS, NUM_FRAMES), (
        f"Expected (1, {NUM_MELS}, {NUM_FRAMES}), got {spec.shape}"
    )


def test_full_pipeline_dtype():
    pcm = _make_pcm()
    spec = run_pipeline(pcm)
    assert spec.dtype == np.float32


def test_full_pipeline_deterministic():
    pcm = _make_pcm(seed=7)
    spec1 = run_pipeline(pcm)
    spec2 = run_pipeline(pcm)
    np.testing.assert_array_equal(spec1, spec2)
