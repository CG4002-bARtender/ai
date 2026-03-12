# Speech Classifier Pipeline — Implementation Plan

## Context

Build a Python reference implementation of a 10-class keyword spotting pipeline targeting the AMD Xilinx Ultra96-V2 (ZU3EG). The pipeline must be bit-exact. Every constant, formula, and ordering will be mirrored in downstream FPGA/HLS hardware. This design doc is the authoritative spec; no deviations are permitted without alerting the user and updating the design document as appropriate.

**Current phase:** Spectrogram generation + class selection. The 10 drink-name classes have not been finalised. This phase generates spectrograms from candidate audio and selects the 10 most acoustically separable classes. The full training pipeline will be planned after class selection.

---

## Repository Structure (Phase 1)

```
data/
  candidates/                 ← one sub-folder per candidate drink name
    <drink_name>/             ← .wav files for that class (at least 10 per class)
artifacts/
  hann_window.npy             ← float64 [256]
  mel_weights.npy             ← float64 [40, 129]
  pipeline_params.json        ← all locked parameters
src/
  pipeline.py                 ← Stages 1–7: WAV → spectrogram [1, 40, 123]
  select_classes.py           ← load candidates, score separability, pick best 10
tests/
  test_pipeline.py            ← numerical unit tests for each pipeline stage
```

---

## Locked Pipeline Parameters (DO NOT CHANGE)

These values are frozen and will be mirrored exactly in FPGA/HLS hardware.

| Parameter | Value | Notes |
|-----------|-------|-------|
| Sample rate | 8 000 Hz | Matches ESP32 I2S config |
| Total samples | 16 000 | One full inference window (2 seconds) |
| Fragment size | 128 samples | One data packet from ESP32 |
| FFT size (N) | 256 | Power of 2 |
| Hop size | 128 | 50% overlap |
| Total frames | 123 | `floor((16000 - 256) / 128) + 1` |
| Mel bins | 40 | |
| fmin | 0 Hz | |
| fmax | 4 000 Hz | Nyquist at 8 kHz |
| Log floor epsilon | 1e-6 | Prevents log(0) |
| CNN input shape | [1, 40, 123] | [channels, mel_bins, frames] |
| Output classes | 10 | TBD — chosen by class selection step |
| Sample dtype | int16 | Two's complement, −32768 to 32767 |

> **Note on mel filterbank sr:** librosa must be called with `sr=8000, fmax=4000` to match the locked 8 kHz sample rate. Using `sr=16000, fmax=8000` would produce different bin boundaries and break hardware alignment.

---

## Stage-by-Stage Preprocessing Specification

### Stage 1 — Frame Extraction

- **Input:** `[16000]` int16
- **Output:** `[123, 256]` int16
- **Method:** `np.stack([pcm[i*128 : i*128+256] for i in range(123)])`
- **Constraints:** No float conversion. No librosa/torchaudio. Pure numpy slicing.

### Stage 2 — Hann Window

- **Input:** `[123, 256]` int16
- **Output:** `[123, 256]` float64
- **Method:** `window = np.hanning(256)` then `frames.astype(np.float64) * window`
- **Formula:** `w[n] = 0.5 * (1 - cos(2π·n / (N−1)))`, N=256
- **Artifact:** Export as `artifacts/hann_window.npy` (float64)

### Stage 3 — Real FFT

- **Input:** `[123, 256]` float64
- **Output:** `[123, 129]` complex128
- **Method:** `np.fft.rfft(windowed, n=256, axis=1)`
- **Constraint:** Use `rfft` not `fft`. Keep all 129 bins.

### Stage 4 — Power Spectrum

- **Input:** `[123, 129]` complex128
- **Output:** `[123, 129]` float64
- **Formula:** `P[k] = re[k]² + im[k]²` (equivalently `np.abs(ffts)**2`)
- **Constraints:** No square root. No scaling.

### Stage 5 — Mel Filterbank

- **Input:** `[123, 129]` float64
- **Output:** `[123, 40]` float64
- **Method:**
  ```python
  mel_basis = librosa.filters.mel(
      sr=8000, n_fft=256, n_mels=40,
      fmin=0, fmax=4000, norm=None, dtype=np.float64
  )  # shape [40, 129]
  mel = power @ mel_basis.T  # shape [123, 40]
  ```
- **Constraint:** `norm=None` is mandatory. Any normalization breaks hardware filterbank.
- **Artifact:** Export `mel_basis` as `artifacts/mel_weights.npy` (float64)

### Stage 6 — Log Compression

- **Input:** `[123, 40]` float64
- **Output:** `[123, 40]` float64
- **Formula:** `np.log(mel + 1e-6)`
- **Constraints:** Natural log (`np.log`), not log10 or log2. Epsilon = 1e-6 is mandatory.

### Stage 7 — Stack Frames

- **Input:** `[123, 40]` float64
- **Output:** `[1, 40, 123]` float32
- **Method:**
  ```python
  spectrogram = log_mel.T               # [40, 123]
  spectrogram = spectrogram[np.newaxis] # [1, 40, 123]
  spectrogram = spectrogram.astype(np.float32)
  ```

---

## Audio Loading

- Load as int16 via `scipy.io.wavfile`
- Resample to 8 000 Hz if source differs (use `scipy.signal.resample_poly`)
- Shorter than 16 000 samples → zero-pad on the right
- Longer than 16 000 samples → truncate to 16 000
- Do NOT normalize or scale int16 values before the pipeline

---

## Class Selection (`src/select_classes.py`)

### Goal

Given N candidate drink-name classes (each with ≥10 WAV recordings), select the 10 classes whose spectrograms are most separable in mel-spectrogram space.

### Method

**Step 1 — Compute class mean spectrogram**

For each candidate class:
1. Load all WAV files, run each through the full pipeline → `[1, 40, 123]` float32
2. Flatten each spectrogram to a 1-D vector of length 4920 (= 40 × 123)
3. Average across all samples → one mean vector per class

**Step 2 — Pairwise cosine distance matrix**

```python
from sklearn.metrics.pairwise import cosine_distances
# class_means: array of shape [N_candidates, 4920]
dist_matrix = cosine_distances(class_means)  # [N_candidates, N_candidates]
```

Cosine distance is preferred over L2 here because it is invariant to overall loudness differences between recordings.

**Step 3 — Greedy max-min selection**

Select 10 classes that maximise the minimum pairwise distance (max-min diversity):

```python
def greedy_max_min_select(dist_matrix, k=10):
    n = len(dist_matrix)
    # seed: pair with largest distance
    i, j = np.unravel_index(np.argmax(dist_matrix), (n, n))
    selected = [i, j]
    while len(selected) < k:
        # for each candidate, find its min dist to already-selected set
        min_dists = np.min(dist_matrix[np.ix_(list(range(n)), selected)], axis=1)
        # mask already selected
        min_dists[selected] = -1
        selected.append(int(np.argmax(min_dists)))
    return selected
```

**Step 4 — Output**

- Print ranked selected classes and the minimum pairwise distance (the "separation score")
- Save `artifacts/class_distance_matrix.png` — heatmap of full N×N distance matrix with selected classes highlighted
- Save `artifacts/selected_spectrograms.png` — grid of mean spectrograms for the 10 selected classes (one subplot per class)
- Print the final 10 class names to stdout (copy-paste ready for the next phase)

### Acceptance Criterion

The minimum pairwise cosine distance among the 10 selected classes should be ≥ 0.10. If it is lower, the candidate pool may be too phonetically similar and more candidate recordings should be collected.

---

## Unit Tests (`tests/test_pipeline.py`)

| Test | What to Verify |
|------|---------------|
| `test_frame_extraction` | Shape [123,256]. Frame 0 = samples 0–255. Frame 1 = 128–383. Last frame ends at sample 15999. |
| `test_hann_window` | Shape [123,256]. `w[0]` and `w[255]` ≈ 0. `w[127]` and `w[128]` ≈ 1. dtype float64. |
| `test_rfft` | Shape [123,129]. dtype complex128. DC bin is real-valued. Verified against known sine wave. |
| `test_power_spectrum` | Shape [123,129]. All values ≥ 0. `re²+im²` formula explicitly checked. |
| `test_mel_filterbank` | Shape [123,40]. All values ≥ 0. Mel basis shape [40,129]. |
| `test_log_compression` | No `-inf` values (epsilon working). Shape [123,40]. `log(1+eps)` spot check. |
| `test_stack_frames` | Shape [1,40,123]. dtype float32. |

---

## Critical Constraints (Non-Negotiable)

1. **No `torchaudio.transforms.MelSpectrogram`** — implement stage-by-stage as specified
2. **`norm=None`** in librosa mel filterbank — any normalization changes weights and breaks hardware
3. **Natural log** (`np.log`) only — hardware CORDIC/LUT implements natural log
4. **int16 through Stage 1** — do not cast to float prematurely
5. **float64** for `hann_window.npy` and `mel_weights.npy` — hardware team quantizes from full precision
6. **`sr=8000, fmax=4000`** in the librosa mel call — must match the locked 8 kHz sample rate

---

## Verification Checklist (Phase 1)

- [ ] All 7 pipeline unit tests pass
- [ ] `artifacts/hann_window.npy` — shape [256], float64
- [ ] `artifacts/mel_weights.npy` — shape [40, 129], float64
- [ ] `artifacts/pipeline_params.json` — all locked parameters present
- [ ] `artifacts/class_distance_matrix.png` — heatmap generated
- [ ] `artifacts/selected_spectrograms.png` — 10 mean spectrograms plotted
- [ ] 10 class names printed to stdout with separation score ≥ 0.10

---

## Dependencies

Use uv instead of pip to install and manage dependencies:

| Package | Version | Purpose |
|---------|---------|---------|
| numpy | ≥1.24.0 | All numerical pipeline stages |
| librosa | ≥0.10.0 | Mel filterbank generation only |
| scipy | ≥1.10.0 | WAV file loading, resampling |
| scikit-learn | ≥1.3.0 | Cosine distance matrix |
| matplotlib | ≥3.7.0 | Spectrogram and heatmap visualisation |
