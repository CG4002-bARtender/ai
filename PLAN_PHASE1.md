# Phase 1 — Data Preprocessing & Visualisation

## Goal

Load the 10 drink-name WAV classes from `data/`, compute mel spectrograms, save the
arrays, and produce a comparison figure of average spectrograms per class.

---

## Audio Standard (shared across all phases)

| Parameter | Value |
|-----------|-------|
| Sample rate | 8 000 Hz |
| Duration | 2 seconds |
| Total samples | 16 000 |
| n_fft | 256 |
| hop_length | 128 |
| n_mels | 40 |
| fmin | 0 Hz |
| fmax | 4 000 Hz |
| Log epsilon | 1e-6 |
| Output shape | [1, 40, 123] float32 |

---

## Repository Structure

```
data/
  negroni00.wav
  negroni01.wav
  ...                        ← flat files, class inferred from filename prefix
src/
  pipeline.py                ← shared load_wav() and wav_to_mel()
  phase1_preprocess.py       ← Phase 1 entry point
artifacts/
  spectrograms/
    negroni.npy              ← shape [N, 1, 40, 123]
    martini.npy
    ...
  avg_spectrograms.png       ← comparison figure
```

---

## Steps

### 1. Parse `data/`

Infer class name from filename prefix by stripping trailing digits:

```python
# e.g. "negroni00.wav" → "negroni"
import re
class_name = re.sub(r'\d+$', '', Path(f).stem)
```

Group all `.wav` files by class name.

### 2. `load_wav(path) → np.ndarray [16000] int16`

Defined in `src/pipeline.py` (shared):
- Load via `scipy.io.wavfile.read`
- Mono-mix if stereo (take channel 0)
- Resample to 8 000 Hz if needed via `librosa.resample`
- Zero-pad on right if shorter than 16 000 samples
- Truncate to 16 000 samples if longer
- Return as int16

### 3. `wav_to_mel(pcm) → np.ndarray [1, 40, 123] float32`

Defined in `src/pipeline.py` (shared):

```python
S = librosa.feature.melspectrogram(
    y=pcm.astype(np.float32) / 32768.0,
    sr=8000, n_fft=256, hop_length=128,
    n_mels=40, fmin=0, fmax=4000,
)
log_S = np.log(S + 1e-6)                  # [40, 123]
return log_S[np.newaxis].astype(np.float32)  # [1, 40, 123]
```

### 4. Save spectrogram arrays

For each class, stack all sample spectrograms and save:

```
artifacts/spectrograms/<class_name>.npy   shape: [N, 1, 40, 123]
```

### 5. Visualise

One subplot per class showing the class mean spectrogram `[40, 123]`.
Layout: 2 rows × 5 cols (10 classes).
Save to `artifacts/avg_spectrograms.png`.

---

## Entry Point

```
uv run python -m src.phase1_preprocess
uv run python -m src.phase1_preprocess --data data/ --artifacts artifacts/
```

---

## Outputs

| File | Description |
|------|-------------|
| `artifacts/spectrograms/<class>.npy` | Stacked spectrograms, shape `[N, 1, 40, 123]` |
| `artifacts/avg_spectrograms.png` | Mean spectrogram grid, one panel per class |

---

## Acceptance Criteria

- All 10 classes produce a `.npy` file with correct shape `[N, 1, 40, 123]`
- `avg_spectrograms.png` shows visually distinct patterns per class
- No `-inf` values in any saved array
