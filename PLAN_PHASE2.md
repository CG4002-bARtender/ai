# Phase 2 — Foundation CNN Pretraining (Google Speech Commands)

## Goal

Train a small DPU-compatible CNN on Google Speech Commands v2 (35-class keyword recognition)
using the same audio spec as Phase 1. Save the trained weights for Phase 3 transfer learning.

---

## Audio Standard

Same as Phase 1 — 8 kHz, 2 seconds, output shape `[1, 40, 123] float32`.

Speech Commands clips are originally 16 kHz / 1 second. They are resampled and padded to match:

```python
y_8k = librosa.resample(y, orig_sr=16000, target_sr=8000)  # 8000 samples
y_padded = np.pad(y_8k, (0, 8000))                          # zero-pad to 16000
```

---

## Repository Structure

```
src/
  pipeline.py                ← shared load_wav() and wav_to_mel()
  dataset.py                 ← SpeechCommandsDataset, DrinkDataset (used in Phase 3 too)
  model.py                   ← SmallResNet definition
  phase2_pretrain.py         ← Phase 2 training entry point
artifacts/
  phase2_model.pth           ← saved weights after training
  phase2_training_curve.png  ← loss + accuracy over epochs
```

---

## Dataset

- **Source:** Google Speech Commands v2
- **Download:** `torchaudio.datasets.SPEECHCOMMANDS(root, download=True)`
- **Classes:** 35 keywords (~105k utterances total)
- **Preprocessing per sample:** resample 16kHz→8kHz, zero-pad to 16 000 samples, `wav_to_mel()`
- **Splits:** use the official `validation_list.txt` and `testing_list.txt` provided in the dataset

---

## Model Architecture — `SmallResNet`

Defined in `src/model.py`. All ops are DPU-legal (Conv2d, BN, ReLU, MaxPool, GlobalAvgPool, Add, FC).

```
Input: [B, 1, 40, 123]

Stem:    Conv2d(1→16, 3×3, pad=1) → BN → ReLU

Block 1: Conv2d(16→32, 3×3, pad=1) → BN → ReLU
         Conv2d(32→32, 3×3, pad=1) → BN
         skip: Conv2d(16→32, 1×1) → BN
         Add → ReLU → MaxPool2d(2×2)

Block 2: Conv2d(32→64, 3×3, pad=1) → BN → ReLU
         Conv2d(64→64, 3×3, pad=1) → BN
         skip: Conv2d(32→64, 1×1) → BN
         Add → ReLU → MaxPool2d(2×2)

Block 3: Conv2d(64→64, 3×3, pad=1) → BN → ReLU
         Conv2d(64→64, 3×3, pad=1) → BN
         skip: identity
         Add → ReLU

GlobalAvgPool2d → Flatten → FC(64 → n_classes)
```

- ~300K parameters
- Only ReLU (no GELU/Swish) — Vitis AI DPU compatible
- `n_classes=35` for Phase 2; swapped to 10 in Phase 3

---

## Training Configuration

| Hyperparameter | Value |
|---------------|-------|
| Optimiser | AdamW |
| Learning rate | 1e-3 |
| Weight decay | 1e-4 |
| Scheduler | CosineAnnealingLR |
| Epochs | 30 |
| Early stopping patience | 5 (on val loss) |
| Batch size | 64 |
| Loss | CrossEntropyLoss |

---

## Entry Point

```
uv run python -m src.phase2_pretrain
uv run python -m src.phase2_pretrain --data-dir ~/.cache/speechcommands --epochs 30 --batch-size 64
```

---

## Outputs

| File | Description |
|------|-------------|
| `artifacts/phase2_model.pth` | Full model state dict after best val epoch |
| `artifacts/phase2_training_curve.png` | Train/val loss and accuracy curves |

---

## Acceptance Criteria

- Val accuracy on Speech Commands ≥ 70%
- `phase2_model.pth` loadable by Phase 3 without errors
- Training curve shows convergence (no obvious divergence or plateauing at chance)

---

## Notes

- Speech Commands download is ~2.4 GB — ensure disk space
- Training on CPU is feasible but slow; GPU strongly recommended for 30 epochs over 105k samples
