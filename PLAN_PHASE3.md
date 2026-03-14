# Phase 3 — Transfer Learning / Drink Name Classifier

## Goal

Load the Phase 2 weights, replace the classification head with a 10-class output,
and fine-tune on the 10 drink-name classes from `data/`. Evaluate and save the final model.

---

## Audio Standard

Same as Phases 1 and 2 — 8 kHz, 2 seconds, output shape `[1, 40, 123] float32`.
Input data is already in the correct format in `data/`.

---

## Repository Structure

```
data/
  negroni00.wav ... negroni04.wav
  martini00.wav ... martini04.wav
  ...                              ← 10 classes × 5 samples = 50 total
src/
  pipeline.py                      ← shared load_wav() and wav_to_mel()
  dataset.py                       ← DrinkDataset (with augmentation)
  model.py                         ← SmallResNet (same architecture, n_classes param)
  phase3_finetune.py               ← Phase 3 entry point
artifacts/
  phase2_model.pth                 ← input: Phase 2 weights
  phase3_model.pth                 ← output: fine-tuned weights
  phase3_training_curve.png
  phase3_confusion_matrix.png
```

---

## Classes

Inferred from filename prefixes (same logic as Phase 1):

```
godfather, irishcoffee, martini, midorisour, negroni,
oldfashioned, scotchneat, tuxedo, vodkaneat, whiskeyneat
```

---

## Dataset & Augmentation

With only 5 samples per class (50 total), augmentation is critical.

Applied on-the-fly in `DrinkDataset` during training:

| Augmentation | Parameters |
|-------------|------------|
| Time shift | ±10% of clip (±1 600 samples), circular shift |
| Gaussian noise | SNR randomly sampled 20–40 dB |
| Pitch shift | ±1 semitone via `librosa.effects.pitch_shift` |

Each augmentation is applied independently with 50% probability, giving ~4–5× effective data.

**Split:** 80/20 train/val, stratified by class (4 train / 1 val per class before augmentation).

---

## Transfer Strategy

### Stage A — Head warmup (2 epochs)
- Load `artifacts/phase2_model.pth`
- Replace `FC(64 → 35)` with `FC(64 → 10)` (random init)
- **Freeze all layers except the new FC head**
- Train for 2 epochs to bring the head to a reasonable starting point
- lr = 1e-3

### Stage B — Full fine-tuning (up to 20 epochs)
- **Unfreeze all layers**
- Train end-to-end with a low learning rate
- lr = 1e-4, CosineAnnealingLR
- Early stopping on val loss, patience = 5

---

## Training Configuration

| Hyperparameter | Stage A | Stage B |
|---------------|---------|---------|
| Epochs | 2 | up to 20 |
| Learning rate | 1e-3 | 1e-4 |
| Optimiser | AdamW | AdamW |
| Weight decay | 1e-4 | 1e-4 |
| Scheduler | — | CosineAnnealingLR |
| Batch size | 16 | 16 |
| Loss | CrossEntropyLoss | CrossEntropyLoss |

---

## Entry Point

```
uv run python -m src.phase3_finetune
uv run python -m src.phase3_finetune --data data/ --weights artifacts/phase2_model.pth
```

---

## Outputs

| File | Description |
|------|-------------|
| `artifacts/phase3_model.pth` | Fine-tuned weights (10-class head) |
| `artifacts/phase3_training_curve.png` | Train/val loss and accuracy curves |
| `artifacts/phase3_confusion_matrix.png` | Confusion matrix on val set |

Console output: per-class accuracy and overall accuracy on val set.

---

## Acceptance Criteria

- Val accuracy ≥ 70% on the 10 drink classes
- Confusion matrix saved and readable
- `phase3_model.pth` loads cleanly with `SmallResNet(n_classes=10)`

---

## Notes

- 50 samples is a very small fine-tuning set. If val accuracy is unsatisfactory,
  the primary remedy is collecting more recordings — not changing the model.
- Vitis AI INT8 quantization is a separate post-Phase-3 step and not covered here.
- The fine-tuned model is the direct input to future quantization and DPU deployment.
