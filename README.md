# Drink Classifier Pipeline

PyTorch → Vitis AI quantization → `.xmodel` for deployment on Ultra96-V2 DPU.

## Prerequisites

- Docker + Docker Compose
- `nvidia-container-toolkit` (for GPU training)
- SSH access to the Ultra96 board

## Pipeline

Each stage depends on the previous one's output in `artifacts/`.

| Command | Description |
|---|---|
| `make train` | Train base model on SpeechCommands → `artifacts/base.pt` |
| `make finetune` | Finetune on custom drink audio dataset → `artifacts/model.pt` |
| `make quantize` | Quantize model using Vitis AI → `artifacts/SmallResNet_int.xmodel` |
| `make compile` | Compile to DPU target → `artifacts/drink_classifier.xmodel` |
| `make deploy` | SCP `.xmodel` to Ultra96 at `~/models/` |

## Run the full pipeline

```bash
make build     
make train
make finetune
make quantize
make compile
make deploy
```

## Other commands

```bash
make build    # rebuild Docker images after code/dependency changes
make clean    # tear down all containers and volumes
```

## Data

Place custom drink audio samples in `data/audio/raw/` before running `make finetune`.

SpeechCommands dataset is downloaded automatically on first `make train` and cached in `~/.cache/speechcommands`.

## Board target

The model is compiled for `DPUCZDX8G_ISA1_B4096` (Ultra96-V2). To target a different board, update `arch/DPUCZDX8G/Ultra96/arch.json` with the correct fingerprint and run `make build compile deploy`.
