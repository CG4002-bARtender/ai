# Drink Classifier Pipeline

PyTorch → Vitis AI quantization → `.xmodel` for deployment on Ultra96-V2 DPU.

## Prerequisites

### 1. Docker Engine

1. Run `wsl --install` in PowerShell (enables WSL2, required for Docker's GPU support)
2. Install [Docker Desktop for Windows](https://docs.docker.com/desktop/install/windows-install/) — select WSL2 backend during setup
3. Open PowerShell and run `docker run hello-world`

### 2. NVIDIA Container Toolkit (GPU training)

Install the [NVIDIA driver for WSL](https://developer.nvidia.com/cuda/wsl) on the Windows host (not inside WSL). No separate `nvidia-container-toolkit` install needed.

### 3. SSH config for Ultra96

Add the following to `C:\Users\<you>\.ssh\config` (create the file if it doesn't exist):

```
Host ultra96
  HostName makerslab-fpga-40.ddns.comp.nus.edu.sg
  User xilinx
```

Then `make deploy` and `ssh ultra96` will resolve without needing the full hostname.

#### 4. Install Make

`make` requires WSL2 or [chocolatey](https://chocolatey.org/): `choco install make`. Docker Desktop handles the containers natively.

## Quickstart

```bash
make pipeline   
```
## Individual stages

Run any stage independently — useful when iterating on a specific step:

| Command | Container | Output |
|---|---|---|
| `make build` | — | builds both Docker images |
| `make train` | trainer (GPU) | `artifacts/base.pt` |
| `make finetune` | trainer (GPU) | `artifacts/model.pt` |
| `make quantize` | vitis (CPU) | `artifacts/SmallResNet_int.xmodel` |
| `make compile` | vitis (CPU) | `artifacts/drink_classifier.xmodel` |
| `make deploy` | — | copies xmodel to Ultra96 |
| `make clean` | — | tears down containers and volumes |

## Data

Place custom drink audio samples in `data/audio/raw/` before running `make finetune`.

SpeechCommands is downloaded automatically on first `make train` and cached in `~/.cache/speechcommands`.

## Retargeting to a different board

Update `arch/DPUCZDX8G/Ultra96/arch.json` with the correct DPU fingerprint, then:
