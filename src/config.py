from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class AudioConfig:
    target_sr: int = 8000
    target_samples: int = 16000
    n_fft: int = 256
    hop_length: int = 128
    n_mels: int = 40
    fmin: float = 0.0
    fmax: float = 4000.0


@dataclass
class ModelConfig:
    n_base_classes: int = 35
    n_drink_classes: int = 10
    base_channels: list[int] = field(default_factory=lambda: [16, 32, 64, 64])


@dataclass
class TrainConfig:
    base_epochs: int = 30
    finetune_warmup_epochs: int = 2
    finetune_epochs: int = 100
    batch_size: int = 64
    lr_base: float = 1e-3
    lr_finetune_head: float = 1e-3
    lr_finetune_full: float = 1e-4
    weight_decay: float = 1e-4
    patience: int = 5
    num_workers: int = 4
    speech_commands_cache: Path = field(
        default_factory=lambda: Path.home() / ".cache" / "speechcommands"
    )
    data_dir: Path = field(default_factory=lambda: Path("recordings"))
    artifacts_dir: Path = field(default_factory=lambda: Path("artifacts"))
    base_checkpoint: str = "base.pt"
    model_checkpoint: str = "model.pt"


@dataclass
class VitisConfig:
    target: str = "DPUCZDX8G_ISA1_B4096"  # Ultra96-V2
    calib_batch_size: int = 16
    calib_batches: int = 32
    artifacts_dir: Path = field(default_factory=lambda: Path("artifacts"))
    xmodel_name: str = "drink_classifier"
