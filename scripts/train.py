import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from torch.utils.data import DataLoader

from config import AudioConfig, ModelConfig, TrainConfig
from spectrogram import SpectrogramConverter
from speech_commands_dataset import SpeechCommandsDataset
from model import SmallResNet
from trainer import Trainer


def main():
    audio_cfg = AudioConfig()
    model_cfg = ModelConfig()
    train_cfg = TrainConfig()

    train_cfg.artifacts_dir.mkdir(parents=True, exist_ok=True)

    spec_converter = SpectrogramConverter(audio_cfg)

    print("Loading SpeechCommands dataset...")
    train_ds = SpeechCommandsDataset(
        train_cfg.speech_commands_cache, "training", audio_cfg, spec_converter
    )
    val_ds = SpeechCommandsDataset(
        train_cfg.speech_commands_cache, "validation", audio_cfg, spec_converter
    )
    print(f"  Train: {len(train_ds):,}  Val: {len(val_ds):,}  Classes: {len(train_ds.classes)}")

    train_loader = DataLoader(
        train_ds, batch_size=train_cfg.batch_size, shuffle=True,
        num_workers=train_cfg.num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=train_cfg.batch_size, shuffle=False,
        num_workers=train_cfg.num_workers, pin_memory=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = SmallResNet(n_classes=model_cfg.n_base_classes, channels=model_cfg.base_channels).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    trainer = Trainer(model, device, train_cfg)
    checkpoint = train_cfg.artifacts_dir / train_cfg.base_checkpoint
    history = trainer.train(
        train_loader, val_loader,
        checkpoint_path=checkpoint,
        lr=train_cfg.lr_base,
        epochs=train_cfg.base_epochs,
    )
    trainer.save_curve(history, train_cfg.artifacts_dir / "train_curve.png")
    print(f"Base model saved -> {checkpoint}")


if __name__ == "__main__":
    main()
