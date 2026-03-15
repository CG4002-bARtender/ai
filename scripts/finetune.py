import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import AudioConfig, ModelConfig, TrainConfig
from audio_loader import AudioLoader
from spectrogram import SpectrogramConverter
from augmentation import Augmenter
from drink_dataset import DrinkDataset
from model import SmallResNet
from trainer import Trainer


def main():
    audio_cfg = AudioConfig()
    model_cfg = ModelConfig()
    train_cfg = TrainConfig()

    train_cfg.artifacts_dir.mkdir(parents=True, exist_ok=True)

    audio_loader = AudioLoader(audio_cfg)
    spec_converter = SpectrogramConverter(audio_cfg)
    augmenter = Augmenter(audio_cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load base model and replace classification head
    model = SmallResNet(n_classes=model_cfg.n_base_classes).to(device)
    trainer = Trainer(model, device, train_cfg)
    trainer.load_checkpoint(train_cfg.artifacts_dir / train_cfg.base_checkpoint)
    model.fc = nn.Linear(64, model_cfg.n_drink_classes)
    nn.init.xavier_uniform_(model.fc.weight)
    nn.init.zeros_(model.fc.bias)
    model = model.to(device)

    train_ds, val_ds, class_names = DrinkDataset.make_splits(
        train_cfg.data_dir, audio_loader, spec_converter, augmenter
    )
    print(f"Classes: {class_names}")
    print(f"Train: {len(train_ds)}  Val: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=2)

    model_ckpt = train_cfg.artifacts_dir / train_cfg.model_checkpoint

    # Stage A: head warm-up (backbone frozen)
    print("\n-- Stage A: head warm-up (backbone frozen) --")
    history_a = trainer.train(
        train_loader, val_loader,
        checkpoint_path=model_ckpt,
        lr=train_cfg.lr_finetune_head,
        epochs=train_cfg.finetune_warmup_epochs,
        freeze_backbone=True,
    )

    # Stage B: full fine-tuning
    print("\n-- Stage B: full fine-tuning --")
    history_b = trainer.train(
        train_loader, val_loader,
        checkpoint_path=model_ckpt,
        lr=train_cfg.lr_finetune_full,
        epochs=train_cfg.finetune_epochs,
        freeze_backbone=False,
    )

    history = {k: history_a[k] + history_b[k] for k in history_a}
    trainer.save_curve(history, train_cfg.artifacts_dir / "finetune_curve.png")

    trainer.load_checkpoint(model_ckpt)
    trainer.save_confusion_matrix(
        val_loader, class_names, train_cfg.artifacts_dir / "confusion_matrix.png"
    )
    print(f"Fine-tuned model saved -> {model_ckpt}")


if __name__ == "__main__":
    main()
