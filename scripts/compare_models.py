"""
Trains both SmallResNet (original) and LargerResNet (new) on the same drink
dataset using the same base checkpoint, then prints a side-by-side accuracy
comparison so we can see whether the bigger model actually helps.

Uses the existing base.pt (already trained on Speech Commands) as the
starting point for both, so the only variable is model size.
"""
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

CONFIGS = {
    "small  (original)": [16, 32, 64, 64],
    "larger (new)     ": [32, 64, 128, 128],
}


def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total if total > 0 else 0.0


def run_config(name, channels, audio_cfg, model_cfg, train_cfg, device,
               train_loader, val_loader):
    print(f"\n{'='*55}")
    print(f"  {name}  channels={channels}")
    print(f"{'='*55}")

    model = SmallResNet(n_classes=model_cfg.n_base_classes,
                        channels=channels).to(device)

    # Load base checkpoint — must use matching architecture
    base_ckpt = train_cfg.artifacts_dir / train_cfg.base_checkpoint
    trainer = Trainer(model, device, train_cfg)

    # The saved base.pt was trained with [32,64,128,128].
    # For the small model we retrain from scratch on drink data only.
    if channels == [32, 64, 128, 128]:
        print("  Loading base.pt pretrained weights...")
        trainer.load_checkpoint(base_ckpt)
    else:
        print("  No base checkpoint (different arch) — fine-tuning from random init")

    # Replace head
    model.fc = nn.Linear(model.n_features, model_cfg.n_drink_classes)
    nn.init.xavier_uniform_(model.fc.weight)
    nn.init.zeros_(model.fc.bias)
    model = model.to(device)

    tmp_ckpt = train_cfg.artifacts_dir / f"tmp_{name.strip()}.pt"

    # Stage A: head warmup
    print("  Stage A: head warmup (2 epochs)...")
    trainer.train(train_loader, val_loader,
                  checkpoint_path=tmp_ckpt,
                  lr=train_cfg.lr_finetune_head,
                  epochs=train_cfg.finetune_warmup_epochs,
                  freeze_backbone=True)

    # Stage B: full fine-tune
    print("  Stage B: full fine-tune (up to 100 epochs, early stop patience=5)...")
    history = trainer.train(train_loader, val_loader,
                            checkpoint_path=tmp_ckpt,
                            lr=train_cfg.lr_finetune_full,
                            epochs=train_cfg.finetune_epochs,
                            freeze_backbone=False)

    # Load best checkpoint and evaluate
    trainer.load_checkpoint(tmp_ckpt)
    val_acc = evaluate(model, val_loader, device)
    best_val = max(history["val_acc"])
    params = sum(p.numel() for p in model.parameters())

    print(f"\n  >> Parameters : {params:,}")
    print(f"  >> Best val acc (during training) : {best_val:.1%}")
    print(f"  >> Final val acc (best checkpoint): {val_acc:.1%}")

    # Clean up temp checkpoint
    tmp_ckpt.unlink(missing_ok=True)

    return {"params": params, "best_val": best_val, "final_val": val_acc}


def main():
    audio_cfg = AudioConfig()
    model_cfg = ModelConfig()
    train_cfg = TrainConfig()

    audio_loader = AudioLoader(audio_cfg)
    spec_converter = SpectrogramConverter(audio_cfg)
    augmenter = Augmenter(audio_cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_ds, val_ds, class_names = DrinkDataset.make_splits(
        train_cfg.data_dir, audio_loader, spec_converter, augmenter
    )
    print(f"Classes ({len(class_names)}): {class_names}")
    print(f"Train: {len(train_ds)}  Val: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=2)

    results = {}
    for name, channels in CONFIGS.items():
        results[name] = run_config(
            name, channels, audio_cfg, model_cfg, train_cfg,
            device, train_loader, val_loader
        )

    print(f"\n\n{'='*55}")
    print("  COMPARISON SUMMARY")
    print(f"{'='*55}")
    print(f"  {'Model':<24} {'Params':>10}  {'Best val':>9}  {'Final val':>9}")
    print(f"  {'-'*52}")
    for name, r in results.items():
        print(f"  {name:<24} {r['params']:>10,}  {r['best_val']:>9.1%}  {r['final_val']:>9.1%}")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
