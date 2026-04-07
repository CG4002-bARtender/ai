import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from torch.utils.data import DataLoader

from config import AudioConfig, ModelConfig, TrainConfig, VitisConfig
from audio_loader import AudioLoader
from spectrogram import SpectrogramConverter
from augmentation import Augmenter
from combined_dataset import make_combined_splits
from model import SmallResNet
from trainer import Trainer
from quantizer import Quantizer


def main():
    audio_cfg = AudioConfig()
    model_cfg = ModelConfig()
    train_cfg = TrainConfig()
    vitis_cfg = VitisConfig()

    audio_loader = AudioLoader(audio_cfg)
    spec_converter = SpectrogramConverter(audio_cfg)
    augmenter = Augmenter(audio_cfg)

    device = torch.device("cpu")  # Vitis AI quantization runs on CPU
    model = SmallResNet(n_classes=model_cfg.n_drink_classes, channels=model_cfg.base_channels).to(device)
    trainer = Trainer(model, device, train_cfg)
    trainer.load_checkpoint(train_cfg.artifacts_dir / train_cfg.model_checkpoint)
    model.eval()

    _, val_ds, _ = make_combined_splits(
        [train_cfg.data_dir], audio_loader, spec_converter, augmenter
    )
    calib_loader = DataLoader(
        val_ds, batch_size=vitis_cfg.calib_batch_size, shuffle=False, num_workers=2
    )

    print("Quantizing model...")
    quantizer = Quantizer(model, vitis_cfg)
    quantizer.quantize(calib_loader)


if __name__ == "__main__":
    main()
