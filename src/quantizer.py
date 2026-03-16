import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pytorch_nndct.apis import torch_quantizer

from config import VitisConfig

class Quantizer:
    def __init__(self, model: nn.Module, cfg: VitisConfig):
        self._model = model
        self._cfg = cfg

    def quantize(self, calib_loader: DataLoader) -> None:

        cfg = self._cfg
        out_dir = str(cfg.artifacts_dir)
        model = self._model.eval()

        x_sample, _ = next(iter(calib_loader))
        x_sample = x_sample[:1]

        # Calibration pass
        quantizer = torch_quantizer("calib", model, x_sample, output_dir=out_dir)
        quant_model = quantizer.quant_model
        quant_model.eval()
        with torch.no_grad():
            for i, (x, _) in enumerate(calib_loader):
                quant_model(x)
                if i + 1 >= cfg.calib_batches:
                    break
        quantizer.export_quant_config()
        print(f"Calibration done, config exported to {out_dir}")

        # Export xmodel
        quantizer = torch_quantizer("test", model, x_sample, output_dir=out_dir)
        quant_model = quantizer.quant_model
        quant_model.eval()
        with torch.no_grad():
            quant_model(x_sample)
        quantizer.export_xmodel(output_dir=out_dir, deploy_check=False)
        print(f"Exported xmodel to {out_dir}")
