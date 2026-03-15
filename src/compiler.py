import subprocess
from pathlib import Path

from config import VitisConfig


class Compiler:
    def __init__(self, cfg: VitisConfig):
        self._cfg = cfg

    def compile(self, xmodel_path: Path) -> Path:
        cfg = self._cfg
        out_dir = cfg.artifacts_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            "vai_c_xir",
            "--xmodel", str(xmodel_path),
            "--arch", "/opt/vitis_ai/compiler/arch/DPUCZDX8G/Ultra96/arch.json",
            "--output_dir", str(out_dir),
            "--net_name", cfg.xmodel_name,
        ]
        print(f"Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)

        output = out_dir / f"{cfg.xmodel_name}.xmodel"
        if not output.exists():
            raise RuntimeError(f"Compilation appeared to succeed but output not found: {output}")
        print(f"Compiled -> {output}")
        return output
