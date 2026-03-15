import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config import VitisConfig
from compiler import Compiler


def main():
    vitis_cfg = VitisConfig()
    # torch_quantizer names the xmodel after the model class
    xmodel_path = vitis_cfg.artifacts_dir / "SmallResNet_int.xmodel"

    compiler = Compiler(vitis_cfg)
    output = compiler.compile(xmodel_path)
    print(f"Compilation complete -> {output}")


if __name__ == "__main__":
    main()
