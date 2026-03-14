import argparse
import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from src.pipeline import load_wav, wav_to_mel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/")
    parser.add_argument("--artifacts", default="artifacts/")
    args = parser.parse_args()

    data_dir = Path(args.data)
    artifacts_dir = Path(args.artifacts)
    spec_dir = artifacts_dir / "spectrograms"
    spec_dir.mkdir(parents=True, exist_ok=True)

    # Group files by class
    classes: dict[str, list[Path]] = {}
    for f in sorted(data_dir.glob("*.wav")):
        cls = re.sub(r"\d+$", "", f.stem)
        classes.setdefault(cls, []).append(f)

    class_names = sorted(classes)
    print(f"Found {len(class_names)} classes: {class_names}")

    class_means = {}
    for cls in class_names:
        files = classes[cls]
        mels = np.stack([wav_to_mel(load_wav(f)) for f in files])  # [N, 1, 40, 123]
        assert mels.shape[1:2] == (1,) and mels.shape[2] == 40, f"Unexpected shape {mels.shape} for {cls}"
        assert not np.any(np.isneginf(mels)), f"-inf values found in {cls}"
        out = spec_dir / f"{cls}.npy"
        np.save(out, mels)
        print(f"  {cls}: {mels.shape} → {out}")
        class_means[cls] = mels[:, 0].mean(axis=0)  # [40, 123]

    # Visualise
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    for ax, cls in zip(axes.flat, class_names):
        ax.imshow(class_means[cls], origin="lower", aspect="auto")
        ax.set_title(cls)
        ax.axis("off")
    fig.suptitle("Mean mel spectrograms per class")
    fig.tight_layout()
    out_fig = artifacts_dir / "avg_spectrograms.png"
    fig.savefig(out_fig, dpi=120)
    plt.close(fig)
    print(f"Saved figure → {out_fig}")


if __name__ == "__main__":
    main()
