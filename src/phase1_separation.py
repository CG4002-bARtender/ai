"""
Separation analysis for Phase 1 mel spectrograms.

Outputs:
  artifacts/separation_distance.png  — pairwise cosine-distance heatmap between class means
  artifacts/separation_pca.png       — PCA scatter of all samples
  artifacts/separation_tsne.png      — t-SNE scatter of all samples
"""
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize


def load_data(spec_dir: Path):
    """Return X [N, D] float32, labels [N] str, class_names list."""
    class_names, arrays = [], []
    for npy in sorted(spec_dir.glob("*.npy")):
        data = np.load(npy)               # [N, 1, 40, T]
        flat = data.reshape(len(data), -1).astype(np.float32)
        arrays.append(flat)
        class_names.append(npy.stem)
    X = np.concatenate(arrays, axis=0)    # [total_N, D]
    labels = np.concatenate([
        np.full(len(a), cls) for cls, a in zip(class_names, arrays)
    ])
    return X, labels, class_names


def plot_distance_heatmap(class_names, class_means, out_path):
    """Cosine distance between every pair of class mean spectrograms."""
    normed = normalize(class_means)
    D = cdist(normed, normed, metric="cosine")   # [C, C]

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(D, vmin=0, cmap="YlOrRd")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    fig.colorbar(im, ax=ax, label="Cosine distance (lower = more similar)")
    ax.set_title("Pairwise cosine distance between class mean spectrograms\n(small value = classes are hard to separate)")

    # Annotate cells
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, f"{D[i, j]:.3f}", ha="center", va="center",
                    fontsize=7, color="black" if D[i, j] < 0.15 else "white")

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Saved → {out_path}")

    # Print the most confusable pairs
    pairs = []
    for i in range(len(class_names)):
        for j in range(i + 1, len(class_names)):
            pairs.append((D[i, j], class_names[i], class_names[j]))
    pairs.sort()
    print("\nMost similar class pairs (lowest cosine distance):")
    for d, a, b in pairs[:5]:
        print(f"  {a} ↔ {b}  distance={d:.4f}")
    print("\nMost distinct class pairs:")
    for d, a, b in pairs[-3:]:
        print(f"  {a} ↔ {b}  distance={d:.4f}")


def plot_scatter(X_2d, labels, class_names, title, out_path):
    colors = cm.tab10(np.linspace(0, 1, len(class_names)))
    color_map = dict(zip(class_names, colors))

    fig, ax = plt.subplots(figsize=(8, 6))
    for cls in class_names:
        mask = labels == cls
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                   label=cls, color=color_map[cls], s=60, alpha=0.8)
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
    ax.set_title(title)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Saved → {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", default="artifacts/")
    args = parser.parse_args()

    artifacts_dir = Path(args.artifacts)
    spec_dir = artifacts_dir / "spectrograms"

    X, labels, class_names = load_data(spec_dir)
    print(f"Loaded {len(X)} samples, {len(class_names)} classes, feature dim={X.shape[1]}")

    # Class means for distance heatmap
    class_means = np.stack([X[labels == cls].mean(axis=0) for cls in class_names])

    # --- Distance heatmap ---
    plot_distance_heatmap(class_names, class_means, artifacts_dir / "separation_distance.png")

    # --- PCA ---
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    var = pca.explained_variance_ratio_
    plot_scatter(X_pca, labels, class_names,
                 f"PCA — PC1 {var[0]:.1%} var, PC2 {var[1]:.1%} var",
                 artifacts_dir / "separation_pca.png")

    # --- t-SNE ---
    tsne = TSNE(n_components=2, perplexity=min(15, len(X) - 1), random_state=42)
    X_tsne = tsne.fit_transform(X)
    plot_scatter(X_tsne, labels, class_names, "t-SNE", artifacts_dir / "separation_tsne.png")


if __name__ == "__main__":
    main()
