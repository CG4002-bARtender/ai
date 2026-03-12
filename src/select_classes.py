"""
Class selection evaluator.

Given a set of candidate drink-name classes (each with WAV recordings in
data/candidates/<class_name>/), this script:

  1. Runs the full pipeline on each WAV to generate spectrograms.
  2. Computes a mean spectrogram per class.
  3. Builds a pairwise cosine-distance matrix across all candidate classes.
  4. Uses greedy max-min selection to choose the 10 most separable classes.
  5. Saves a heatmap and a grid of mean spectrograms to artifacts/.
  6. Prints the selected class names and the separation score.

Usage:
    uv run python -m src.select_classes
    uv run python -m src.select_classes --candidates data/candidates --k 10
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics.pairwise import cosine_distances

from src.pipeline import load_wav, run_pipeline, save_artifacts

MIN_SAMPLES_WARNING = 10
SEPARATION_THRESHOLD = 0.10


def load_class_spectrograms(candidates_dir: str) -> dict[str, np.ndarray]:
    """
    Walk candidates_dir. For each sub-folder (= one candidate class),
    load all .wav files and run the pipeline.

    Returns:
        { class_name: np.ndarray of shape [N, 4920] }
        where 4920 = 40 * 123 (flattened spectrogram vectors)
    """
    class_data: dict[str, np.ndarray] = {}

    class_names = sorted(
        d for d in os.listdir(candidates_dir)
        if os.path.isdir(os.path.join(candidates_dir, d))
    )

    if not class_names:
        sys.exit(f"No sub-folders found in {candidates_dir}. "
                 "Create one folder per candidate class containing .wav files.")

    for name in class_names:
        folder = os.path.join(candidates_dir, name)
        wav_paths = [
            os.path.join(folder, f)
            for f in sorted(os.listdir(folder))
            if f.lower().endswith(".wav")
        ]

        if not wav_paths:
            print(f"  [skip] {name}: no .wav files found")
            continue

        if len(wav_paths) < MIN_SAMPLES_WARNING:
            print(f"  [warn] {name}: only {len(wav_paths)} sample(s) — "
                  f"mean spectrogram may not be representative (recommend ≥{MIN_SAMPLES_WARNING})")

        vectors = []
        for path in wav_paths:
            try:
                pcm = load_wav(path)
                spec = run_pipeline(pcm)        # [1, 40, 123]
                vectors.append(spec.flatten())  # [4920]
            except Exception as e:
                print(f"  [warn] skipping {path}: {e}")

        if vectors:
            class_data[name] = np.stack(vectors)  # [N, 4920]
            print(f"  {name}: {len(vectors)} sample(s) loaded")

    return class_data


def compute_class_means(class_data: dict[str, np.ndarray]) -> tuple[list[str], np.ndarray]:
    """
    Returns:
        names: list of class names (ordered)
        means: float64 array of shape [N_classes, 4920]
    """
    names = sorted(class_data.keys())
    means = np.stack([class_data[n].mean(axis=0) for n in names]).astype(np.float64)
    return names, means


def greedy_max_min_select(dist_matrix: np.ndarray, k: int) -> list[int]:
    """
    Select k indices from dist_matrix that maximise the minimum pairwise
    distance within the selected set (max-min diversity / dispersion).

    Seeds with the pair of classes that have the largest distance, then
    greedily adds the class whose minimum distance to the already-selected
    set is largest.
    """
    n = dist_matrix.shape[0]
    k = min(k, n)

    # Seed: pair with largest pairwise distance
    flat_idx = np.argmax(np.triu(dist_matrix, k=1))
    i, j = divmod(flat_idx, n)
    selected = [i, j]

    while len(selected) < k:
        # min distance from each candidate to the current selected set
        min_dists = dist_matrix[:, selected].min(axis=1)
        min_dists[selected] = -1.0  # mask already-selected
        selected.append(int(np.argmax(min_dists)))

    return selected


def plot_distance_heatmap(
    dist_matrix: np.ndarray,
    names: list[str],
    selected_indices: list[int],
    output_path: str,
) -> None:
    fig, ax = plt.subplots(figsize=(max(8, len(names)), max(7, len(names) - 1)))
    im = ax.imshow(dist_matrix, cmap="viridis", vmin=0)
    fig.colorbar(im, ax=ax, label="Cosine distance")

    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(names, fontsize=9)

    # Highlight selected classes with a border
    for idx in selected_indices:
        ax.add_patch(plt.Rectangle(
            (idx - 0.5, idx - 0.5), 1, 1,
            fill=False, edgecolor="red", linewidth=2,
        ))

    ax.set_title("Pairwise cosine distance — selected classes outlined in red")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Saved: {output_path}")


def plot_mean_spectrograms(
    class_data: dict[str, np.ndarray],
    selected_names: list[str],
    output_path: str,
) -> None:
    k = len(selected_names)
    cols = 5
    rows = (k + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 2.5))
    axes = np.array(axes).flatten()

    for ax, name in zip(axes, selected_names):
        mean_vec = class_data[name].mean(axis=0)          # [4920]
        mean_spec = mean_vec.reshape(40, 123)              # [40, 123]
        ax.imshow(mean_spec, origin="lower", aspect="auto", cmap="magma")
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("Frame")
        ax.set_ylabel("Mel bin")

    # Hide any unused subplots
    for ax in axes[k:]:
        ax.axis("off")

    fig.suptitle("Mean mel-spectrograms — selected classes", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Saved: {output_path}")


def select_classes(candidates_dir: str, k: int, artifacts_dir: str) -> list[str]:
    os.makedirs(artifacts_dir, exist_ok=True)

    print(f"\n[1/4] Loading spectrograms from {candidates_dir}/")
    class_data = load_class_spectrograms(candidates_dir)

    if len(class_data) < k:
        sys.exit(
            f"Only {len(class_data)} classes with audio found — need at least {k}. "
            "Add more candidate classes or lower --k."
        )

    print(f"\n[2/4] Computing class mean spectrograms ({len(class_data)} classes)")
    names, means = compute_class_means(class_data)

    print(f"\n[3/4] Building {len(names)}×{len(names)} cosine distance matrix")
    dist_matrix = cosine_distances(means)

    print(f"\n[4/4] Greedy max-min selection (k={k})")
    selected_indices = greedy_max_min_select(dist_matrix, k)
    selected_names = [names[i] for i in selected_indices]

    # Separation score = minimum pairwise distance within the selected set
    selected_dists = dist_matrix[np.ix_(selected_indices, selected_indices)]
    np.fill_diagonal(selected_dists, np.inf)
    separation_score = selected_dists.min()

    # Save artifacts
    heatmap_path = os.path.join(artifacts_dir, "class_distance_matrix.png")
    spectrogram_path = os.path.join(artifacts_dir, "selected_spectrograms.png")
    plot_distance_heatmap(dist_matrix, names, selected_indices, heatmap_path)
    plot_mean_spectrograms(class_data, selected_names, spectrogram_path)
    save_artifacts(artifacts_dir)

    # Report
    print("\n" + "=" * 60)
    print(f"Selected {k} classes  |  separation score: {separation_score:.4f}")
    if separation_score < SEPARATION_THRESHOLD:
        print(
            f"  WARNING: score {separation_score:.4f} < {SEPARATION_THRESHOLD} threshold.\n"
            "  Classes may be too phonetically similar. Consider expanding\n"
            "  the candidate pool or recording more samples per class."
        )
    print("=" * 60)
    print("\nSelected class names (copy-paste for Phase 2):\n")
    for idx, name in enumerate(selected_names):
        print(f"  {idx:2d}  {name}")
    print()

    return selected_names


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select k most separable drink-name classes from candidate WAV recordings."
    )
    parser.add_argument(
        "--candidates", default="data/candidates",
        help="Directory with one sub-folder per candidate class (default: data/candidates)"
    )
    parser.add_argument(
        "--k", type=int, default=10,
        help="Number of classes to select (default: 10)"
    )
    parser.add_argument(
        "--artifacts", default="artifacts",
        help="Directory for output artifacts (default: artifacts)"
    )
    args = parser.parse_args()

    select_classes(args.candidates, args.k, args.artifacts)


if __name__ == "__main__":
    main()
