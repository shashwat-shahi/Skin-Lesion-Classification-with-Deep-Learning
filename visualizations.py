import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from typing import Dict, Optional, List
from sklearn.metrics import confusion_matrix as sklearn_cm
import os


def plot_learning(metrics: Dict[str, List[float]], save_path: Optional[str] = None):
    """Plot training and validation metrics."""

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss plot
    if "train" in metrics and "loss" in metrics["train"]:
        axes[0].plot(metrics["train"]["loss"], label="Train Loss", alpha=0.7)
    if "valid" in metrics and "loss" in metrics["valid"]:
        axes[0].plot(
            range(0, len(metrics["valid"]["loss"]) * 10, 10),
            metrics["valid"]["loss"],
            label="Valid Loss",
            alpha=0.7
        )
    axes[0].set_xlabel("Steps")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss over Training Steps")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Precision and Recall plot
    if "valid" in metrics:
        if "precision_weighted" in metrics["valid"]:
            axes[1].plot(
                range(0, len(metrics["valid"]["precision_weighted"]) * 10, 10),
                metrics["valid"]["precision_weighted"],
                label="Precision",
                alpha=0.7
            )
        if "recall_weighted" in metrics["valid"]:
            axes[1].plot(
                range(0, len(metrics["valid"]["recall_weighted"]) * 10, 10),
                metrics["valid"]["recall_weighted"],
                label="Recall",
                alpha=0.7
            )
    axes[1].set_xlabel("Steps")
    axes[1].set_ylabel("Score")
    axes[1].set_title("Validation Metrics over Training Steps")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim([0, 1])

    plt.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.show()


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    save_path: Optional[str] = None
):
    """Plot confusion matrix."""

    fig, ax = plt.subplots(figsize=(12, 10))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        cbar_kws={"label": "Count"}
    )

    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Confusion Matrix")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.show()


def plot_class_distribution(
    metadata: pd.DataFrame,
    save_path: Optional[str] = None
):
    """Plot class distribution."""

    fig, ax = plt.subplots(figsize=(10, 6))

    class_counts = metadata["class"].value_counts().sort_values()
    class_counts.plot(kind="barh", ax=ax)

    ax.set_xlabel("Number of Images")
    ax.set_ylabel("Class")
    ax.set_title("Class Distribution in Dataset")
    plt.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.show()


def plot_split_distribution(
    metadata: pd.DataFrame,
    save_path: Optional[str] = None
):
    """Plot split distribution across classes."""

    fig, ax = plt.subplots(figsize=(12, 6))

    split_dist = pd.crosstab(metadata["class"], metadata["split"])
    split_dist.plot(kind="bar", ax=ax)

    ax.set_xlabel("Class")
    ax.set_ylabel("Number of Images")
    ax.set_title("Data Split Distribution by Class")
    ax.legend(title="Split")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.show()


def plot_random_images(
    metadata: pd.DataFrame,
    num_images: int = 9,
    save_path: Optional[str] = None
):
    """Plot random images from dataset."""

    from PIL import Image

    num_cols = 3
    num_rows = (num_images + num_cols - 1) // num_cols

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 4 * num_rows))
    axes = axes.flatten() if num_images > 1 else [axes]

    for idx in range(num_images):
        if idx < len(metadata):
            row = metadata.iloc[idx]
            try:
                img = Image.open(row["full_path"]).convert("RGB")
                axes[idx].imshow(img)
                axes[idx].set_title(f'{row["class"]}\n({row["split"]})')
                axes[idx].axis("off")
            except Exception as e:
                axes[idx].text(0.5, 0.5, f"Error: {e}", ha="center", va="center")
                axes[idx].axis("off")

    for idx in range(num_images, len(axes)):
        axes[idx].axis("off")

    plt.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.show()


def plot_sample_per_class(
    metadata: pd.DataFrame,
    ncols: int = 3,
    save_path: Optional[str] = None
):
    """Plot one random sample per class."""

    from PIL import Image

    unique_classes = metadata["class"].unique()
    nrows = (len(unique_classes) + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows))
    axes = axes.flatten()

    for idx, class_name in enumerate(sorted(unique_classes)):
        class_samples = metadata[metadata["class"] == class_name]
        sample = class_samples.sample(1).iloc[0]

        try:
            img = Image.open(sample["full_path"]).convert("RGB")
            axes[idx].imshow(img)
            axes[idx].set_title(class_name.capitalize())
            axes[idx].axis("off")
        except Exception as e:
            axes[idx].text(0.5, 0.5, f"Error: {e}", ha="center", va="center")
            axes[idx].axis("off")

    for idx in range(len(unique_classes), len(axes)):
        axes[idx].axis("off")

    plt.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.show()
