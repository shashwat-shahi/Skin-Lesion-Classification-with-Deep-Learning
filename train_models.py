"""
Complete training pipeline for skin lesion classification.

This script trains multiple model variants and compares their performance.
"""

import os
import sys
import torch
import torch.optim as optim
import numpy as np
import pandas as pd
import argparse
from pathlib import Path

from dataset import DatasetBuilder, create_dataloaders
from models import get_model
from training import train, get_predictions, get_confusion_matrix
from visualizations import plot_learning, plot_confusion_matrix, plot_class_distribution


def setup_data(data_dir: str, batch_size: int = 32):
    """Setup dataset and dataloaders."""

    print("\n" + "=" * 80)
    print("SETTING UP DATASET")
    print("=" * 80)

    builder = DatasetBuilder(data_dir)
    dataset_splits, metadata = builder.build(
        preprocessors=[lambda x: x],
        image_size=(224, 224, 3),
        splits={"train": 0.70, "valid": 0.20, "test": 0.10},
        rng_seed=42
    )

    dataloaders = create_dataloaders(dataset_splits, batch_size=batch_size, num_workers=0)
    num_classes = len(metadata["class"].unique())
    class_names = sorted(metadata["class"].unique())

    print(f"\nDataset Statistics:")
    print(f"  Total images: {len(metadata)}")
    print(f"  Number of classes: {num_classes}")
    print(f"  Classes: {', '.join(class_names)}")
    print(f"\nData Split:")
    print(f"  Train: {len(dataset_splits['train'][0])} images")
    print(f"  Valid: {len(dataset_splits['valid'][0])} images")
    print(f"  Test: {len(dataset_splits['test'][0])} images")

    return dataloaders, dataset_splits, metadata, num_classes, class_names


def train_model(
    model_name: str,
    dataloaders: dict,
    num_classes: int,
    class_names: list,
    num_steps: int = 2000,
    learning_rate: float = 0.001,
    batch_size: int = 32,
    save_dir: str = "./models",
    device: torch.device = None
):
    """Train a single model variant."""

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n" + "=" * 80)
    print(f"TRAINING: {model_name.upper()}")
    print("=" * 80)

    # Create model
    if model_name == "simple_cnn":
        model = get_model(model_name, num_classes, dropout_rate=0.0)
    else:
        model = get_model(model_name, num_classes, dropout_rate=0.0)

    model_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel Parameters:")
    print(f"  Total: {model_params:,}")
    print(f"  Trainable: {trainable_params:,}")

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    # Learning rate schedule
    total_steps = num_steps
    warmup_steps = int(0.2 * total_steps)

    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        else:
            return 0.5 * (1 + np.cos(np.pi * (step - warmup_steps) / (total_steps - warmup_steps)))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    print(f"\nTraining Configuration:")
    print(f"  Steps: {num_steps}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Batch size: {batch_size}")
    print(f"  Device: {device}")

    # Train
    model, metrics = train(
        model=model,
        train_loader=dataloaders["train"],
        valid_loader=dataloaders["valid"],
        num_classes=num_classes,
        num_steps=num_steps,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        eval_every=100,
        save_path=os.path.join(save_dir, model_name)
    )

    # Plot training metrics
    print(f"\nPlotting training metrics...")
    os.makedirs(save_dir, exist_ok=True)
    plot_learning(metrics, save_path=f"{save_dir}/{model_name}_learning.png")

    # Evaluate on test set
    print(f"Evaluating on test set...")
    cm = get_confusion_matrix(model, dataloaders["test"], num_classes, device)
    plot_confusion_matrix(cm, class_names, save_path=f"{save_dir}/{model_name}_confusion.png")

    # Calculate per-class metrics
    from sklearn.metrics import precision_recall_fscore_support
    predictions = get_predictions(model, dataloaders["test"], device)
    precision, recall, f1, _ = precision_recall_fscore_support(
        predictions["labels"],
        predictions["preds"],
        labels=range(num_classes),
        average=None
    )

    print(f"\nTest Set Performance:")
    print(f"  {'Class':<30} {'Precision':<12} {'Recall':<12} {'F1':<12}")
    print("-" * 70)
    for idx, class_name in enumerate(class_names):
        print(f"  {class_name:<30} {precision[idx]:.4f}       {recall[idx]:.4f}       {f1[idx]:.4f}")

    weighted_precision = np.average(precision, weights=[np.sum(predictions["labels"] == i) for i in range(num_classes)])
    weighted_recall = np.average(recall, weights=[np.sum(predictions["labels"] == i) for i in range(num_classes)])
    weighted_f1 = np.average(f1, weights=[np.sum(predictions["labels"] == i) for i in range(num_classes)])

    print("-" * 70)
    print(f"  {'Weighted Average':<30} {weighted_precision:.4f}       {weighted_recall:.4f}       {weighted_f1:.4f}")

    results = {
        "model": model_name,
        "precision": weighted_precision,
        "recall": weighted_recall,
        "f1": weighted_f1,
        "params": trainable_params
    }

    return model, results


def main():
    """Main training pipeline."""

    parser = argparse.ArgumentParser(description="Train skin cancer detection models")
    parser.add_argument("--data_dir", type=str, default="./data/raw", help="Path to raw data")
    parser.add_argument("--save_dir", type=str, default="./models", help="Path to save models")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--num_steps", type=int, default=2000, help="Number of training steps")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")

    args = parser.parse_args()

    # Configuration
    data_dir = args.data_dir
    save_dir = args.save_dir
    batch_size = args.batch_size
    num_steps = args.num_steps
    learning_rate = args.learning_rate
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Models to train
    models_to_train = [
        "simple_cnn",
        "resnet_scratch",
        "resnet_head",
        "resnet_finetuned"
    ]

    # Setup data
    dataloaders, dataset_splits, metadata, num_classes, class_names = setup_data(data_dir, batch_size)

    # Visualize data
    print("\n" + "=" * 80)
    print("GENERATING DATA VISUALIZATIONS")
    print("=" * 80)
    print("Plotting class distribution...")
    plot_class_distribution(metadata, save_path=f"{save_dir}/class_distribution.png")

    # Train all models
    all_results = []

    for model_name in models_to_train:
        model, results = train_model(
            model_name=model_name,
            dataloaders=dataloaders,
            num_classes=num_classes,
            class_names=class_names,
            num_steps=num_steps,
            learning_rate=learning_rate,
            batch_size=batch_size,
            save_dir=save_dir,
            device=device
        )

        all_results.append(results)

    # Summary comparison
    print("\n" + "=" * 80)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 80)
    print(f"\n{'Model':<25} {'Precision':<15} {'Recall':<15} {'F1':<15} {'Params':<15}")
    print("-" * 85)

    for result in all_results:
        print(f"{result['model']:<25} {result['precision']:<15.4f} {result['recall']:<15.4f} {result['f1']:<15.4f} {result['params']:<15,}")

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80)
    print(f"Results saved to: {save_dir}")
    print(f"Check {save_dir}/ for:")
    print(f"  - {model_name}_learning.png: Training curves")
    print(f"  - {model_name}_confusion.png: Confusion matrices")
    print(f"  - {model_name}/model.pth: Trained weights")


if __name__ == "__main__":
    main()
