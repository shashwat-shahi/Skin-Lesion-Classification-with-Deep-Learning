import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from typing import Dict, Tuple, Optional, List
from sklearn.metrics import precision_score, recall_score, confusion_matrix
import json
import os


class MetricsLogger:
    """Logs metrics during training."""

    def __init__(self):
        self.logs = {}
        self.current_step_logs = {}

    def log_step(self, split: str, **kwargs):
        """Log metrics for a step."""
        if split not in self.logs:
            self.logs[split] = {k: [] for k in kwargs.keys()}

        for key, value in kwargs.items():
            if isinstance(value, torch.Tensor):
                value = value.item()
            self.logs[split][key].append(value)

    def latest(self, keys: List[str]) -> str:
        """Get latest values as string."""
        result = []
        for split in ["train", "valid"]:
            if split in self.logs:
                for key in keys:
                    if key in self.logs[split] and len(self.logs[split][key]) > 0:
                        val = self.logs[split][key][-1]
                        result.append(f"{split}_{key}={val:.4f}")
        return " | ".join(result)

    def export(self) -> Dict:
        """Export logs."""
        return self.logs


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int
) -> Dict[str, float]:
    """Compute weighted precision and recall."""

    # Ensure arrays are numpy
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()

    # Flatten if needed
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    recall = recall_score(
        y_true, y_pred,
        labels=range(num_classes),
        average="weighted",
        zero_division=0
    )
    precision = precision_score(
        y_true, y_pred,
        labels=range(num_classes),
        average="weighted",
        zero_division=0
    )

    return {
        "recall_weighted": recall,
        "precision_weighted": precision
    }


def train_step(
    model: nn.Module,
    batch: Dict,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    num_classes: int
) -> Dict[str, float]:
    """Perform a single training step."""

    model.train()

    images = batch["image"].to(device)
    labels = batch["label"].to(device)

    # Forward pass
    outputs = model(images)
    loss = criterion(outputs, labels)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Compute metrics
    with torch.no_grad():
        preds = torch.argmax(outputs, dim=1)
        metrics = compute_metrics(labels.cpu().numpy(), preds.cpu().numpy(), num_classes)

    metrics["loss"] = loss.item()

    return metrics


@torch.no_grad()
def eval_step(
    model: nn.Module,
    batch: Dict,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int
) -> Dict[str, float]:
    """Perform evaluation on a batch."""

    model.eval()

    images = batch["image"].to(device)
    labels = batch["label"].to(device)

    # Forward pass
    outputs = model(images)
    loss = criterion(outputs, labels)

    # Compute metrics
    preds = torch.argmax(outputs, dim=1)
    metrics = compute_metrics(labels.cpu().numpy(), preds.cpu().numpy(), num_classes)

    metrics["loss"] = loss.item()

    return metrics


def train(
    model: nn.Module,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    num_classes: int,
    num_steps: int,
    optimizer: optim.Optimizer,
    scheduler: Optional[object] = None,
    device: torch.device = None,
    eval_every: int = 10,
    save_path: Optional[str] = None
) -> Tuple[nn.Module, Dict]:
    """Train the model."""

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    metrics_logger = MetricsLogger()

    # Create iterator
    train_iter = iter(train_loader)

    steps = tqdm(range(num_steps))
    for step in steps:
        steps.set_description(f"Step {step + 1}")

        # Get batch
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        # Train step
        batch_metrics = train_step(model, batch, criterion, optimizer, device, num_classes)
        metrics_logger.log_step(split="train", **batch_metrics)

        # Update learning rate
        if scheduler is not None:
            scheduler.step()

        # Evaluation
        if step % eval_every == 0:
            for batch in valid_loader:
                batch_metrics = eval_step(model, batch, criterion, device, num_classes)
                metrics_logger.log_step(split="valid", **batch_metrics)

        steps.set_postfix_str(metrics_logger.latest(["loss"]))

    # Save model
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(save_path, "model.pth"))

    return model, metrics_logger.export()


def get_predictions(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device = None
) -> Dict:
    """Get predictions on entire dataset."""

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    model.eval()

    all_preds = []
    all_labels = []
    all_frame_ids = []

    with torch.no_grad():
        for batch in data_loader:
            images = batch["image"].to(device)
            labels = batch["label"]
            frame_ids = batch["frame_id"]

            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.numpy())
            all_frame_ids.extend(frame_ids if isinstance(frame_ids, list) else frame_ids.tolist())

    predictions = {
        "preds": np.concatenate(all_preds),
        "labels": np.concatenate(all_labels),
        "frame_ids": all_frame_ids
    }

    return predictions


def get_confusion_matrix(
    model: nn.Module,
    data_loader: DataLoader,
    num_classes: int,
    device: torch.device = None
) -> np.ndarray:
    """Get confusion matrix."""

    predictions = get_predictions(model, data_loader, device)

    cm = confusion_matrix(
        predictions["labels"],
        predictions["preds"],
        labels=range(num_classes)
    )

    return cm
