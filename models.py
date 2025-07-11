import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional


class CnnBackbone(nn.Module):
    """Compact convolutional feature extractor for image data."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.fc = nn.Linear(64 * 56 * 56, 256)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class SkinLesionClassifierHead(nn.Module):
    """Skin lesion classification MLP head."""

    def __init__(self, num_classes: int, dropout_rate: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(256, 256)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=dropout_rate)

        self.fc2 = nn.Linear(256, 128)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=dropout_rate)

        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)

        x = self.fc3(x)
        return x


class SimpleCnn(nn.Module):
    """Simple CNN model with small convolutional backbone and classifier head."""

    def __init__(self, num_classes: int, dropout_rate: float = 0.0):
        super().__init__()
        self.backbone = CnnBackbone()
        self.head = SkinLesionClassifierHead(num_classes, dropout_rate)

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x


class ResNetFromScratch(nn.Module):
    """ResNet model initialized from scratch with a custom classification head."""

    def __init__(self, num_classes: int, layers: int = 50, dropout_rate: float = 0.0):
        super().__init__()
        self.layers = layers
        self.dropout_rate = dropout_rate

        # Create ResNet backbone without pretrained weights
        if layers == 18:
            self.backbone = models.resnet18(weights=None)
        elif layers == 34:
            self.backbone = models.resnet34(weights=None)
        elif layers == 50:
            self.backbone = models.resnet50(weights=None)
        else:
            raise ValueError(f"Unsupported number of layers: {layers}")

        # Remove the final classification layer
        feature_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        self.head = SkinLesionClassifierHead(num_classes, dropout_rate)
        self._feature_dim = feature_dim

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x


class FinetunedResNet(nn.Module):
    """ResNet model with pretrained weights and full fine-tuning."""

    def __init__(self, num_classes: int, layers: int = 50, dropout_rate: float = 0.0):
        super().__init__()
        self.layers = layers
        self.dropout_rate = dropout_rate

        # Create ResNet backbone with pretrained weights
        if layers == 18:
            self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        elif layers == 34:
            self.backbone = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        elif layers == 50:
            self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        else:
            raise ValueError(f"Unsupported number of layers: {layers}")

        # Remove the final classification layer
        feature_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        self.head = SkinLesionClassifierHead(num_classes, dropout_rate)
        self._feature_dim = feature_dim

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x


class FinetunedHeadResNet(nn.Module):
    """ResNet model with a frozen backbone and trainable classification head."""

    def __init__(self, num_classes: int, layers: int = 50, dropout_rate: float = 0.0):
        super().__init__()
        self.layers = layers
        self.dropout_rate = dropout_rate

        # Create ResNet backbone with pretrained weights
        if layers == 18:
            self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        elif layers == 34:
            self.backbone = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        elif layers == 50:
            self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        else:
            raise ValueError(f"Unsupported number of layers: {layers}")

        # Freeze backbone
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Remove the final classification layer
        feature_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        self.head = SkinLesionClassifierHead(num_classes, dropout_rate)
        self._feature_dim = feature_dim

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x


def get_model(
    model_name: str,
    num_classes: int,
    layers: int = 50,
    dropout_rate: float = 0.0
) -> nn.Module:
    """Get model by name."""

    models_dict = {
        "simple_cnn": SimpleCnn,
        "resnet_scratch": ResNetFromScratch,
        "resnet_finetuned": FinetunedResNet,
        "resnet_head": FinetunedHeadResNet,
    }

    model_class = models_dict.get(model_name)
    if model_class is None:
        raise ValueError(f"Unknown model: {model_name}")

    if model_name == "simple_cnn":
        return model_class(num_classes, dropout_rate)
    else:
        return model_class(num_classes, layers, dropout_rate)
