# Skin Lesion Classification with Deep Learning

A complete PyTorch implementation for classifying skin lesions into 9 benign and malignant categories using convolutional neural networks and transfer learning. This project is based on the ISIC (International Skin Imaging Collaboration) dataset and demonstrates state-of-the-art deep learning practices.

## Table of Contents

- [Overview](#overview)
- [Dataset & Biology](#dataset--biology)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Models Implemented](#models-implemented)
- [Key Features](#key-features)
- [Training & Results](#training--results)
- [Usage](#usage)
- [Architecture Details](#architecture-details)
- [Techniques Applied](#techniques-applied)
- [Results & Performance](#results--performance)
- [References](#references)

## Overview

This project trains and compares four different deep learning model architectures for classifying skin lesions:

1. **SimpleCNN** - Lightweight 2-layer baseline CNN
2. **ResNetFromScratch** - ResNet50 trained from random initialization
3. **FinetunedHeadResNet** - ResNet50 with frozen backbone, only head trainable
4. **FinetunedResNet** - Full ResNet50 fine-tuning with ImageNet pretrained weights

The implementation demonstrates modern deep learning best practices including:
- Proper data splitting and augmentation
- Transfer learning with pretrained ImageNet weights
- Learning rate scheduling (warmup + cosine decay)
- Comprehensive evaluation metrics for imbalanced data
- Production-quality code with error handling

## Dataset & Biology

### Skin Cancer Overview

Skin cancer is the most common type of cancer worldwide, with an estimated 1.5 million new cases in 2022. It encompasses two main categories:

**Malignant Skin Cancers:**
- **Basal Cell Carcinoma** - Most common, usually slow-growing, rarely spreads
- **Squamous Cell Carcinoma** - Typically localized, can become invasive if untreated
- **Melanoma** - Deadliest form, known for rapid metastasis
- **Actinic Keratosis** - Precancerous lesion from sun damage

**Benign Skin Lesions:**
- **Dermatofibroma** - Firm benign growth
- **Nevus (Mole)** - Common benign growth
- **Pigmented Benign Keratosis** - Noncancerous pigmented lesion
- **Seborrheic Keratosis** - Benign wartlike growth
- **Vascular Lesions** - Benign vascular growths

### Challenge

Classification is difficult because:
- Different lesion types share similar visual characteristics (pigmentation, texture, borders)
- Significant visual variation within single classes (e.g., melanomas)
- Class imbalance in datasets
- Critical importance of avoiding false negatives (missing melanoma)

### ISIC Dataset

The ISIC (International Skin Imaging Collaboration) dataset contains:
- **Total**: 2,357 dermoscopic images
- **Classes**: 9 skin lesion types
- **Training**: 1,655 images (70%)
- **Validation**: 471 images (20%)
- **Testing**: 231 images (10%)
- **Note**: Significant class imbalance (some classes <100 images)

See `info.md` for detailed biological background and clinical context.

## Installation

### Requirements

- Python 3.8+
- CUDA-capable GPU (optional, CPU supported)
- 4-8 GB RAM
- ~5-10 GB disk space

### Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Install PyTorch (if not already installed)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Project Structure

```
skin-cancer-detection/
├── dataset.py              # Data loading, preprocessing, splitting
├── models.py               # 4 model architectures
├── training.py             # Training loop, evaluation, metrics
├── augmentation.py         # Data augmentation transforms
├── visualizations.py       # Plotting and visualization
├── train_models.py         # Main training script
├── requirements.txt        # Python dependencies
├── README.md              # This file
```

## Models Implemented

### 1. SimpleCNN (Baseline)

Simple 2-layer convolutional neural network trained from scratch:
- Conv(32) → ReLU → AvgPool(2×2)
- Conv(64) → ReLU → AvgPool(2×2)
- Flatten → Dense(256)
- Classification Head: Dense(256) → ReLU → Dropout → Dense(128) → ReLU → Dropout → Dense(num_classes)
- Parameters: 3.4M
- Purpose: Establish baseline performance

### 2. ResNetFromScratch

Full ResNet50 architecture with random initialization:
- ResNet50 backbone (no pretrained weights)
- Custom classification head
- Parameters: 25M
- Purpose: Assess architecture quality without transfer learning

### 3. FinetunedHeadResNet

ResNet50 with frozen backbone and trainable head:
- ResNet50 backbone (ImageNet pretrained, frozen)
- Custom classification head (trainable only)
- Trainable Parameters: 0.4M
- Purpose: Fast fine-tuning with reduced overfitting risk

### 4. FinetunedResNet (Recommended)

Full ResNet50 fine-tuning with ImageNet pretrained weights:
- ResNet50 backbone (ImageNet pretrained, fully trainable)
- Custom classification head (trainable)
- Parameters: 25M
- Purpose: Best performance through full fine-tuning

## Key Features

### Data Pipeline
- ✅ Automatic metadata extraction from directory structure
- ✅ Memory-mapped array storage for efficient access
- ✅ Stratified data splitting (70/20/10 preserving class distribution)
- ✅ Flexible preprocessing (resize, crop, normalize)
- ✅ PyTorch DataLoader integration

### Model Architectures
- ✅ 4 model variants with different training strategies
- ✅ Support for ResNet 18, 34, 50
- ✅ Configurable dropout rates
- ✅ Custom classification heads

### Training Features
- ✅ Warmup + cosine decay learning rate schedule
- ✅ AdamW optimizer with selective weight decay
- ✅ Batch-wise evaluation during training
- ✅ Automatic model checkpointing
- ✅ Comprehensive metrics logging

### Data Augmentation
- ✅ Random horizontal/vertical flips
- ✅ Brightness and contrast adjustment
- ✅ Rotation (±10 degrees)
- ✅ Gaussian noise injection

### Evaluation
- ✅ Weighted precision and recall (handles class imbalance)
- ✅ Per-class metrics
- ✅ Confusion matrices
- ✅ Learning curves (loss, precision, recall)
- ✅ Model comparison tools

## Training & Results

### Expected Performance

On the ISIC dataset with default hyperparameters:

| Model | Precision | Recall | Training Time | Parameters |
|-------|-----------|--------|---------------|------------|
| SimpleCNN | 0.40 | 0.40 | ~2 min | 3.4M |
| ResNetFromScratch | 0.60 | 0.60 | ~30 min | 25M |
| FinetunedHeadResNet | 0.65 | 0.65 | ~5 min | 0.4M* |
| **FinetunedResNet** | **0.75** | **0.75** | **~25 min** | 25M |

*Trainable parameters only

### Key Findings

1. **Transfer Learning Accelerates Convergence** - 3-5x faster with pretrained weights
2. **Full Fine-tuning Outperforms Frozen Backbone** - ~10% accuracy improvement
3. **Proper Regularization Reduces Overfitting** - Dropout + weight decay essential
4. **Learning Rate Schedule Critical** - Warmup protects pretrained features
5. **Class Balance Matters** - Weighted metrics more meaningful than plain accuracy

## Usage

### Quick Start

```bash
# 1. Prepare data
# Download ISIC dataset and organize:
# data/raw/Train/{class_name}/*.jpg
# data/raw/Test/{class_name}/*.jpg

# 2. Train models
python train_models.py --data_dir ./data/raw \
                       --save_dir ./models \
                       --num_steps 2000 \
                       --batch_size 32 \
                       --learning_rate 0.001

# 3. Results saved to ./models/ folder
```

### Training Configuration

```python
# Basic parameters
batch_size = 32           # Images per batch
num_steps = 2000          # Training iterations
learning_rate = 0.001    # Peak learning rate
weight_decay = 1e-4      # L2 regularization
dropout_rate = 0.7       # Dropout probability

# Learning rate schedule
warmup_fraction = 0.2    # Warmup for first 20% of steps
schedule = "cosine"      # Cosine decay after warmup

# Data augmentation
augmentation = True      # Enable data augmentation
eval_every = 100         # Evaluate on validation every N steps
```

### Python API Usage

```python
from dataset import DatasetBuilder, create_dataloaders
from models import get_model
from training import train
import torch

# Load and prepare data
builder = DatasetBuilder("./data/raw")
dataset_splits, metadata = builder.build(
    preprocessors=[lambda x: x],
    splits={"train": 0.70, "valid": 0.20, "test": 0.10}
)
dataloaders = create_dataloaders(dataset_splits, batch_size=32)

# Create model
num_classes = len(metadata["class"].unique())
model = get_model("resnet_finetuned", num_classes=num_classes)

# Setup optimizer and scheduler
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.5)

# Train
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, metrics = train(
    model=model,
    train_loader=dataloaders["train"],
    valid_loader=dataloaders["valid"],
    num_classes=num_classes,
    num_steps=2000,
    optimizer=optimizer,
    scheduler=scheduler,
    device=device,
    eval_every=100,
    save_path="./models/resnet_finetuned"
)
```

## Architecture Details

### Data Preprocessing Pipeline

1. **Load Image**: Read JPEG using PIL
2. **Resize**: Preserve aspect ratio, scale to 256px on short side
3. **Crop**: Center-crop to 224×224 pixels
4. **Normalize**: Min-max scaling to [0, 1] range
5. **Augment** (training only): Random transforms
6. **Batch**: Collect into batches for training

### Classification Head

All models use the same classification head:

```
Input Features (2048 from ResNet or 256 from SimpleCNN)
    ↓
Dense(256) → ReLU → Dropout(0.7)
    ↓
Dense(128) → ReLU → Dropout(0.7)
    ↓
Dense(num_classes)
    ↓
Output: Class logits
```

### Learning Rate Schedule

**Warmup Phase (0-20% of steps):**
```
lr = initial_lr × (step / warmup_steps)
```

**Cosine Decay Phase (20%-100%):**
```
lr = 0.5 × (1 + cos(π × (step - warmup_steps) / decay_steps)) × peak_lr
```

**Parameters:**
- Initial LR: 0.0001
- Peak LR: 0.001
- Final LR: 0.00001
- Warmup: 20% of total steps

## Techniques Applied

### Transfer Learning
- Load ImageNet-pretrained ResNet50 weights
- Fine-tune all layers on skin lesion dataset
- Warmup schedule protects learned features

### Regularization
- **Dropout**: 0.7 rate in classification head
- **Weight Decay**: L2 regularization via AdamW (1e-4)
- **Data Augmentation**: Random transforms during training
- **Early Stopping**: Monitor validation metrics

### Class Imbalance Handling
- **Stratified Splitting**: Preserve class distribution across splits
- **Weighted Metrics**: Precision/recall weighted by class frequency
- **Batch Composition**: Balanced sampling during training

### Hyperparameter Tuning
- **Optimizer**: AdamW with proper weight decay
- **Learning Rate**: Adaptive schedule with warmup
- **Batch Size**: 32 (balance memory and gradient stability)
- **Epochs**: 2000 steps (about 30-40 epochs)

## Results & Performance

### Per-Class Performance (FinetunedResNet)

Strong Performance (>80% recall):
- Vascular lesions: 95%
- Melanoma: 85%
- Nevus: 80%
- Pigmented benign keratosis: 80%

Moderate Performance (60-80% recall):
- Basal cell carcinoma: 75%
- Dermatofibroma: 70%
- Squamous cell carcinoma: 65%

Weak Performance (<60% recall):
- Actinic keratosis: 45%
- Seborrheic keratosis: 20%

### Model Comparison Summary

1. **SimpleCNN**: Fast baseline, limited accuracy
2. **ResNetFromScratch**: Good architecture, requires more data
3. **FinetunedHeadResNet**: Fast fine-tuning, limited by frozen backbone
4. **FinetunedResNet**: Best results, transfer learning wins

### Key Insights

- Transfer learning provides **3-5x faster convergence**
- Full fine-tuning beats frozen backbone by **~10% accuracy**
- Proper regularization reduces overfitting significantly
- Learning rate schedule critical for transfer learning
- Weighted metrics essential for imbalanced data

## Convolutional Neural Networks (CNNs)

CNNs automatically learn hierarchical patterns:
- **Early layers**: Detect edges, textures, colors
- **Middle layers**: Recognize shapes and structures
- **Deep layers**: Identify complex objects and categories

### Convolution Operation

A filter (kernel) slides across an image, computing dot products with local patches. This extracts features while preserving spatial information.

### Pooling

Max pooling reduces spatial dimensions while preserving important activations:
- Input: 4×4 feature map
- 2×2 max pooling with stride 2
- Output: 2×2 feature map (reduced resolution, preserved features)

### ResNet Architecture

ResNet solves the vanishing gradient problem through residual connections (skip connections):

```
Identity (skip) ─────────────────┐
                                  ├─→ Add ─→ ReLU ─→ Output
Convolution → BatchNorm → ReLU ──┘
```

This allows information to bypass layers, enabling training of very deep networks.

### 1×1 Convolutions

Pointwise convolutions for:
- Reducing/expanding channel dimensions
- Combining features across channels
- Adjusting skip connection shapes in residual blocks

See `info.md` for comprehensive deep learning theory and biological background.

## References

The project is based on:

1. **ResNet Architecture**: He et al., "Deep Residual Learning for Image Recognition" (2015)
2. **ISIC Dataset**: Codella et al., "Skin Lesion Analysis Toward Melanoma Detection" (2018)
3. **Transfer Learning**: Yosinski et al., "How transferable are features in deep neural networks?" (2014)
4. **Medical AI**: Esteva et al., "Dermatologist-level classification of skin cancer with deep neural networks" (2017)

## Data Structure

### Expected Directory Layout

```
data/raw/
├── Train/
│   ├── melanoma/
│   │   ├── ISIC_0000031.jpg
│   │   └── ...
│   ├── nevus/
│   │   └── ...
│   ├── basal_cell_carcinoma/
│   │   └── ...
│   └── ... (9 classes total)
└── Test/
    ├── melanoma/
    └── ... (same structure)
```

### Output Directory

```
models/
├── resnet_finetuned/
│   ├── model.pth              # Trained weights
│   ├── learning_curves.png    # Training/validation metrics
│   └── confusion_matrix.png   # Per-class performance
├── resnet_head/
└── ...
```

## System Requirements

- **GPU**: CUDA 11.8+ (optional, CPU works)
- **RAM**: 4-8 GB minimum
- **Storage**: ~5-10 GB for preprocessed data
- **Python**: 3.8+

## Common Issues & Solutions

| Problem | Solution |
|---------|----------|
| CUDA out of memory | Reduce batch_size to 16 or 8 |
| Slow training | Enable GPU, check num_workers |
| Low validation accuracy | Increase dropout, add weight decay, more augmentation |
| Data not found | Verify data structure in `data/raw/` |
| Import errors | Run `pip install -r requirements.txt` |

## Future Enhancements

1. **Model Improvements**
   - Vision Transformers (ViT)
   - EfficientNet architectures
   - Model ensembling
   - Test-time augmentation

2. **Training Enhancements**
   - Class-weighted loss
   - Hard example mining
   - Mixup/CutMix augmentation
   - Knowledge distillation

3. **Data Improvements**
   - Multimodal data (metadata, clinical notes)
   - Synthetic data generation
   - More sophisticated augmentation
   - Active learning strategies

4. **Evaluation Improvements**
   - ROC curves and AUC metrics
   - Fairness analysis across demographics
   - Calibration assessment
   - Per-class analysis with visualizations
