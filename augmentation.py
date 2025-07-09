import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from typing import Callable


def rich_augmentor(image: np.ndarray, rng_seed: int = None) -> np.ndarray:
    """
    Applies random flips, brightness, contrast, hue changes, and rotation.
    """
    if rng_seed is not None:
        np.random.seed(rng_seed)

    # Ensure image is in uint8 format for albumentations
    if image.dtype == np.float32 or image.dtype == np.float64:
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)

    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomBrightnessContrast(
            brightness_limit=0.1,
            contrast_limit=0.1,
            p=0.5
        ),
        A.Rotate(limit=10, p=0.5),
        A.GaussNoise(p=0.3),
    ])

    augmented = transform(image=image)
    result = augmented["image"].astype(np.float32) / 255.0

    return result


def no_augmentation(image: np.ndarray) -> np.ndarray:
    """No augmentation - returns image as is."""
    if image.dtype != np.float32:
        return image.astype(np.float32) / 255.0 if image.max() > 1 else image.astype(np.float32)
    return image


def get_augmentor(augment_name: str) -> Callable:
    """Get augmentor by name."""
    augmentors = {
        "rich": rich_augmentor,
        "none": no_augmentation,
    }
    return augmentors.get(augment_name, no_augmentation)
