import os
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Callable, Optional, Dict, Any, Tuple
import pickle


class SkinLesionDataset(Dataset):
    """Dataset for skin lesion classification."""

    def __init__(
        self,
        metadata: pd.DataFrame,
        images_memmap: np.memmap,
        preprocessor: Callable,
        augmentor: Optional[Callable] = None,
        rng_seed: int = 42
    ):
        self.metadata = metadata
        self.images_memmap = images_memmap
        self.preprocessor = preprocessor
        self.augmentor = augmentor
        self.rng = np.random.RandomState(rng_seed)

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        image = self.images_memmap[idx]
        label = self.metadata.iloc[idx]["label"]
        frame_id = self.metadata.iloc[idx]["frame_id"]

        if self.augmentor is not None:
            image = self.augmentor(image)

        image = torch.from_numpy(image).float()
        return {
            "image": image,
            "label": torch.tensor(label, dtype=torch.long),
            "frame_id": frame_id
        }


class MetadataLoader:
    """Loads and manages dataset metadata."""

    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)

    def load(self, class_map: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """Load metadata from directory structure."""
        metadata = []

        for path in self.data_dir.rglob("*.jpg"):
            try:
                split, class_name, filename = path.parts[-3:]

                # Map class names
                if class_map is not None:
                    class_label = class_map.get(class_name, class_name)
                else:
                    class_label = class_name

                metadata.append({
                    "split_orig": split,
                    "class_orig": class_name,
                    "class": class_label,
                    "full_path": str(path),
                    "filename": filename
                })
            except (ValueError, IndexError):
                continue

        df = pd.DataFrame(metadata)

        # Create label mapping
        if len(df) > 0:
            unique_classes = df["class"].unique()
            class_to_label = {cls: idx for idx, cls in enumerate(sorted(unique_classes))}
            df["label"] = df["class"].map(class_to_label)
            df["class_to_label"] = pd.Series([class_to_label] * len(df))

        return df.reset_index(drop=True).rename_axis("frame_id").reset_index()


class ImageLoader:
    """Loads and preprocesses images into memory-mapped arrays."""

    def __init__(self, metadata: pd.DataFrame):
        self.metadata = metadata

    def load(
        self,
        preprocessors: list,
        image_size: Tuple[int, int, int] = (224, 224, 3)
    ) -> Dict[str, np.memmap]:
        """Load and preprocess images into memmaps."""

        num_samples = len(self.metadata)
        h, w, c = image_size

        # Load raw images into memory
        raw_images = np.zeros((num_samples, h, w, c), dtype=np.float32)

        for idx, row in self.metadata.iterrows():
            try:
                img = Image.open(row["full_path"]).convert("RGB")
                img_array = np.array(img, dtype=np.float32)

                # Basic resize and crop
                img_array = self._resize_preserve_aspect(img_array, short_side=256)
                img_array = self._center_crop(img_array, size=224)
                img_array = img_array / 255.0

                raw_images[idx] = img_array
            except Exception as e:
                print(f"Error loading image at index {idx}: {e}")
                raw_images[idx] = np.zeros((h, w, c), dtype=np.float32)

        loaded = {}
        for preprocessor in preprocessors:
            loaded[preprocessor.__name__] = raw_images.copy()

        return loaded

    @staticmethod
    def _resize_preserve_aspect(image: np.ndarray, short_side: int = 256) -> np.ndarray:
        """Resize image preserving aspect ratio."""
        h, w = image.shape[:2]
        scale = short_side / min(h, w)
        new_h, new_w = int(h * scale), int(w * scale)

        img_pil = Image.fromarray((image * 255).astype(np.uint8) if image.max() <= 1 else image.astype(np.uint8))
        img_pil = img_pil.resize((new_w, new_h), Image.BILINEAR)
        return np.array(img_pil, dtype=np.float32) / 255.0

    @staticmethod
    def _center_crop(image: np.ndarray, size: int = 224) -> np.ndarray:
        """Center crop image to square."""
        h, w = image.shape[:2]
        top = (h - size) // 2
        left = (w - size) // 2
        return image[top:top+size, left:left+size]


class DatasetBuilder:
    """Builds dataset splits."""

    def __init__(self, data_dir: str):
        self.data_dir = data_dir

    def build(
        self,
        preprocessors: list,
        image_size: Tuple[int, int, int] = (224, 224, 3),
        splits: Dict[str, float] = None,
        class_map: Optional[Dict[str, Any]] = None,
        rng_seed: int = 42
    ) -> Dict[str, Tuple[SkinLesionDataset, pd.DataFrame]]:
        """Build dataset splits."""

        if splits is None:
            splits = {"train": 0.70, "valid": 0.20, "test": 0.10}

        # Load metadata
        metadata = MetadataLoader(self.data_dir).load(class_map)

        # Load images
        images_dict = ImageLoader(metadata).load(preprocessors, image_size)

        # Assign splits
        np.random.seed(rng_seed)
        num_samples = len(metadata)
        split_names = list(splits.keys())
        split_probs = list(splits.values())

        metadata["split"] = np.random.choice(
            split_names,
            p=split_probs,
            size=num_samples
        )

        # Create datasets
        dataset_splits = {}
        for split_name in split_names:
            split_mask = metadata["split"] == split_name
            split_metadata = metadata[split_mask].reset_index(drop=True)
            split_indices = np.where(split_mask)[0]

            # Create memmap for this split
            split_images = images_dict[preprocessors[0].__name__][split_indices]

            dataset_splits[split_name] = (
                SkinLesionDataset(
                    metadata=split_metadata,
                    images_memmap=split_images,
                    preprocessor=preprocessors[0],
                    augmentor=None
                ),
                split_metadata
            )

        return dataset_splits, metadata


def create_dataloaders(
    dataset_splits: Dict[str, Tuple[SkinLesionDataset, pd.DataFrame]],
    batch_size: int = 32,
    num_workers: int = 0
) -> Dict[str, DataLoader]:
    """Create dataloaders from datasets."""

    dataloaders = {}
    for split_name, (dataset, _) in dataset_splits.items():
        shuffle = split_name == "train"
        dataloaders[split_name] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers
        )

    return dataloaders
