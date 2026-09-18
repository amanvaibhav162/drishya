"""
DRISHYA SOTA Training: Dataset & Augmentation Pipeline
Supports:
  - 512x512 preprocessed fundus images (3-channel composite or RGB)
  - 4-channel dense lesion masks (MA, EX, HE, SE)
  - Partial lesion supervision (handles images with and without ground truth masks)
  - Robust exception handling to prevent worker crashes on corrupted images
  - Patient-stratified 5-fold cross-validation
"""

import os
import cv2
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Callable, Dict, Any, List, Tuple
import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transforms(img_size: int = 512) -> A.Compose:
    """
    Albumentations training augmentation pipeline preserving micro-lesion geometry.
    """
    return A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Affine(
            scale=(0.9, 1.1),
            translate_percent=(-0.0625, 0.0625),
            rotate=(-45, 45),
            mode=cv2.BORDER_CONSTANT,
            cval=0,
            cval_mask=0,
            p=0.7
        ),
        A.OneOf([
            A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0.05, p=1.0),
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=1.0),
        ], p=0.5),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0
        ),
        ToTensorV2()
    ], additional_targets={'mask': 'mask'})


def get_valid_transforms(img_size: int = 512) -> A.Compose:
    """
    Deterministic validation transform: Resize and ImageNet normalization.
    """
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0
        ),
        ToTensorV2()
    ], additional_targets={'mask': 'mask'})


def generate_composite_3ch(img_bgr: np.ndarray) -> np.ndarray:
    """
    Constructs the 3-channel synthetic composite input tensor:
      Channel 0: Green CLAHE (Contrast-Limited Adaptive Histogram Equalization)
      Channel 1: Ben Graham Local Gaussian Normalization
      Channel 2: Morphological Black-Hat Filter isolating microaneurysms
    """
    green = img_bgr[:, :, 1]

    # Channel 0: CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    ch0 = clahe.apply(green)

    # Channel 1: Ben Graham Normalization
    blurred = cv2.GaussianBlur(green, (0, 0), sigmaX=30)
    ch1 = cv2.addWeighted(green, 4, blurred, -4, 128)

    # Channel 2: Black-Hat Filter
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    ch2 = cv2.morphologyEx(green, cv2.MORPH_BLACKHAT, kernel)

    composite = np.stack([ch0, ch1, ch2], axis=-1)
    return composite


class DRISHYADataset(Dataset):
    """
    Robust Multi-Task Dataset for Diabetic Retinopathy.
    Safely loads images and 4-channel lesion masks with fail-safe fallbacks.
    """
    def __init__(
        self,
        df: pd.DataFrame,
        data_dir: str = "",
        transform: Optional[Callable] = None,
        is_preprocessed: bool = True,
        img_size: int = 512,
        on_the_fly_composite: bool = False
    ):
        """
        Args:
            df: DataFrame containing at least ['image_path', 'grade']
                Optional columns: ['mask_path', 'patient_id', 'has_mask']
            data_dir: Base directory prepended to relative paths
            transform: Albumentations transform pipeline
            is_preprocessed: True if images already contain the 3-ch composite
            img_size: Target image size (512x512)
            on_the_fly_composite: If True, computes [Green-CLAHE, Ben-Graham, Black-Hat] on the fly
        """
        self.df = df.reset_index(drop=True)
        self.data_dir = data_dir
        self.transform = transform
        self.is_preprocessed = is_preprocessed
        self.img_size = img_size
        self.on_the_fly_composite = on_the_fly_composite

    def __len__(self) -> int:
        return len(self.df)

    def _resolve_path(self, path: str) -> str:
        if not path:
            return ""
        if os.path.isabs(path):
            return path
        return os.path.join(self.data_dir, path)

    def _load_image(self, path: str) -> np.ndarray:
        resolved = self._resolve_path(path)
        if resolved.endswith('.npy'):
            img = np.load(resolved)
        else:
            img = cv2.imread(resolved)
            if img is None:
                raise FileNotFoundError(f"Failed to read image at {resolved}")
            if self.on_the_fly_composite or not self.is_preprocessed:
                img = generate_composite_3ch(img)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def _load_mask(self, mask_path: str) -> Tuple[np.ndarray, float]:
        """
        Loads a 4-channel lesion mask (Channel 0: MA, 1: EX, 2: HE, 3: Vessels/SE).
        Returns mask array [H, W, 4] and has_mask flag (1.0 or 0.0).
        """
        if not mask_path or pd.isna(mask_path):
            dummy_mask = np.zeros((self.img_size, self.img_size, 4), dtype=np.float32)
            return dummy_mask, 0.0

        resolved = self._resolve_path(str(mask_path))

        # Check if individual lesion component PNGs exist in the mask directory
        base_prefix = None
        for suffix in ['_composite.png', '_composite.jpg', '_MA.png', '_EX.png', '_HE.png', '_vessels.png', '_SE.png']:
            if resolved.endswith(suffix):
                base_prefix = resolved[:-len(suffix)]
                break

        if base_prefix is not None:
            ma_path = f"{base_prefix}_MA.png"
            ex_path = f"{base_prefix}_EX.png"
            he_path = f"{base_prefix}_HE.png"
            ve_path = f"{base_prefix}_vessels.png"
            se_path = f"{base_prefix}_SE.png"
            ch3_path = ve_path if os.path.exists(ve_path) else se_path

            if os.path.exists(ma_path) or os.path.exists(ex_path) or os.path.exists(he_path) or os.path.exists(ch3_path):
                h, w = self.img_size, self.img_size
                ma = cv2.imread(ma_path, cv2.IMREAD_GRAYSCALE) if os.path.exists(ma_path) else np.zeros((h, w), dtype=np.uint8)
                ex = cv2.imread(ex_path, cv2.IMREAD_GRAYSCALE) if os.path.exists(ex_path) else np.zeros((h, w), dtype=np.uint8)
                he = cv2.imread(he_path, cv2.IMREAD_GRAYSCALE) if os.path.exists(he_path) else np.zeros((h, w), dtype=np.uint8)
                ch3 = cv2.imread(ch3_path, cv2.IMREAD_GRAYSCALE) if os.path.exists(ch3_path) else np.zeros((h, w), dtype=np.uint8)

                ma = ma if ma is not None else np.zeros((h, w), dtype=np.uint8)
                ex = ex if ex is not None else np.zeros((h, w), dtype=np.uint8)
                he = he if he is not None else np.zeros((h, w), dtype=np.uint8)
                ch3 = ch3 if ch3 is not None else np.zeros((h, w), dtype=np.uint8)

                if ma.shape != (h, w):
                    ma = cv2.resize(ma, (w, h), interpolation=cv2.INTER_NEAREST)
                if ex.shape != (h, w):
                    ex = cv2.resize(ex, (w, h), interpolation=cv2.INTER_NEAREST)
                if he.shape != (h, w):
                    he = cv2.resize(he, (w, h), interpolation=cv2.INTER_NEAREST)
                if ch3.shape != (h, w):
                    ch3 = cv2.resize(ch3, (w, h), interpolation=cv2.INTER_NEAREST)

                mask = np.stack([ma > 127, ex > 127, he > 127, ch3 > 127], axis=-1).astype(np.float32)
                return mask, 1.0

        if not os.path.exists(resolved):
            dummy_mask = np.zeros((self.img_size, self.img_size, 4), dtype=np.float32)
            return dummy_mask, 0.0

        if resolved.endswith('.npz'):
            try:
                with np.load(resolved) as npz:
                    ma = npz['microaneurysms'] if 'microaneurysms' in npz else np.zeros((self.img_size, self.img_size), dtype=bool)
                    ex = npz['exudates'] if 'exudates' in npz else np.zeros((self.img_size, self.img_size), dtype=bool)
                    he = npz['hemorrhages'] if 'hemorrhages' in npz else np.zeros((self.img_size, self.img_size), dtype=bool)
                    ve = npz['vessels'] if 'vessels' in npz else np.zeros((self.img_size, self.img_size), dtype=bool)
                    mask = np.stack([ma, ex, he, ve], axis=-1).astype(np.float32)
                    return (mask > 0.5).astype(np.float32), 1.0
            except Exception:
                dummy_mask = np.zeros((self.img_size, self.img_size, 4), dtype=np.float32)
                return dummy_mask, 0.0
        elif resolved.endswith('.npy'):
            mask = np.load(resolved).astype(np.float32)
            if mask.ndim == 2:
                mask = np.stack([mask] * 4, axis=-1)
            elif mask.shape[0] == 4 and mask.ndim == 3:
                mask = np.transpose(mask, (1, 2, 0))
            return (mask > 0.5).astype(np.float32), 1.0
        else:
            m = cv2.imread(resolved, cv2.IMREAD_UNCHANGED)
            if m is None:
                dummy_mask = np.zeros((self.img_size, self.img_size, 4), dtype=np.float32)
                return dummy_mask, 0.0
            if m.ndim == 2:
                mask = np.stack([(m > 127).astype(np.float32)] * 4, axis=-1)
            elif m.shape[2] >= 4:
                mask = (m[:, :, :4] > 127).astype(np.float32)
            elif m.shape[2] == 3:
                ch4 = np.zeros((m.shape[0], m.shape[1], 1), dtype=np.float32)
                mask = np.concatenate([(m > 127).astype(np.float32), ch4], axis=-1)
            else:
                dummy_mask = np.zeros((self.img_size, self.img_size, 4), dtype=np.float32)
                return dummy_mask, 0.0

            if mask.shape[:2] != (self.img_size, self.img_size):
                mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)

            return (mask > 0.5).astype(np.float32), 1.0

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Fault-tolerant item retrieval. If an image is corrupt, falls back to idx 0.
        """
        try:
            row = self.df.iloc[idx]
            image_path = str(row['image_path'])
            grade = int(row['grade'])
            mask_path = str(row.get('mask_path', ''))

            image = self._load_image(image_path)
            mask, has_mask = self._load_mask(mask_path)

            if self.transform is not None:
                augmented = self.transform(image=image, mask=mask)
                image = augmented['image']
                mask = augmented['mask']
            else:
                image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
                mask = torch.from_numpy(mask.transpose(2, 0, 1)).float()

            if isinstance(mask, np.ndarray):
                if mask.ndim == 2:
                    mask = np.stack([mask] * 4, axis=-1)
                mask = torch.from_numpy(mask.transpose(2, 0, 1)).float()
            elif isinstance(mask, torch.Tensor):
                if mask.ndim == 2:
                    mask = mask.unsqueeze(0).repeat(4, 1, 1).float()
                elif mask.ndim == 3 and mask.shape[-1] == 4:
                    mask = mask.permute(2, 0, 1).float()

            return {
                "image": image,                          # [3, 512, 512]
                "grade": torch.tensor(grade, dtype=torch.long),  # scalar
                "mask": mask,                            # [4, 512, 512]
                "has_mask": torch.tensor(has_mask, dtype=torch.float32),
                "image_path": image_path,
                "global_idx": int(row.get("global_idx", idx))
            }

        except Exception as e:
            # Defensive recovery: fallback to index 0 to avoid crashing training workers
            print(f"[WARNING] Error reading sample index {idx}: {e}. Falling back to index 0.")
            return self.__getitem__(0)


def create_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    data_dir: str = "",
    batch_size: int = 8,
    num_workers: int = 8,
    img_size: int = 512,
    is_preprocessed: bool = True,
    on_the_fly_composite: bool = False
) -> Tuple[DataLoader, DataLoader]:
    """
    Creates high-throughput PyTorch DataLoaders configured for Quadro RTX 6000.
    Uses pin_memory=True and persistent_workers=True for zero CPU-GPU transfer stalls.
    """
    train_dataset = DRISHYADataset(
        df=train_df,
        data_dir=data_dir,
        transform=get_train_transforms(img_size=img_size),
        is_preprocessed=is_preprocessed,
        img_size=img_size,
        on_the_fly_composite=on_the_fly_composite
    )

    val_dataset = DRISHYADataset(
        df=val_df,
        data_dir=data_dir,
        transform=get_valid_transforms(img_size=img_size),
        is_preprocessed=is_preprocessed,
        img_size=img_size,
        on_the_fly_composite=on_the_fly_composite
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=(num_workers > 0)
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=(num_workers > 0)
    )

    return train_loader, val_loader
