import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

class DRISHYADataset(Dataset):
    """
    Custom Dataset for Multi-Task Learning DR Pipeline.
    Loads the primary fundus image alongside its 4 lesion segmentation masks.
    Applies identical spatial augmentations to the image and all masks.
    """
    def __init__(self, df: pd.DataFrame, root_dir: str, transform=None):
        """
        Args:
            df (pd.DataFrame): DataFrame containing at least 'id_code', 'label', and 'dataset_name' 
                               (e.g., 'aptos', 'eyepacs').
            root_dir (str): Path to the final_processed directory.
            transform (albumentations.Compose): Data augmentations.
        """
        self.df = df
        self.root_dir = root_dir
        self.transform = transform
        
        # Default transforms if none provided (basic normalization and tensor conversion)
        if self.transform is None:
            self.transform = A.Compose([
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        id_code = str(row['id_code'])
        dataset_name = str(row['dataset_name']).lower()
        
        # Some CSVs include the extension (e.g., '1034_right.png'). Strip it.
        base_id = id_code.split('.')[0]
        
        # The structure is: final_processed/dataset_name/images/base_id/
        img_folder = os.path.join(self.root_dir, dataset_name, 'images', base_id)
        
        # Paths for the main image and 4 masks
        img_path_jpg = os.path.join(img_folder, f"{base_id}.jpg")
        img_path_png = os.path.join(img_folder, f"{base_id}.png")
        
        if os.path.exists(img_path_jpg):
            img_path = img_path_jpg
        else:
            img_path = img_path_png
            
        ma_path = os.path.join(img_folder, f"{base_id}_ma.png")
        ex_path = os.path.join(img_folder, f"{base_id}_ex.png")
        hemo_path = os.path.join(img_folder, f"{base_id}_hemorrhage.png")
        vessel_path = os.path.join(img_folder, f"{base_id}_vessel.png")

        # Load Main Image
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Image not found at {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        H, W, _ = image.shape
        
        # Load Masks (Grayscale) with safe zero-fill fallbacks
        mask_ma = cv2.imread(ma_path, cv2.IMREAD_GRAYSCALE)
        if mask_ma is None: mask_ma = np.zeros((H, W), dtype=np.uint8)
        
        mask_ex = cv2.imread(ex_path, cv2.IMREAD_GRAYSCALE)
        if mask_ex is None: mask_ex = np.zeros((H, W), dtype=np.uint8)
        
        mask_hemo = cv2.imread(hemo_path, cv2.IMREAD_GRAYSCALE)
        if mask_hemo is None: mask_hemo = np.zeros((H, W), dtype=np.uint8)
        
        mask_vessel = cv2.imread(vessel_path, cv2.IMREAD_GRAYSCALE)
        if mask_vessel is None: mask_vessel = np.zeros((H, W), dtype=np.uint8)
        
        # Stack masks into a (H, W, 4) numpy array
        # Albumentations expects masks as a multi-channel image when applying spatial transforms
        masks_stacked = np.stack([mask_ma, mask_ex, mask_hemo, mask_vessel], axis=-1)
        
        # Binarize masks just in case they have anti-aliasing (0 or 1)
        masks_stacked = (masks_stacked > 127).astype(np.float32)

        # Apply Augmentations to BOTH image and stacked masks simultaneously
        augmented = self.transform(image=image, mask=masks_stacked)
        
        image_tensor = augmented['image']
        # Albumentations returns mask as (H, W, 4). PyTorch expects (4, H, W)
        masks_tensor = augmented['mask'].permute(2, 0, 1) 
        
        # Label (ICDR Grade 0-4)
        label = int(row['label'])
        
        return image_tensor, masks_tensor, label

if __name__ == "__main__":
    print("Testing DRISHYADataset logic (dry run)...")
    
    # Create a mock dataframe pointing to the file we discovered: s00r00000 in aptos
    mock_df = pd.DataFrame({
        'id_code': ['s00r00000'],
        'label': [0],
        'dataset_name': ['aptos']
    })
    
    root_path = "final_processed"
    
    # Simple transform pipeline
    transform = A.Compose([
        A.Resize(384, 384),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    try:
        dataset = DRISHYADataset(df=mock_df, root_dir=root_path, transform=transform)
        img_t, masks_t, label = dataset[0]
        
        print("Data loaded successfully!")
        print(f"Image tensor shape: {img_t.shape} (Expected: 3, 384, 384)")
        print(f"Masks tensor shape: {masks_t.shape} (Expected: 4, 384, 384)")
        print(f"Label: {label}")
    except Exception as e:
        print(f"Failed to load dataset: {e}")
