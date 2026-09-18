"""
DRISHYA SOTA Training: Student Distillation CLI
Distills the 620M 3-teacher ensemble into the EfficientNetV2-S U-Net student model.
Target: 22.09M parameters, 42.1 MB FP16 ONNX, <12 ms inference.
"""

import os
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from models import build_student_model
from dataset import create_dataloaders, DRISHYADataset, get_train_transforms, get_valid_transforms
from losses import DRISHYAMultiTaskLoss, DistillationLoss
from train_engine import TrainingEngine


class DistillationDataset(Dataset):
    """
    Wraps DRISHYADataset to dynamically yield soft probability distributions
    from cached HDF5 pseudo-labels.
    """
    def __init__(self, base_dataset: DRISHYADataset, pseudo_h5_path: str):
        self.base_dataset = base_dataset
        self.pseudo_h5_path = pseudo_h5_path
        self._h5 = None

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        item = self.base_dataset[idx]
        global_idx = item.get("global_idx", idx)
        if self._h5 is None and os.path.exists(self.pseudo_h5_path):
            import h5py
            # SWMR (Single Writer Multiple Reader) mode for thread-safe worker reading
            self._h5 = h5py.File(self.pseudo_h5_path, 'r', swmr=True)

        if self._h5 is not None and "probs" in self._h5 and global_idx < len(self._h5["probs"]):
            soft_probs = torch.from_numpy(self._h5["probs"][global_idx]).float()
            item["soft_probs"] = soft_probs

        return item


def parse_args():
    parser = argparse.ArgumentParser(description="Distill EfficientNetV2-S Student Model")
    parser.add_argument("--splits_csv", type=str, default="data/splits/5fold_splits.csv")
    parser.add_argument("--data_dir", type=str, default="")
    parser.add_argument("--pseudo_h5", type=str, default="cache/pseudo_labels.h5",
                        help="Path to cached 3-teacher consensus pseudo-labels (HDF5)")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--accum_steps", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Distillation loss weighting: (1-alpha)*hard + alpha*soft")
    parser.add_argument("--temperature", type=float, default=2.0,
                        help="Softmax distillation temperature")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output_dir", type=str, default="models/student")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\n[INFO] Initializing EfficientNetV2-S Student Model on device: {device}")

    # 1. Load Data
    if not os.path.exists(args.splits_csv):
        raise FileNotFoundError(f"Splits CSV not found at {args.splits_csv}")

    df = pd.read_csv(args.splits_csv)
    df["global_idx"] = np.arange(len(df))
    train_df = df[df["fold"] != args.fold].reset_index(drop=True)
    val_df = df[df["fold"] == args.fold].reset_index(drop=True)

    train_base = DRISHYADataset(
        df=train_df,
        data_dir=args.data_dir,
        transform=get_train_transforms(img_size=512),
        is_preprocessed=True,
        img_size=512
    )

    if os.path.exists(args.pseudo_h5):
        print(f"[INFO] Attaching soft pseudo-label cache: {args.pseudo_h5}")
        train_dataset = DistillationDataset(train_base, args.pseudo_h5)
    else:
        print(f"[WARNING] Pseudo-label cache {args.pseudo_h5} not found. Training with hard labels only.")
        train_dataset = train_base

    val_dataset = DRISHYADataset(
        df=val_df,
        data_dir=args.data_dir,
        transform=get_valid_transforms(img_size=512),
        is_preprocessed=True,
        img_size=512
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=(args.workers > 0)
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=(args.workers > 0)
    )

    # 2. Build Student Model
    model = build_student_model(pretrained=True).to(device)

    # 3. Setup Optimizer & Scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=1e-6)

    # 4. Distillation Criterion
    base_criterion = DRISHYAMultiTaskLoss(
        num_classes=5,
        lambda_ordinal=1.0,
        lambda_smooth_l1=0.25,
        lambda_dice=0.50,
        lambda_tversky=0.50
    )

    criterion = DistillationLoss(
        mtl_criterion=base_criterion,
        alpha=args.alpha,
        temperature=args.temperature
    )

    # 5. Training Engine with fault-tolerant checkpointing
    engine = TrainingEngine(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        output_dir=args.output_dir,
        device=device,
        num_epochs=args.epochs,
        gradient_accumulation_steps=args.accum_steps,
        experiment_name="Student_EfficientNetV2_S_Distillation"
    )

    if args.resume:
        engine.auto_resume()

    engine.run()


if __name__ == "__main__":
    main()
