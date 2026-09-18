"""
DRISHYA SOTA Training: Teacher Model Trainer CLI
Usage:
  python training/train_teacher.py --teacher_id 1 --splits_csv /path/to/splits.csv --fold 0
"""

import os
import argparse
import pandas as pd
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from models import build_teacher_model
from dataset import create_dataloaders
from losses import DRISHYAMultiTaskLoss
from train_engine import TrainingEngine


def parse_args():
    parser = argparse.ArgumentParser(description="Train Orthogonal Teacher Network on 24 GB GPU")
    parser.add_argument("--teacher_id", type=int, required=True, choices=[1, 2, 3],
                        help="1: ConvNeXtV2-Base, 2: Swin-Base-384, 3: EfficientNet-B5")
    parser.add_argument("--splits_csv", type=str, default="data/splits/5fold_splits.csv",
                        help="Path to patient-stratified 5-fold split CSV")
    parser.add_argument("--data_dir", type=str, default="",
                        help="Base directory containing preprocessed images")
    parser.add_argument("--fold", type=int, default=0,
                        help="Validation fold (0 to 4)")
    parser.add_argument("--epochs", type=int, default=30,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Physical batch size (calibrated to 8 for 24 GB RTX 6000)")
    parser.add_argument("--accum_steps", type=int, default=2,
                        help="Gradient accumulation steps (effective batch size = batch_size * accum_steps = 16)")
    parser.add_argument("--lr", type=float, default=2e-4,
                        help="Initial peak learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-2,
                        help="Weight decay for AdamW")
    parser.add_argument("--workers", type=int, default=8,
                        help="DataLoader worker processes")
    parser.add_argument("--output_dir", type=str, default="",
                        help="Checkpoint and logging directory")
    parser.add_argument("--resume", action="store_true",
                        help="Resume automatically from existing checkpoint if available")
    return parser.parse_args()


def main():
    args = parse_args()

    teacher_names = {
        1: "convnextv2_base",
        2: "swin_base",
        3: "efficientnet_b5"
    }
    model_name = teacher_names[args.teacher_id]

    if not args.output_dir:
        args.output_dir = f"models/teachers/{model_name}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[INFO] Starting Teacher {args.teacher_id} ({model_name}) on device: {device}")

    # 1. Load Split Data
    if not os.path.exists(args.splits_csv):
        raise FileNotFoundError(f"Splits CSV not found at {args.splits_csv}. Please provide a valid split file.")

    df = pd.read_csv(args.splits_csv)
    train_df = df[df["fold"] != args.fold].reset_index(drop=True)
    val_df = df[df["fold"] == args.fold].reset_index(drop=True)

    print(f"[INFO] Loaded Dataset: {len(train_df)} train samples, {len(val_df)} val samples (Holdout Fold {args.fold})")

    # 2. Build DataLoaders
    train_loader, val_loader = create_dataloaders(
        train_df=train_df,
        val_df=val_df,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.workers,
        img_size=512,
        is_preprocessed=True
    )

    # 3. Build Model
    model = build_teacher_model(teacher_id=args.teacher_id, pretrained=True)
    model = model.to(device)

    # 4. Setup Optimizer & Cosine Scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=1e-6)

    # 5. Loss Function
    criterion = DRISHYAMultiTaskLoss(
        num_classes=5,
        lambda_ordinal=1.0,
        lambda_smooth_l1=0.25,
        lambda_dice=0.50,
        lambda_tversky=0.50
    )

    # 6. Initialize Resilient Engine
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
        experiment_name=f"Teacher_{args.teacher_id}_{model_name}"
    )

    # Auto-resume if requested or checkpoint exists
    if args.resume:
        engine.auto_resume()

    # Run training
    engine.run()


if __name__ == "__main__":
    main()
