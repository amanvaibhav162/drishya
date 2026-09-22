import os
import argparse
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Import custom modules
from dataset import DRISHYADataset
from model_mtl import DRISHYAMTLModel

def coral_loss(logits, labels, num_classes=5):
    """
    Computes Rank Consistent Ordinal Regression (CORAL) Loss.
    logits: (B, num_classes - 1)
    labels: (B,) containing integer ranks 0 to num_classes - 1
    """
    # Create binary labels: 1 if label > k, else 0
    levels = torch.arange(num_classes - 1, device=logits.device).unsqueeze(0).expand(labels.size(0), -1)
    labels_ext = labels.unsqueeze(1).expand(-1, num_classes - 1)
    binary_labels = (labels_ext > levels).float()
    
    return torch.nn.functional.binary_cross_entropy_with_logits(logits, binary_labels)

def get_coral_probs(logits):
    """ Converts CORAL K-1 logits into K probabilities. """
    probs = torch.sigmoid(logits)
    # prob of rank > k
    # prob of rank k = prob(rank > k-1) - prob(rank > k)
    # prob of rank 0 = 1 - prob(rank > 0)
    # prob of rank K-1 = prob(rank > K-2)
    B = probs.size(0)
    K = probs.size(1) + 1
    out_probs = torch.zeros(B, K, device=logits.device)
    out_probs[:, 0] = 1.0 - probs[:, 0]
    for k in range(1, K - 1):
        out_probs[:, k] = probs[:, k-1] - probs[:, k]
    out_probs[:, K-1] = probs[:, K-2]
    
    # Ensure no negative probabilities due to numerical precision
    return torch.clamp(out_probs, min=0.0, max=1.0)

def get_transforms():
    train_transform = A.Compose([
        A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=180, p=0.6, border_mode=0),
        A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05, hue=0.0, p=0.4),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    val_transform = A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    return train_transform, val_transform

def train_epoch(model, loader, optimizer, scaler, clf_criterion, seg_criterion, device, fast_dev_run=False):
    model.train()
    running_loss = 0.0
    valid_batches = 0
    
    pbar = tqdm(loader, desc="Training")
    for batch_idx, (images, masks, labels) in enumerate(pbar):
        images = images.to(device)
        masks = masks.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        with torch.amp.autocast('cuda'):
            logits, pred_masks = model(images)
            
            # Multi-Task Loss Calculation
            if getattr(model, 'use_coral', False):
                loss_clf = coral_loss(logits, labels, num_classes=5)
                probs = get_coral_probs(logits)
            else:
                # Label Smoothing 0.1 for FocalLoss equivalent (via CrossEntropy)
                # Note: SMP FocalLoss doesn't natively support label smoothing argument in some versions,
                # but we can use CrossEntropy with label_smoothing=0.1
                loss_clf = torch.nn.functional.cross_entropy(logits, labels, label_smoothing=0.1)
                probs = torch.softmax(logits, dim=1)
            
            # Ordinal / QWK Regularization
            class_indices = torch.arange(5, device=device).float()
            expected_preds = (probs * class_indices).sum(dim=1)
            loss_qwk = torch.nn.functional.mse_loss(expected_preds, labels.float())
            
            loss_seg = seg_criterion(pred_masks, masks)
            # lambda_qwk defaults to 0.1, alpha defaults to 0.5 for segmentation
            loss = loss_clf + (0.1 * loss_qwk) + (0.5 * loss_seg)
            
        # NaN guard: skip corrupted batches instead of destroying model weights
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"  [WARNING] NaN/Inf detected at batch {batch_idx}, skipping...")
            optimizer.zero_grad()
            continue
            
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        
        running_loss += loss.item()
        valid_batches += 1
        pbar.set_postfix({'loss': loss.item()})
        
        if fast_dev_run and batch_idx >= 1:
            break
            
    return running_loss / max(valid_batches, 1)

def val_epoch(model, loader, clf_criterion, seg_criterion, device, fast_dev_run=False):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc="Validating")
    with torch.no_grad():
        for batch_idx, (images, masks, labels) in enumerate(pbar):
            images = images.to(device)
            masks = masks.to(device)
            labels = labels.to(device)
            
            with torch.amp.autocast('cuda'):
                logits, pred_masks = model(images)
                
                if getattr(model, 'use_coral', False):
                    loss_clf = coral_loss(logits, labels, num_classes=5)
                    probs = get_coral_probs(logits)
                else:
                    loss_clf = clf_criterion(logits, labels)
                    probs = torch.softmax(logits, dim=1)
                    
                loss_seg = seg_criterion(pred_masks, masks)
                loss = loss_clf + (0.5 * loss_seg)
                
            running_loss += loss.item()
            
            # Calculate accuracy
            _, predicted = torch.max(probs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if fast_dev_run and batch_idx >= 1:
                break
                
    acc = 100 * correct / total
    return running_loss / min(len(loader), 2 if fast_dev_run else len(loader)), acc

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1. Load Split Blueprint
    df = pd.read_csv(args.data_csv)
    train_df = df[df['fold'] != args.fold].reset_index(drop=True)
    val_df = df[df['fold'] == args.fold].reset_index(drop=True)
    
    train_tfm, val_tfm = get_transforms()
    
    train_dataset = DRISHYADataset(train_df, root_dir='final_processed', transform=train_tfm)
    val_dataset = DRISHYADataset(val_df, root_dir='final_processed', transform=val_tfm)
    
    num_workers = min(4, os.cpu_count() or 2)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, persistent_workers=(num_workers > 0))
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, persistent_workers=(num_workers > 0))
    
    # 2. Initialize Architecture
    model = DRISHYAMTLModel(backbone_name=args.model_name, pretrained=True, use_coral=args.use_coral)
    model.to(device)
    
    # 3. Optimizers & Loss
    clf_criterion = smp.losses.FocalLoss(mode='multiclass')
    seg_criterion = smp.losses.DiceLoss(mode='multilabel')
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    scaler = torch.amp.GradScaler('cuda')
    
    best_acc = 0.0
    
    print(f"Starting Training for Fold {args.fold}...")
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        
        train_loss = train_epoch(model, train_loader, optimizer, scaler, clf_criterion, seg_criterion, device, args.fast_dev_run)
        val_loss, val_acc = val_epoch(model, val_loader, clf_criterion, seg_criterion, device, args.fast_dev_run)
        
        if not args.fast_dev_run:
            scheduler.step()
        
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
        # Save Checkpoints
        latest_path = os.path.join(args.output_dir, f"teacher_{args.model_name}_fold{args.fold}_latest.pth")
        torch.save(model.state_dict(), latest_path)
        
        if val_acc > best_acc or args.fast_dev_run:
            best_acc = val_acc
            save_path = os.path.join(args.output_dir, f"teacher_{args.model_name}_fold{args.fold}_best.pth")
            torch.save(model.state_dict(), save_path)
            print(f"Saved new best model to {save_path} (Val Acc: {val_acc:.2f}%)")
            
        if args.fast_dev_run:
            print("Fast dev run complete. Stopping.")
            break

    if not args.fast_dev_run:
        done_marker = os.path.join(args.output_dir, f"teacher_{args.model_name}_fold{args.fold}_done.txt")
        with open(done_marker, "w") as f:
            f.write(f"Completed {args.epochs} epochs\n")
        print(f"Training fully completed. Wrote completion marker to {done_marker}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_csv', type=str, default='data/splits/5fold_splits.csv')
    parser.add_argument('--model_name', type=str, default='tu-tf_efficientnetv2_s.in21k_ft_in1k', help='Options: tu-tf_efficientnetv2_s.in21k_ft_in1k, tu-convnextv2_tiny.fcmae_ft_in22k_in1k, tu-tiny_vit_21m_224.dist_in22k_ft_in1k, tu-vit_small_patch16_224.dino_v3')
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=8, help='Reduced to 8 to prevent OOM on 8GB GPUs')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--output_dir', type=str, default='models/ensemble_checkpoints/')
    parser.add_argument('--use_coral', action='store_true', help='Use CORAL ordinal loss')
    parser.add_argument('--fast_dev_run', action='store_true', help='Run 2 batches only to test code')
    args = parser.parse_args()
    
    main(args)
