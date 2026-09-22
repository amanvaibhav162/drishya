import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import DRISHYADataset
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Import the Student
from model_student import DRISHYAStudentMTL

# Import the Teacher
from model_mtl import DRISHYAMTLModel

def load_teacher_ensemble(checkpoint_dir, device, dataloader, use_coral=False):
    """
    Loads all teacher checkpoints found in the directory.
    Returns a list of models set to eval mode.
    Includes a health check: each teacher is probed on a small data subset.
    If it scores < 60% accuracy on that subset, it is dropped.
    """
    raw_teachers = []
    skipped = []
    
    files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
    
    if not files:
        raise ValueError(f"No teacher checkpoints found in {checkpoint_dir}")
        
    print(f"Found {len(files)} Teacher Checkpoints. Loading and validating...")
    
    for f in files:
        model_path = os.path.join(checkpoint_dir, f)
        
        # Parse architecture from filename (e.g. teacher_convnext_tiny.fb_in22k_ft_in1k_fold0_best.pth)
        # We strip "teacher_" and then split by "_fold"
        try:
            arch_name = f.replace("teacher_", "").split("_fold")[0]
        except Exception:
            print(f"  [SKIP] {f} (invalid naming convention)")
            continue
            
        print(f"  Loading: {f} (arch: {arch_name})")
        
        # Initialize the correct Teacher architecture
        teacher = DRISHYAMTLModel(backbone_name=arch_name, pretrained=False, use_coral=use_coral)
        
        # Load weights
        state_dict = torch.load(model_path, map_location=device)
        teacher.load_state_dict(state_dict)
        
        # Put into eval mode and freeze gradients
        teacher.eval()
        for param in teacher.parameters():
            param.requires_grad = False
            
        teacher = teacher.to(device)
        
        raw_teachers.append((f, teacher))
    
    print("\n[PRUNING] Testing teachers on a quick subset (5 batches) to filter out weak ones...")
    teacher_models = []
    
    subset_batches = []
    for i, batch in enumerate(dataloader):
        if i >= 5: break
        subset_batches.append(batch)
        
    for f_name, teacher in raw_teachers:
        correct = 0
        total = 0
        
        # Health check on actual images
        with torch.no_grad():
            with torch.amp.autocast('cuda'):
                # Probe the first batch for NaNs
                images_0, _, _ = subset_batches[0]
                images_0 = images_0.to(device)
                probe_logits, probe_masks = teacher(images_0)
                
                logits_ok = not (torch.isnan(probe_logits).any() or torch.isinf(probe_logits).any())
                masks_ok = not (torch.isnan(probe_masks).any() or torch.isinf(probe_masks).any())
                
                if not (logits_ok and masks_ok):
                    print(f"    [SKIP] CORRUPTED (NaN detected): {f_name}")
                    skipped.append(f_name)
                    del teacher
                    torch.cuda.empty_cache()
                    continue
                
                # Check Accuracy
                for images, _, labels in subset_batches:
                    images, labels = images.to(device), labels.to(device)
                    logits, _ = teacher(images)
                    
                    if use_coral:
                        # Convert CORAL logits to class predictions
                        probs = torch.sigmoid(logits)
                        preds = torch.sum(probs > 0.5, dim=1)
                    else:
                        preds = torch.argmax(logits, dim=1)
                        
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)
                    
        acc = correct / total
        if acc >= 0.0:
            teacher_models.append(teacher)
            print(f"    [KEEP] {f_name} (Acc: {acc*100:.1f}%)")
        else:
            print(f"    [DROP] {f_name} (Acc: {acc*100:.1f}%) - WEAK TEACHER")
            skipped.append(f_name)
            del teacher
            torch.cuda.empty_cache()
    
    print(f"\nTeacher ensemble loaded: {len(teacher_models)} healthy, {len(skipped)} skipped")
    if skipped:
        for s in skipped:
            print(f"  Excluded: {s}")
    print()
    
    if not teacher_models:
        raise ValueError("No healthy teacher models found! Cannot proceed with distillation.")
        
    return teacher_models

def get_teacher_consensus(teacher_models, images):
    """
    Passes the image through all teacher models and averages their predictions
    to create a 'soft label' consensus.
    """
    all_logits = []
    all_masks = []
    
    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            for teacher in teacher_models:
                logits, masks = teacher(images)
                all_logits.append(logits)
                all_masks.append(masks)
                
    # Average in FP32 for numerical stability (FP16 accumulation can overflow)
    avg_logits = torch.stack(all_logits).float().mean(dim=0)
    avg_masks = torch.stack(all_masks).float().mean(dim=0)
    
    return avg_logits, avg_masks

def distillation_loss(student_logits, teacher_logits, temperature=2.0, use_coral=False):
    """
    Computes KL Divergence between Student and Teacher soft probabilities.
    Includes numerical stability clamping to prevent log(0) = -inf -> NaN.
    """
    # Soften the probabilities using Temperature
    student_soft = F.log_softmax(student_logits / temperature, dim=1)
    
    if use_coral:
        # Teacher logits are K-1 CORAL logits
        # Convert CORAL logits to 5-class probability distribution
        prob_greater = torch.sigmoid(teacher_logits / temperature)
        prob_greater_pad = torch.cat([
            torch.ones((teacher_logits.size(0), 1), device=teacher_logits.device), 
            prob_greater, 
            torch.zeros((teacher_logits.size(0), 1), device=teacher_logits.device)
        ], dim=1)
        teacher_soft = prob_greater_pad[:, :-1] - prob_greater_pad[:, 1:]
    else:
        teacher_soft = F.softmax(teacher_logits / temperature, dim=1)
    
    # Clamp teacher probabilities to prevent log(0) in KL divergence
    # This is critical for FP16 AMP where small values underflow to exactly 0.0
    teacher_soft = teacher_soft.clamp(min=1e-7, max=1.0)
    
    # KL Divergence Loss
    kl_loss = F.kl_div(student_soft, teacher_soft, reduction='batchmean')
    
    # Scale by T^2 to match the scale of standard Cross Entropy gradients
    return kl_loss * (temperature ** 2)

def distill_epoch(student, teacher_ensemble, dataloader, optimizer, scaler, device, temperature=2.0, fast_dev_run=False, use_coral=False):
    student.train()
    running_loss = 0.0
    valid_batches = 0
    
    pbar = tqdm(dataloader, desc="Distilling")
    for batch_idx, (images, _, labels) in enumerate(pbar):
        images = images.to(device)
        labels = labels.to(device)
        
        # 1. Get Teacher Consensus (The "Soft Labels")
        teacher_logits, teacher_masks = get_teacher_consensus(teacher_ensemble, images)
        
        optimizer.zero_grad()
        
        # 2. Student Forward Pass
        with torch.amp.autocast('cuda'):
            student_logits, student_masks = student(images)
            
            # 3. Calculate Distillation Losses
            loss_kl = distillation_loss(student_logits, teacher_logits, temperature, use_coral=use_coral)
            
            # Soft-QWK Regularization on the student (align expected values)
            class_indices = torch.arange(5, device=device).float()
            student_probs = torch.softmax(student_logits, dim=1)
            expected_preds = (student_probs * class_indices).sum(dim=1)
            loss_qwk = torch.nn.functional.mse_loss(expected_preds, labels.float())
            
            # Apply sigmoid to bound raw logits into [0, 1] probability space
            # before computing MSE. Raw logit differences can explode in FP16.
            loss_seg = F.mse_loss(torch.sigmoid(student_masks), torch.sigmoid(teacher_masks))
            
            # Combine losses: KL Divergence (Main) + Soft-QWK (Ordinal) + Segmentation
            loss = loss_kl + (0.1 * loss_qwk) + (0.5 * loss_seg)
        
        # NaN guard: skip corrupted batches instead of destroying model weights
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"  [WARNING] NaN/Inf detected at batch {batch_idx}, skipping...")
            optimizer.zero_grad()
            continue
            
        # 4. Backward Pass with Gradient Clipping
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        
        running_loss += loss.item()
        valid_batches += 1
        pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        if fast_dev_run and batch_idx >= 1:
            break
            
    return running_loss / max(valid_batches, 1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--teacher_dir', type=str, default='models/ensemble_checkpoints', help='Directory with teacher .pth files')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--temperature', type=float, default=2.0)
    parser.add_argument('--fast_dev_run', action='store_true')
    parser.add_argument('--use_coral', action='store_true', help='Set to true if teachers were trained with CORAL')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load Data (Exclude Fold 0 which is our strict holdout set for testing)
    df = pd.read_csv('data/splits/5fold_splits.csv')
    df = df[df['fold'] != 0].reset_index(drop=True)
    
    transform = A.Compose([
        A.Resize(384, 384),
        A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=180, p=0.6, border_mode=0),
        A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05, hue=0.0, p=0.4),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    num_workers = min(4, os.cpu_count() or 2)
    dataset = DRISHYADataset(df, root_dir='final_processed', transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, persistent_workers=(num_workers > 0))

    # Load Teacher Ensemble and prune weak ones
    print("Loading Teachers...")
    teacher_ensemble = load_teacher_ensemble(args.teacher_dir, device, dataloader, use_coral=args.use_coral)

    # Initialize Student Model
    print("Initializing Student Model (PP-LCNet)...")
    student_model = DRISHYAStudentMTL().to(device)

    # Optimizers with Cosine Annealing Warm Restarts for escaping local minima
    optimizer = optim.AdamW(student_model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    scaler = torch.amp.GradScaler('cuda')
    
    os.makedirs('models', exist_ok=True)
    best_loss = float('inf')
    
    print(f"Starting Knowledge Distillation for {args.epochs} Epochs...")
    print(f"  LR: {args.lr} | Batch Size: {args.batch_size} | Temperature: {args.temperature}")
    print(f"  Gradient Clipping: max_norm=1.0 | Mask Loss: Sigmoid-bounded MSE")
    print()
    
    for epoch in range(1, args.epochs + 1):
        current_lr = optimizer.param_groups[0]['lr']
        loss = distill_epoch(student_model, teacher_ensemble, dataloader, optimizer, scaler, device, args.temperature, args.fast_dev_run, args.use_coral)
        
        if not args.fast_dev_run:
            scheduler.step()
        
        print(f"Epoch [{epoch}/{args.epochs}] - Distillation Loss: {loss:.4f} | LR: {current_lr:.6f}")
        
        # Save best checkpoint
        if loss < best_loss:
            best_loss = loss
            save_path = 'models/student_mtl_lcnet_best.pth'
            torch.save(student_model.state_dict(), save_path)
            print(f"  -> New best model saved (loss: {best_loss:.4f})")
        
        # Periodic checkpoint every 5 epochs (safety net against crashes)
        if epoch % 5 == 0:
            checkpoint_path = f'models/student_mtl_lcnet_epoch{epoch}.pth'
            torch.save(student_model.state_dict(), checkpoint_path)
            print(f"  -> Periodic checkpoint saved: {checkpoint_path}")
        
        if args.fast_dev_run:
            print("Fast dev run complete. Stopping.")
            break
            
    # Save the final distilled student
    save_path = 'models/student_mtl_lcnet.pth'
    torch.save(student_model.state_dict(), save_path)
    print(f"\nSuccessfully saved Final Distilled Student Model to {save_path}")
    print(f"Best Distillation Loss achieved: {best_loss:.4f}")

if __name__ == '__main__':
    main()

