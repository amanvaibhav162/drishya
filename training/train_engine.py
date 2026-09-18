"""
DRISHYA SOTA Training: Resilient & Interruptionless Training Engine
Key Resilience Features:
  - Atomic checkpoint writing (.tmp -> os.replace) prevents corruption on sudden shutdown
  - Traps SIGINT / SIGTERM to dump emergency_checkpoint.pth before process exit
  - Full state serialization: model, optimizer, scaler, scheduler, RNGs (Python, NumPy, PyTorch, CUDA)
  - Seamless auto-resume with --resume
  - OOM defensive catch: catches OutOfMemoryError, clears cache, and resumes without losing run
  - Strict FP16 AMP with GradScaler for Turing Quadro RTX 6000
  - Continuous evaluation of Quadratic Weighted Kappa (QWK), Exact Accuracy, Referral Sensitivity
"""

import os
import sys
import time
import json
import signal
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import cohen_kappa_score, accuracy_score, recall_score
from typing import Optional, Dict, Any, Tuple
from tqdm import tqdm

from losses import (
    DRISHYAMultiTaskLoss,
    coral_logits_to_expected_grade,
    coral_logits_to_class_probs
)


class TrainingEngine:
    """
    Production Training Engine engineered for fault-tolerant multi-task DR training.
    """
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        criterion: DRISHYAMultiTaskLoss,
        output_dir: str,
        device: torch.device,
        num_epochs: int = 30,
        gradient_accumulation_steps: int = 2,
        eval_interval: int = 1,
        early_stopping_patience: int = 8,
        experiment_name: str = "drishya_experiment"
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.output_dir = output_dir
        self.device = device
        self.num_epochs = num_epochs
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.eval_interval = eval_interval
        self.early_stopping_patience = early_stopping_patience
        self.experiment_name = experiment_name

        os.makedirs(self.output_dir, exist_ok=True)
        self.log_file = os.path.join(self.output_dir, "training_metrics.jsonl")

        # Mixed precision GradScaler (FP16 optimized for Quadro RTX 6000 Turing)
        self.scaler = torch.amp.GradScaler("cuda")

        # State trackers
        self.start_epoch = 0
        self.current_epoch = 0
        self.global_step = 0
        self.best_qwk = -1.0
        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.history = []
        self._interrupted = False

        # Register signal handlers for graceful shutdown
        self._register_signal_handlers()

    def _register_signal_handlers(self):
        """Catches SIGINT (Ctrl+C) and SIGTERM (cluster preemption)."""
        def _handle_signal(sig, frame):
            sig_name = "SIGINT" if sig == signal.SIGINT else "SIGTERM"
            print(f"\n[ALERT] Intercepted {sig_name}! Performing atomic emergency checkpoint...")
            self._interrupted = True
            epoch_to_save = getattr(self, "current_epoch", self.start_epoch)
            try:
                self.save_checkpoint(epoch=epoch_to_save, is_emergency=True)
                print("[INFO] Emergency checkpoint saved safely. Exiting.")
            except Exception as e:
                print(f"[WARNING] Could not save emergency checkpoint: {e}")
            sys.exit(0)

        signal.signal(signal.SIGINT, _handle_signal)
        signal.signal(signal.SIGTERM, _handle_signal)

    def _get_rng_states(self) -> Dict[str, Any]:
        """Captures complete RNG states for exact bitwise resumption."""
        states = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            states["cuda"] = torch.cuda.get_rng_state()
        return states

    def _set_rng_states(self, states: Dict[str, Any]):
        """Restores complete RNG states."""
        try:
            if "python" in states:
                random.setstate(states["python"])
            if "numpy" in states:
                np.random.set_state(states["numpy"])
            if "torch" in states:
                torch.set_rng_state(states["torch"])
            if "cuda" in states and torch.cuda.is_available():
                torch.cuda.set_rng_state(states["cuda"])
        except Exception as e:
            print(f"[WARNING] Could not fully restore RNG states: {e}")

    def save_checkpoint(self, epoch: int, is_best: bool = False, is_emergency: bool = False):
        """
        Saves checkpoint atomically via temporary file and atomic os.replace.
        Guarantees that interrupted file writes never corrupt an existing checkpoint.
        """
        state = {
            "epoch": epoch,
            "global_step": self.global_step,
            "experiment_name": self.experiment_name,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "best_qwk": self.best_qwk,
            "best_val_loss": self.best_val_loss,
            "history": self.history,
            "rng_states": self._get_rng_states()
        }

        if is_emergency:
            target_path = os.path.join(self.output_dir, "emergency_checkpoint.pth")
        elif is_best:
            target_path = os.path.join(self.output_dir, "best_model.pth")
        else:
            target_path = os.path.join(self.output_dir, "last_checkpoint.pth")

        tmp_path = target_path + ".tmp"
        torch.save(state, tmp_path)
        os.replace(tmp_path, target_path)

        if is_best:
            # Also keep a copy of the best epoch
            epoch_path = os.path.join(self.output_dir, f"checkpoint_epoch_{epoch}_qwk_{self.best_qwk:.4f}.pth")
            torch.save(state, epoch_path)

    def load_checkpoint(self, checkpoint_path: str) -> int:
        """
        Resumes training state cleanly from a previous checkpoint.
        """
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")

        print(f"[INFO] Resuming training from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scaler_state_dict" in checkpoint and self.scaler:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        if "scheduler_state_dict" in checkpoint and self.scheduler and checkpoint["scheduler_state_dict"]:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        self.start_epoch = checkpoint.get("epoch", 0) + 1
        self.global_step = checkpoint.get("global_step", 0)
        self.best_qwk = checkpoint.get("best_qwk", -1.0)
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        self.history = checkpoint.get("history", [])

        if "rng_states" in checkpoint:
            self._set_rng_states(checkpoint["rng_states"])

        print(f"[INFO] Successfully resumed at Epoch {self.start_epoch}. Previous Best QWK: {self.best_qwk:.4f}")
        return self.start_epoch

    def auto_resume(self) -> bool:
        """Checks for existing checkpoints in output_dir and resumes automatically if found."""
        emergency_ckpt = os.path.join(self.output_dir, "emergency_checkpoint.pth")
        last_ckpt = os.path.join(self.output_dir, "last_checkpoint.pth")

        if os.path.isfile(emergency_ckpt):
            print(f"[INFO] Found emergency checkpoint: {emergency_ckpt}")
            self.load_checkpoint(emergency_ckpt)
            return True
        elif os.path.isfile(last_ckpt):
            print(f"[INFO] Found standard latest checkpoint: {last_ckpt}")
            self.load_checkpoint(last_ckpt)
            return True
        return False

    def train_one_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Executes one epoch of training with mixed precision and gradient accumulation.
        Includes OOM catch and recovery.
        """
        self.model.train()
        self.current_epoch = epoch
        total_loss = 0.0
        step_losses = []

        self.optimizer.zero_grad(set_to_none=True)
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs} [Train]", leave=False)

        for batch_idx, batch in enumerate(pbar):
            try:
                images = batch["image"].to(self.device, non_blocking=True)
                grades = batch["grade"].to(self.device, non_blocking=True)
                masks = batch["mask"].to(self.device, non_blocking=True)
                has_mask = batch["has_mask"].to(self.device, non_blocking=True)
                soft_targets = batch["soft_probs"].to(self.device, non_blocking=True) if "soft_probs" in batch else None

                # Native FP16 AMP context for Turing GPU
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    logits, pred_masks = self.model(images)
                    loss, breakdown = self.criterion(
                        pred_logits=logits,
                        pred_masks=pred_masks,
                        target_grades=grades,
                        target_masks=masks,
                        has_mask=has_mask,
                        soft_targets=soft_targets
                    )
                    # Scale loss for gradient accumulation
                    loss_scaled = loss / self.gradient_accumulation_steps

                self.scaler.scale(loss_scaled).backward()

                if (batch_idx + 1) % self.gradient_accumulation_steps == 0 or (batch_idx + 1) == len(self.train_loader):
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.global_step += 1

                total_loss += loss.item()
                step_losses.append(loss.item())

                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "ord": f"{breakdown['loss_ordinal']:.3f}",
                    "dice": f"{breakdown['loss_dice']:.3f}"
                })

            except torch.cuda.OutOfMemoryError as oom:
                print(f"\n[WARNING] CUDA Out Of Memory caught at batch {batch_idx}! Flushing cache and continuing...")
                torch.cuda.empty_cache()
                self.optimizer.zero_grad(set_to_none=True)
                continue

            except Exception as e:
                print(f"\n[WARNING] Batch {batch_idx} error: {e}. Skipping batch.")
                continue

        avg_loss = total_loss / max(1, len(step_losses))
        return {"train_loss": avg_loss}

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """
        Evaluates model on validation set.
        Computes QWK, Exact Accuracy, Referral Sensitivity, Specificity, and Lesion Dice.
        """
        self.model.eval()
        total_val_loss = 0.0
        val_steps = 0

        all_preds = []
        all_expected_grades = []
        all_targets = []
        dice_scores = {0: [], 1: [], 2: [], 3: []}  # MA, EX, HE, SE

        for batch in tqdm(self.val_loader, desc="Evaluating", leave=False):
            images = batch["image"].to(self.device, non_blocking=True)
            grades = batch["grade"].to(self.device, non_blocking=True)
            masks = batch["mask"].to(self.device, non_blocking=True)
            has_mask = batch["has_mask"].to(self.device, non_blocking=True)

            with torch.amp.autocast("cuda", dtype=torch.float16):
                logits, pred_masks = self.model(images)
                loss, _ = self.criterion(
                    pred_logits=logits,
                    pred_masks=pred_masks,
                    target_grades=grades,
                    target_masks=masks,
                    has_mask=has_mask
                )

            total_val_loss += loss.item()
            val_steps += 1

            # Predictions from CORAL logits
            if self.model.use_coral:
                expected_grade = coral_logits_to_expected_grade(logits)
                class_probs = coral_logits_to_class_probs(logits)
                preds = torch.argmax(class_probs, dim=-1)
            else:
                probs = torch.softmax(logits, dim=-1)
                preds = torch.argmax(probs, dim=-1)
                weights = torch.arange(5, device=logits.device).float()
                expected_grade = (probs * weights).sum(dim=-1)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_expected_grades.extend(expected_grade.cpu().numpy().tolist())
            all_targets.extend(grades.cpu().numpy().tolist())

            # Evaluate Dice on valid masks
            mask_probs = (torch.sigmoid(pred_masks) > 0.5).float()
            for ch in range(4):
                ch_pred = mask_probs[:, ch]
                ch_target = masks[:, ch]
                inter = (ch_pred * ch_target).sum(dim=(-2, -1))
                card = ch_pred.sum(dim=(-2, -1)) + ch_target.sum(dim=(-2, -1))
                dice = (2.0 * inter + 1e-5) / (card + 1e-5)
                # Filter by has_mask
                valid_dice = dice[has_mask > 0.5].cpu().numpy().tolist()
                dice_scores[ch].extend(valid_dice)

        y_true = np.array(all_targets)
        y_pred = np.array(all_preds)

        # 1. Quadratic Weighted Kappa
        qwk = cohen_kappa_score(y_true, y_pred, weights="quadratic")

        # 2. Exact 5-Class Accuracy
        acc = accuracy_score(y_true, y_pred)

        # 3. Clinical Referral (Grade >= 2) Sensitivity & Specificity
        ref_true = (y_true >= 2).astype(int)
        ref_pred = (y_pred >= 2).astype(int)

        ref_sens = recall_score(ref_true, ref_pred, zero_division=0)
        # Specificity = recall of class 0 in binary referral
        neg_mask = (ref_true == 0)
        ref_spec = (ref_pred[neg_mask] == 0).mean() if neg_mask.sum() > 0 else 0.0

        metrics = {
            "val_loss": total_val_loss / max(1, val_steps),
            "qwk": float(qwk),
            "accuracy": float(acc),
            "referral_sensitivity": float(ref_sens),
            "referral_specificity": float(ref_spec),
            "dice_ma": float(np.mean(dice_scores[0])) if len(dice_scores[0]) > 0 else 0.0,
            "dice_ex": float(np.mean(dice_scores[1])) if len(dice_scores[1]) > 0 else 0.0,
            "dice_he": float(np.mean(dice_scores[2])) if len(dice_scores[2]) > 0 else 0.0,
            "dice_se": float(np.mean(dice_scores[3])) if len(dice_scores[3]) > 0 else 0.0,
        }
        return metrics

    def run(self):
        """
        Executes the complete multi-epoch training loop with early stopping,
        metric tracking, and resilient atomic checkpointing.
        """
        print(f"\n=======================================================")
        print(f"🚀 Launching Resilient Training: {self.experiment_name}")
        print(f"Output Directory: {self.output_dir}")
        print(f"Total Epochs: {self.num_epochs} (Starting at {self.start_epoch})")
        print(f"Gradient Accumulation: {self.gradient_accumulation_steps}")
        print(f"Device: {self.device}")
        print(f"=======================================================\n")

        for epoch in range(self.start_epoch, self.num_epochs):
            t0 = time.time()

            # 1. Train epoch
            train_metrics = self.train_one_epoch(epoch)

            # 2. Evaluate
            val_metrics = {}
            if (epoch + 1) % self.eval_interval == 0 or (epoch + 1) == self.num_epochs:
                val_metrics = self.evaluate()

            # 3. Learning rate step
            if self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics.get("val_loss", train_metrics["train_loss"]))
                else:
                    self.scheduler.step()

            duration = time.time() - t0
            current_lr = self.optimizer.param_groups[0]["lr"]

            epoch_record = {
                "epoch": epoch,
                "duration_sec": round(duration, 2),
                "learning_rate": current_lr,
                **train_metrics,
                **val_metrics
            }
            self.history.append(epoch_record)

            # Append structured log to JSONL
            with open(self.log_file, "a") as f:
                f.write(json.dumps(epoch_record) + "\n")

            # Console Summary
            print(
                f"[Epoch {epoch+1:02d}/{self.num_epochs:02d} - {duration:.1f}s] "
                f"Train Loss: {train_metrics['train_loss']:.4f} | "
                f"Val Loss: {val_metrics.get('val_loss', 0.0):.4f} | "
                f"QWK: {val_metrics.get('qwk', 0.0):.4f} | "
                f"Acc: {val_metrics.get('accuracy', 0.0)*100:.1f}% | "
                f"Ref Sens: {val_metrics.get('referral_sensitivity', 0.0)*100:.1f}% | "
                f"LR: {current_lr:.2e}"
            )

            # Check for best model improvement
            current_qwk = val_metrics.get("qwk", -1.0)
            is_best = current_qwk > self.best_qwk

            if is_best:
                print(f"  ⭐ New Best QWK: {current_qwk:.4f} (Previous: {self.best_qwk:.4f}). Saving best checkpoint.")
                self.best_qwk = current_qwk
                self.patience_counter = 0
                self.save_checkpoint(epoch=epoch, is_best=True)
            else:
                self.patience_counter += 1

            # Always save latest checkpoint atomically
            self.save_checkpoint(epoch=epoch, is_best=False)

            # Early stopping check
            if self.patience_counter >= self.early_stopping_patience:
                print(f"\n[INFO] Early stopping triggered after {self.early_stopping_patience} epochs without QWK improvement.")
                break

        print(f"\n=======================================================")
        print(f"🎉 Training Finished! Best QWK Achieved: {self.best_qwk:.4f}")
        print(f"Checkpoints and logs saved to: {self.output_dir}")
        print(f"=======================================================\n")
