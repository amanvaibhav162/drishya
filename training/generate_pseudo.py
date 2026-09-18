"""
DRISHYA SOTA Training: Offline Ensemble Pseudo-Label Generation
Generates soft consensus probability distributions and lesion masks from the
3 orthogonal teachers and caches them to HDF5.
Supports interruptionless resumption (checks existing keys in HDF5 before processing).
"""

import os
import argparse
import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
import cv2
from tqdm import tqdm

from models import build_teacher_model


class InferenceDataset(Dataset):
    """Simple inference dataset for batching images during pseudo-labeling."""
    def __init__(self, df: pd.DataFrame, data_dir: str = "", img_size: int = 512):
        self.df = df.reset_index(drop=True)
        self.data_dir = data_dir
        self.img_size = img_size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = str(row["image_path"])
        full_path = image_path if os.path.isabs(image_path) else os.path.join(self.data_dir, image_path)

        if full_path.endswith('.npy'):
            img = np.load(full_path)
        else:
            img = cv2.imread(full_path)
            if img is None:
                img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if img.shape[:2] != (self.img_size, self.img_size):
            img = cv2.resize(img, (self.img_size, self.img_size))

        # Normalize with ImageNet stats
        tensor = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        tensor = (tensor - mean) / std

        return {"image": tensor, "path": image_path, "idx": idx}


def parse_args():
    parser = argparse.ArgumentParser(description="Generate 3-Teacher Consensus Pseudo-Labels")
    parser.add_argument("--data_csv", type=str, required=True,
                        help="CSV containing all images to generate pseudo-labels for")
    parser.add_argument("--data_dir", type=str, default="")
    parser.add_argument("--t1_ckpt", type=str, default="models/teachers/convnextv2_base/best_model.pth")
    parser.add_argument("--t2_ckpt", type=str, default="models/teachers/swin_base/best_model.pth")
    parser.add_argument("--t3_ckpt", type=str, default="models/teachers/efficientnet_b5/best_model.pth")
    parser.add_argument("--output_h5", type=str, default="cache/pseudo_labels.h5")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def load_teacher(teacher_id: int, ckpt_path: str, device: torch.device):
    print(f"[INFO] Loading Teacher {teacher_id} checkpoint: {ckpt_path}")
    model = build_teacher_model(teacher_id=teacher_id, pretrained=False)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state.get("model_state_dict", state))
    model = model.to(device).eval()
    return model


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(os.path.dirname(args.output_h5) or ".", exist_ok=True)

    df = pd.read_csv(args.data_csv)
    dataset = InferenceDataset(df, data_dir=args.data_dir, img_size=512)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    # 1. Load Teachers
    t1 = load_teacher(1, args.t1_ckpt, device)
    t2 = load_teacher(2, args.t2_ckpt, device)
    t3 = load_teacher(3, args.t3_ckpt, device)

    # 2. Open HDF5 File (Append mode for resumability)
    with h5py.File(args.output_h5, "a") as h5:
        total = len(dataset)
        if "probs" not in h5:
            # Create datasets: probs shape (N, 5), masks shape (N, 4, 128, 128) - downsampled 4x to save disk
            h5.create_dataset("probs", shape=(total, 5), dtype=np.float32)
            h5.create_dataset("masks", shape=(total, 4, 128, 128), dtype=np.float16)
            h5.create_dataset("processed", shape=(total,), dtype=bool)

        processed_mask = h5["processed"][:]
        num_done = int(np.sum(processed_mask))
        print(f"[INFO] Pseudo-label cache status: {num_done}/{total} already completed.")

        pbar = tqdm(loader, desc="Generating Consensus Pseudo-Labels")
        with torch.no_grad():
            for batch in pbar:
                indices = batch["idx"].numpy()

                # Check if all indices in this batch are already done
                if np.all(processed_mask[indices]):
                    continue

                images = batch["image"].to(device)

                with torch.amp.autocast('cuda', dtype=torch.float16):
                    p1, m1 = t1.predict_class_probabilities(images)
                    p2, m2 = t2.predict_class_probabilities(images)
                    p3, m3 = t3.predict_class_probabilities(images)

                    # Consensus average
                    consensus_p = (p1 + p2 + p3) / 3.0
                    consensus_m = (m1 + m2 + m3) / 3.0

                    # Downsample masks 4x for compact storage in HDF5 (128x128)
                    m_small = torch.nn.functional.interpolate(consensus_m, size=(128, 128), mode="bilinear", align_corners=False)

                p_np = consensus_p.cpu().numpy()
                m_np = m_small.cpu().half().numpy()

                # Write batch to HDF5
                for i, global_idx in enumerate(indices):
                    h5["probs"][global_idx] = p_np[i]
                    h5["masks"][global_idx] = m_np[i]
                    h5["processed"][global_idx] = True

                h5.flush()

    print(f"\n🎉 Pseudo-label generation complete! Cached at: {args.output_h5}")


if __name__ == "__main__":
    main()
