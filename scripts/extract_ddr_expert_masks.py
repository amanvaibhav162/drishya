#!/usr/bin/env python3
"""
DRISHYA DDR Expert Lesion Ground-Truth Standardizer
===================================================
Extracts the 757 ophthalmologist-annotated pixel masks from DDR:
  - EX: Hard Exudates
  - HE: Hemorrhages
  - MA: Microaneurysms
  - SE: Soft Exudates (Cotton Wool Spots)

Standardizes images and masks to 512x512 square FOV with identical spatial alignment.
Saves to:
  <output_root>/ddr/expert_ground_truth_masks/<split>/images/<id>.png
  <output_root>/ddr/expert_ground_truth_masks/<split>/masks/<id>_EX.png
  <output_root>/ddr/expert_ground_truth_masks/<split>/masks/<id>_HE.png
  <output_root>/ddr/expert_ground_truth_masks/<split>/masks/<id>_MA.png
  <output_root>/ddr/expert_ground_truth_masks/<split>/masks/<id>_SE.png
  <output_root>/ddr/expert_ground_truth_masks/<split>/masks/<id>_composite.png

Author: DRISHYA Core Team / SIH26038
"""

import os
import sys
import glob
import cv2
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from pre_processing_pipeline.batch_process_usb import (
    extract_retinal_mask,
    assess_quality,
    evaluate_iqa,
    adaptive_enhance
)

DEFAULT_THRESHOLDS = {
    'F_target': 0.0015,
    'C_target': 0.10,
    'Q_reject': 0.55,
    'Q_good': 0.80
}

def process_expert_sample(args):
    img_path, mask_dict, out_img_path, out_mask_dir, base_name, target_size = args
    try:
        raw_img = cv2.imread(img_path)
        if raw_img is None:
            return False

        h, w = raw_img.shape[:2]

        # 1. Circular Masking & FOV Centering Bounding Box
        mask, cropped_img = extract_retinal_mask(raw_img)

        # Scale factor and spatial crop bounds
        cropped_img = cv2.resize(cropped_img, (target_size, target_size), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, (target_size, target_size), interpolation=cv2.INTER_NEAREST)
        mask = (mask > 127).astype(np.uint8) * 255

        # Illumination normalization
        metrics = assess_quality(cropped_img, mask)
        enhanced_img = adaptive_enhance(cropped_img, mask, metrics, DEFAULT_THRESHOLDS)

        os.makedirs(os.path.dirname(out_img_path), exist_ok=True)
        os.makedirs(out_mask_dir, exist_ok=True)
        cv2.imwrite(out_img_path, enhanced_img)

        # 2. Process each of the 4 expert lesion masks with identical square crop
        # Derive bounding square coordinates from mask
        g = raw_img[:, :, 1]
        _, thresh = cv2.threshold(g, 15, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            x, y, bw, bh = cv2.boundingRect(largest)
            side = max(bw, bh)
            cx, cy = x + bw // 2, y + bh // 2
            x1 = max(0, cx - side // 2)
            y1 = max(0, cy - side // 2)
            x2 = min(w, x1 + side)
            y2 = min(h, y1 + side)
        else:
            x1, y1, x2, y2 = 0, 0, w, h

        lesion_arrays = {}
        for l_type in ['EX', 'HE', 'MA', 'SE']:
            mask_src = mask_dict.get(l_type)
            if mask_src and os.path.exists(mask_src):
                m_raw = cv2.imread(mask_src, cv2.IMREAD_GRAYSCALE)
                if m_raw is not None:
                    # Crop square
                    m_cropped = m_raw[y1:y2, x1:x2]
                    # Resize to 512x512
                    m_512 = cv2.resize(m_cropped, (target_size, target_size), interpolation=cv2.INTER_NEAREST)
                    m_bin = (m_512 > 127).astype(np.uint8) * 255
                    lesion_arrays[l_type] = m_bin
                else:
                    lesion_arrays[l_type] = np.zeros((target_size, target_size), dtype=np.uint8)
            else:
                lesion_arrays[l_type] = np.zeros((target_size, target_size), dtype=np.uint8)

            cv2.imwrite(os.path.join(out_mask_dir, f"{base_name}_{l_type}.png"), lesion_arrays[l_type])

        # 3. Color composite
        comp = np.zeros((target_size, target_size, 3), dtype=np.uint8)
        comp[lesion_arrays['EX'] > 0] = [0, 255, 255]    # Yellow: Hard Exudates
        comp[lesion_arrays['HE'] > 0] = [0, 0, 255]      # Red: Hemorrhages
        comp[lesion_arrays['MA'] > 0] = [0, 255, 0]      # Green: Microaneurysms
        comp[lesion_arrays['SE'] > 0] = [255, 255, 255]  # White: Soft Exudates (Cotton Wool Spots)
        cv2.imwrite(os.path.join(out_mask_dir, f"{base_name}_composite.png"), comp)

        return True
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return False

def main():
    ddr_base = "/run/media/dhiru/1/data/ddr-dataset/DDR-dataset/lesion_segmentation"
    out_base = "/run/media/dhiru/1/data/prepared_datasets/ddr/expert_ground_truth_masks"

    splits = {
        'train': ('train/image', 'train/label'),
        'valid': ('valid/image', 'valid/segmentation label'),
        'test':  ('test/image',  'test/label')
    }

    all_tasks = []
    for split, (img_sub, label_sub) in splits.items():
        img_dir = os.path.join(ddr_base, img_sub)
        label_dir = os.path.join(ddr_base, label_sub)
        if not os.path.exists(img_dir):
            continue

        images = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
        print(f"DDR Expert {split}: {len(images)} images found.")

        for img_p in images:
            fname = os.path.basename(img_p)
            base = os.path.splitext(fname)[0]

            mask_dict = {
                'EX': os.path.join(label_dir, 'EX', f"{base}.tif"),
                'HE': os.path.join(label_dir, 'HE', f"{base}.tif"),
                'MA': os.path.join(label_dir, 'MA', f"{base}.tif"),
                'SE': os.path.join(label_dir, 'SE', f"{base}.tif"),
            }

            out_img = os.path.join(out_base, split, 'images', f"{base}.png")
            out_mask_d = os.path.join(out_base, split, 'masks')
            all_tasks.append((img_p, mask_dict, out_img, out_mask_d, base, 512))

    print(f"Total Expert DDR Samples to Standardize: {len(all_tasks)}")
    with ProcessPoolExecutor(max_workers=20) as executor:
        results = list(tqdm(executor.map(process_expert_sample, all_tasks), total=len(all_tasks), desc="Processing DDR Expert Masks"))

    print(f"Successfully processed {sum(results)} of {len(all_tasks)} DDR Expert Mask Samples!")
    print(f"Saved to: {out_base}")

if __name__ == '__main__':
    main()
