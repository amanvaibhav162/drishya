#!/usr/bin/env python3
"""
DRISHYA Training Data Preprocessing Engine
==========================================
Applies:
  - Stage 1: Circular Masking, FOV Centering, Zero-Padding, CIE L*a*b* Illumination
             Homogenization, Controlled CLAHE, Bilateral Grain Suppression (512x512).
  - Stage 2: Dense Morphological Lesion Mask Extraction (Vessels, Microaneurysms,
             Exudates, Hemorrhages) saved as paired compressed .npz mask files.

Input Format:
    <data_root>/<dataset>/<split>/<grade>/<image_file>

Output Format:
    Preprocessed Images: <data_root>/<dataset>/<split>_preprocessed/<grade>/<image_file>
    Dense Lesion Masks:  <data_root>/<dataset>/<split>_masks/<grade>/<image_file>.npz

Author: DRISHYA Core Team / SIH26038
"""

import os
import sys
import cv2
import glob
import json
import argparse
import numpy as np
import multiprocessing as mp
from functools import partial
from tqdm import tqdm

# Ensure project root is in path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from pre_processing_pipeline.batch_process_usb import (
    extract_retinal_mask,
    assess_quality,
    evaluate_iqa,
    adaptive_enhance
)
from pipeline_stage2.stage2_core import run_full_pipeline

DEFAULT_THRESHOLDS = {
    'F_target': 0.0015,
    'C_target': 0.10,
    'Q_reject': 0.55,
    'Q_good': 0.80
}


def process_single_fundus(task_args):
    """
    Worker task: processes a single image through Stage 1 & Stage 2.
    """
    src_path, dst_path, mask_dst_path, target_size, extract_lesions = task_args

    img_done = os.path.exists(dst_path) and os.path.getsize(dst_path) > 0
    mask_done = (not extract_lesions) or (mask_dst_path and os.path.exists(mask_dst_path) and os.path.getsize(mask_dst_path) > 0)

    if img_done and mask_done:
        return {"status": "SKIPPED", "path": src_path}

    try:
        # Load or generate Stage 1 preprocessed image
        if img_done:
            enhanced_img = cv2.imread(dst_path)
            if enhanced_img is None:
                img_done = False

        if not img_done:
            raw_img = cv2.imread(src_path)
            if raw_img is None:
                return {"status": "ERROR_UNREADABLE", "path": src_path}

            # 1. Circular Masking & FOV Centering
            mask, cropped_img = extract_retinal_mask(raw_img)

            # 2. Resize to target resolution (512x512)
            cropped_img = cv2.resize(cropped_img, (target_size, target_size), interpolation=cv2.INTER_AREA)
            mask = cv2.resize(mask, (target_size, target_size), interpolation=cv2.INTER_NEAREST)
            mask = (mask > 127).astype(np.uint8) * 255

            # 3. Assess IQA & Adaptive Enhance
            metrics = assess_quality(cropped_img, mask)
            iqa_status, _, q_score = evaluate_iqa(metrics, DEFAULT_THRESHOLDS)
            enhanced_img = adaptive_enhance(cropped_img, mask, metrics, DEFAULT_THRESHOLDS)

            # Save preprocessed image
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            cv2.imwrite(dst_path, enhanced_img)

        # 4. Stage 2 Dense Morphological Lesion Mask Extraction
        if extract_lesions and mask_dst_path and not (os.path.exists(mask_dst_path) and os.path.getsize(mask_dst_path) > 0):
            res_stage2 = run_full_pipeline(enhanced_img, res=target_size)
            
            os.makedirs(os.path.dirname(mask_dst_path), exist_ok=True)
            np.savez_compressed(
                mask_dst_path,
                vessels=res_stage2.get('visual_vessel_mask'),
                microaneurysms=res_stage2.get('ma_mask'),
                exudates=res_stage2.get('ex_mask'),
                hemorrhages=res_stage2.get('hemorrhage_mask'),
                optic_disc=res_stage2.get('od_mask'),
                fovea=res_stage2.get('fovea_mask')
            )

        return {"status": "SUCCESS", "path": src_path}

    except Exception as e:
        return {"status": f"ERROR: {str(e)}", "path": src_path}


def prepare_dataset_split(data_root, dataset_name, split="train", target_size=512, 
                           workers=20, batch_limit=None, extract_lesions=True):
    """
    Processes all images for a specific dataset split (e.g. aptos/train/).
    """
    src_split_dir = os.path.join(data_root, dataset_name, split)
    if not os.path.exists(src_split_dir):
        print(f"[SKIP] Directory does not exist: {src_split_dir}")
        return

    dst_split_dir = os.path.join(data_root, dataset_name, f"{split}_preprocessed")
    mask_split_dir = os.path.join(data_root, dataset_name, f"{split}_masks") if extract_lesions else None

    print("\n" + "="*60)
    print(f"Preparing Dataset: {dataset_name} | Split: {split}")
    print(f"  Source:       {src_split_dir}")
    print(f"  Stage 1 Dest: {dst_split_dir}")
    if extract_lesions:
        print(f"  Stage 2 Dest: {mask_split_dir}")
    print(f"  Resolution:   {target_size}x{target_size}")
    print(f"  Workers:      {workers} CPU cores")
    print("="*60)

    exts = ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff")
    tasks = []
    
    for grade_dir in sorted(os.listdir(src_split_dir)):
        grade_path = os.path.join(src_split_dir, grade_dir)
        if not os.path.isdir(grade_path):
            continue
            
        grade_files = []
        for ext in exts:
            grade_files.extend(glob.glob(os.path.join(grade_path, ext)))
            
        grade_files = sorted(grade_files)
        if batch_limit and len(grade_files) > batch_limit:
            grade_files = grade_files[:batch_limit]

        for src_f in grade_files:
            fname = os.path.basename(src_f)
            dst_f = os.path.join(dst_split_dir, grade_dir, fname)
            base_name = os.path.splitext(fname)[0]
            mask_f = os.path.join(mask_split_dir, grade_dir, base_name + ".npz") if extract_lesions else None
            tasks.append((src_f, dst_f, mask_f, target_size, extract_lesions))

    print(f"Total images in queue: {len(tasks)}")
    if not tasks:
        return

    success_count = 0
    skipped_count = 0
    error_count = 0

    with mp.Pool(processes=workers) as pool:
        for result in tqdm(pool.imap_unordered(process_single_fundus, tasks, chunksize=16), total=len(tasks), desc=f"Stage 1+2 {dataset_name} ({split})"):
            st = result.get("status", "")
            if st == "SUCCESS":
                success_count += 1
            elif st == "SKIPPED":
                skipped_count += 1
            else:
                error_count += 1

    print(f"\n[DONE] {dataset_name} ({split}) Summary:")
    print(f"  Success (Processed): {success_count}")
    print(f"  Skipped (Already done): {skipped_count}")
    print(f"  Errors: {error_count}")


def main():
    parser = argparse.ArgumentParser(description="DRISHYA Stage 1 + Stage 2 Batch Preprocessor")
    parser.add_argument("--data_root", type=str, default="/run/media/dhiru/1/data/prepared_datasets", help="Root directory of prepared datasets")
    parser.add_argument("--dataset", type=str, choices=["aptos", "idrid", "messidor2", "ddr", "eyepacs", "all"], default="all", help="Target dataset")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test", "all"], help="Split to process")
    parser.add_argument("--size", type=int, default=512, help="Target image size (default 512)")
    parser.add_argument("--workers", type=int, default=20, help="Worker processes")
    parser.add_argument("--batch_limit", type=int, default=None, help="Limit number of images per grade (for testing)")
    parser.add_argument("--extract_lesions", action="store_true", default=True, help="Extract Stage 2 lesion masks")
    parser.add_argument("--no_lesions", dest="extract_lesions", action="store_false", help="Skip Stage 2 lesion masks")
    args = parser.parse_args()

    target_datasets = ["aptos", "idrid", "messidor2", "ddr", "eyepacs"] if args.dataset == "all" else [args.dataset]
    target_splits = ["train", "val", "test"] if args.split == "all" else [args.split]

    for d in target_datasets:
        for s in target_splits:
            prepare_dataset_split(
                data_root=args.data_root,
                dataset_name=d,
                split=s,
                target_size=args.size,
                workers=args.workers,
                batch_limit=args.batch_limit,
                extract_lesions=args.extract_lesions
            )


if __name__ == '__main__':
    main()
