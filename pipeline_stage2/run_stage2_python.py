"""
DRISHYA Pipeline Stage 2 — Python Runner
Runs the full 512x512 mathematical feature extraction pipeline on retinal fundus images.
Imports all core detection algorithms from stage2_core.py (single source of truth).
"""

import os
import glob
import time
import argparse
import cv2
import numpy as np
import pandas as pd
from stage2_core import run_full_pipeline, DEFAULT_RES

# Default paths
DEFAULT_IMG_DIR = "../data/processed/dr_images"
DEFAULT_OUTPUT_DIR = "../data/processed/stage2_test_python"


def process_dataset(img_dir, output_dir, res=DEFAULT_RES, max_images=None):
    mask_dir = os.path.join(output_dir, "masks")
    os.makedirs(mask_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "stage2_features_python.csv")

    exts = ("*.png", "*.jpg", "*.jpeg")
    test_files = []
    for ext in exts:
        test_files.extend(glob.glob(os.path.join(img_dir, ext)))
    test_files.sort()

    if max_images is not None and max_images > 0:
        test_files = test_files[:max_images]

    print(f"=================================================================")
    print(f"DRISHYA Stage 2 Python Pipeline (Resolution: {res}x{res})")
    print(f"Images to process: {len(test_files)}")
    print(f"Output directory: {output_dir}")
    print(f"=================================================================")

    if not test_files:
        print(f"No images found in {img_dir}.")
        return

    csv_rows = []

    for i, file_path in enumerate(test_files):
        t0 = time.time()
        img_name = os.path.basename(file_path)
        base_name = os.path.splitext(img_name)[0]

        img = cv2.imread(file_path)
        if img is None:
            print(f"[{i+1}/{len(test_files)}] Failed to read {img_name}")
            continue

        result = run_full_pipeline(img, res=res)
        features = result['features']
        features['Image_Name'] = img_name

        # Put Image_Name first
        ordered_row = {'Image_Name': img_name}
        for k, v in features.items():
            if k != 'Image_Name':
                ordered_row[k] = v
        csv_rows.append(ordered_row)

        # Save individual masks
        cv2.imwrite(os.path.join(mask_dir, f"{base_name}_vessel.png"),
                    result['visual_vessel_mask'].astype(np.uint8) * 255)
        cv2.imwrite(os.path.join(mask_dir, f"{base_name}_ma.png"),
                    result['ma_mask'].astype(np.uint8) * 255)
        cv2.imwrite(os.path.join(mask_dir, f"{base_name}_ex.png"),
                    result['ex_mask'].astype(np.uint8) * 255)
        cv2.imwrite(os.path.join(mask_dir, f"{base_name}_hemorrhage.png"),
                    result['hemorrhage_mask'].astype(np.uint8) * 255)
        cv2.imwrite(os.path.join(mask_dir, f"{base_name}_od.png"),
                    result['od_mask'].astype(np.uint8) * 255)

        elapsed = time.time() - t0
        print(f"[{i+1}/{len(test_files)}] {img_name} ({elapsed:.2f}s) — "
              f"MA: {features.get('MA_Count', 0)}, "
              f"EX%: {features.get('EX_Area_Pct', 0.0):.2f}%, "
              f"Hem%: {features.get('HEM_Total_Pct', 0.0):.2f}%, "
              f"Vessel%: {features.get('Vessel_Coverage_Pct', 0.0):.1f}%")

    df = pd.DataFrame(csv_rows)
    df.to_csv(csv_path, index=False)
    print(f"\nPipeline complete. {len(csv_rows)} rows saved to {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="DRISHYA Stage 2 Feature Extraction")
    parser.add_argument("--img_dir", type=str, default=DEFAULT_IMG_DIR, help="Input directory containing fundus images")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Output directory for masks and CSV")
    parser.add_argument("--res", type=int, default=DEFAULT_RES, help="Target resolution (default: 512)")
    parser.add_argument("--max_images", type=int, default=None, help="Maximum number of images to process")
    args = parser.parse_args()

    process_dataset(args.img_dir, args.output_dir, res=args.res, max_images=args.max_images)


if __name__ == "__main__":
    main()
