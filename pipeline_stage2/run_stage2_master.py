"""
DRISHYA Pipeline Stage 2 — Master Batch Processor
Iterates through prepared datasets (aptos, drive, eyepacs, idrid, messidor2),
executes the full 512x512 mathematical feature extraction pipeline via stage2_core.py,
saves all anatomical and pathological masks, and merges the 32-D feature vector into labels.csv.
"""

import os
import sys
import glob
import shutil
import warnings
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from stage2_core import run_full_pipeline, DEFAULT_RES

warnings.filterwarnings("ignore")

# Define Default Paths
INPUT_ROOT = r"../data/formated_data_processed"
OUTPUT_ROOT = r"../data/final_processed"
DATASETS = ["aptos", "drive", "eyepacs", "idrid", "messidor2"]


def process_dataset(dataset_name, input_root=INPUT_ROOT, output_root=OUTPUT_ROOT, res=DEFAULT_RES):
    input_ds_dir = os.path.join(input_root, dataset_name)
    output_ds_dir = os.path.join(output_root, dataset_name)

    if not os.path.exists(input_ds_dir):
        print(f"[{dataset_name.upper()}] Directory not found at {input_ds_dir}. Skipping.")
        return

    print(f"\n==================================================")
    print(f"[{dataset_name.upper()}] Processing dataset at {res}x{res}...")
    print(f"==================================================")

    output_img_dir = os.path.join(output_ds_dir, "images")
    os.makedirs(output_img_dir, exist_ok=True)

    # Find labels CSV
    labels_csv_path = os.path.join(input_ds_dir, "labels.csv")
    df_labels = None
    if os.path.exists(labels_csv_path):
        df_labels = pd.read_csv(labels_csv_path)

    # Gather images
    input_img_dir = os.path.join(input_ds_dir, "images")
    if not os.path.exists(input_img_dir):
        input_img_dir = input_ds_dir

    image_paths = []
    for ext in ("*.png", "*.jpg", "*.jpeg"):
        image_paths.extend(glob.glob(os.path.join(input_img_dir, ext)))
    image_paths.sort()

    print(f"Found {len(image_paths)} images for {dataset_name}.")
    if not image_paths:
        return

    features_data = []

    for img_path in tqdm(image_paths, desc=f"{dataset_name.upper()} Images"):
        try:
            filename = os.path.basename(img_path)
            img_id, ext = os.path.splitext(filename)

            target_folder = os.path.join(output_img_dir, img_id)
            if os.path.exists(os.path.join(target_folder, f"{img_id}_vessel.png")):
                continue  # Skip if already processed (resuming capability)

            os.makedirs(target_folder, exist_ok=True)

            img = cv2.imread(img_path)
            if img is None:
                continue

            # Run standardized core pipeline
            result = run_full_pipeline(img, res=res)
            features = result['features']
            features['image_id'] = img_id

            # Copy resized / raw image
            cv2.imwrite(os.path.join(target_folder, filename), result['img'])

            # Save Masks
            cv2.imwrite(os.path.join(target_folder, f"{img_id}_vessel.png"),
                        result['visual_vessel_mask'].astype(np.uint8) * 255)
            cv2.imwrite(os.path.join(target_folder, f"{img_id}_ma.png"),
                        result['ma_mask'].astype(np.uint8) * 255)
            cv2.imwrite(os.path.join(target_folder, f"{img_id}_ex.png"),
                        result['ex_mask'].astype(np.uint8) * 255)
            cv2.imwrite(os.path.join(target_folder, f"{img_id}_hemorrhage.png"),
                        result['hemorrhage_mask'].astype(np.uint8) * 255)
            cv2.imwrite(os.path.join(target_folder, f"{img_id}_od.png"),
                        result['od_mask'].astype(np.uint8) * 255)

            features_data.append(features)

        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue

    # Merge features with labels dataframe
    if features_data and df_labels is not None:
        df_features = pd.DataFrame(features_data)
        id_col = df_labels.columns[0]

        df_labels[id_col] = df_labels[id_col].astype(str)
        df_features['image_id'] = df_features['image_id'].astype(str)

        df_final = pd.merge(df_labels, df_features, left_on=id_col, right_on='image_id', how='left')
        if id_col != 'image_id':
            df_final = df_final.drop(columns=['image_id'])

        output_csv = os.path.join(output_ds_dir, "labels.csv")
        df_final.to_csv(output_csv, index=False)
        print(f"[{dataset_name.upper()}] Features successfully merged into {output_csv}")
    elif features_data:
        df_features = pd.DataFrame(features_data)
        output_csv = os.path.join(output_ds_dir, "features.csv")
        df_features.to_csv(output_csv, index=False)
        print(f"[{dataset_name.upper()}] Features saved to {output_csv}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Master Stage 2 Multi-Dataset Processor")
    parser.add_argument("--input_root", type=str, default=INPUT_ROOT)
    parser.add_argument("--output_root", type=str, default=OUTPUT_ROOT)
    parser.add_argument("--res", type=int, default=DEFAULT_RES)
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    args = parser.parse_args()

    print(f"Starting Stage 2 batch processing from {args.input_root} to {args.output_root}")
    for ds in args.datasets:
        process_dataset(ds, input_root=args.input_root, output_root=args.output_root, res=args.res)
    print("\nALL DATASETS COMPLETE.")


if __name__ == "__main__":
    main()
