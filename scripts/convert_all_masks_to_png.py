#!/usr/bin/env python3
"""
Convert all .npz mask archives across all datasets into standard visual .png images:
  - <image_id>_MA.png (Microaneurysms)
  - <image_id>_HE.png (Hemorrhages)
  - <image_id>_EX.png (Hard Exudates)
  - <image_id>_vessels.png (Vascular Tree)
  - <image_id>_composite.png (Color Composite: Red=HE, Green=MA, Yellow=EX, Blue=Vessels)

Removes .npz files after conversion.
"""

import os
import glob
import time
import cv2
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

def convert_single_npz(npz_file):
    try:
        data = np.load(npz_file)
        base_dir = os.path.dirname(npz_file)
        fname = os.path.basename(npz_file)
        base_name = os.path.splitext(fname)[0]

        vessels = data['vessels'].astype(np.uint8) * 255
        ma = data['microaneurysms'].astype(np.uint8) * 255
        ex = data['exudates'].astype(np.uint8) * 255
        he = data['hemorrhages'].astype(np.uint8) * 255

        cv2.imwrite(os.path.join(base_dir, f'{base_name}_vessels.png'), vessels)
        cv2.imwrite(os.path.join(base_dir, f'{base_name}_MA.png'), ma)
        cv2.imwrite(os.path.join(base_dir, f'{base_name}_EX.png'), ex)
        cv2.imwrite(os.path.join(base_dir, f'{base_name}_HE.png'), he)

        # Color composite
        comp = np.zeros((512, 512, 3), dtype=np.uint8)
        comp[vessels > 0] = [255, 0, 0]      # Blue: Vessels
        comp[ex > 0] = [0, 255, 255]        # Yellow: Hard Exudates
        comp[he > 0] = [0, 0, 255]          # Red: Hemorrhages
        comp[ma > 0] = [0, 255, 0]          # Green: Microaneurysms
        cv2.imwrite(os.path.join(base_dir, f'{base_name}_composite.png'), comp)

        os.remove(npz_file)
        return True
    except Exception as e:
        return False

def main():
    root = '/run/media/dhiru/1/data/prepared_datasets'
    all_npz = glob.glob(f'{root}/*/train_masks/*/*.npz')
    print(f"Total .npz files to convert to visual PNGs: {len(all_npz)}")
    if not all_npz:
        print("No .npz files found to convert.")
        return

    t0 = time.time()
    with ProcessPoolExecutor(max_workers=20) as executor:
        results = list(tqdm(executor.map(convert_single_npz, all_npz, chunksize=64), total=len(all_npz), desc="Converting to PNG"))

    t1 = time.time()
    print(f"Converted {sum(results)} of {len(all_npz)} files in {t1 - t0:.2f} seconds ({len(all_npz)/(t1-t0):.1f} files/sec)!")

if __name__ == '__main__':
    main()
