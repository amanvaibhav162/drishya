#!/usr/bin/env python3
"""
DRISHYA Data Extraction and Restructuring Engine
=================================================
Extracts and stratifies multi-source clinical DR datasets into standard 
PyTorch ImageFolder directory format:
    <output_dir>/<dataset_name>/train/<grade>/<image_id>.<ext>
    <output_dir>/<dataset_name>/test/<grade>/<image_id>.<ext>
    <output_dir>/<dataset_name>/val/<grade>/<image_id>.<ext> (where applicable)

Author: DRISHYA Core Team / SIH26038
"""

import os
import sys
import csv
import glob
import shutil
import zipfile
import argparse
import random
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


def set_seed(seed=42):
    random.seed(seed)


def safe_extract_member(zf, member_name, target_path):
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    if os.path.exists(target_path) and os.path.getsize(target_path) > 0:
        return True
    with zf.open(member_name) as src, open(target_path, 'wb') as dst:
        shutil.copyfileobj(src, dst)
    return True


def stratified_split(items_by_grade, train_ratio=0.8, seed=42):
    rnd = random.Random(seed)
    train_items = []
    test_items = []
    for grade, items in items_by_grade.items():
        shuffled = list(items)
        rnd.shuffle(shuffled)
        split_idx = int(len(shuffled) * train_ratio)
        train_items.extend([(x, grade) for x in shuffled[:split_idx]])
        test_items.extend([(x, grade) for x in shuffled[split_idx:]])
    return train_items, test_items


# =====================================================================
# 1. APTOS 2019
# =====================================================================
def process_aptos(ssd_root, output_dir, split_ratio=0.8, seed=42, max_workers=8):
    print("\n" + "="*60)
    print("Processing APTOS 2019 Dataset...")
    print("="*60)
    
    zip_path = os.path.join(ssd_root, 'aptos2019-blindness-detection', 'aptos2019-blindness-detection.zip')
    if not os.path.exists(zip_path):
        print(f"[ERROR] APTOS zip not found at: {zip_path}")
        return []

    records = []
    with zipfile.ZipFile(zip_path, 'r') as zf:
        if 'train.csv' not in zf.namelist():
            print("[ERROR] train.csv not found in APTOS zip!")
            return []
            
        with zf.open('train.csv') as f:
            reader = csv.DictReader(line.decode('utf-8', errors='ignore') for line in f)
            items_by_grade = defaultdict(list)
            for row in reader:
                id_code = row['id_code'].strip()
                diagnosis = row['diagnosis'].strip()
                if diagnosis in {'0', '1', '2', '3', '4'}:
                    items_by_grade[diagnosis].append(id_code)

        print(f"APTOS Total Images: {sum(len(v) for v in items_by_grade.values())}")
        for g in sorted(items_by_grade.keys()):
            print(f"  Grade {g}: {len(items_by_grade[g])} images")

        train_set, test_set = stratified_split(items_by_grade, train_ratio=split_ratio, seed=seed)
        print(f"Stratified Split ({split_ratio*100:.0f}/{(1-split_ratio)*100:.0f}): Train={len(train_set)}, Test={len(test_set)}")

        tasks = []
        for id_code, grade in train_set:
            member = f"train_images/{id_code}.png"
            dest = os.path.join(output_dir, 'aptos', 'train', grade, f"{id_code}.png")
            tasks.append((member, dest, 'aptos', 'train', id_code, grade))
            
        for id_code, grade in test_set:
            member = f"train_images/{id_code}.png"
            dest = os.path.join(output_dir, 'aptos', 'test', grade, f"{id_code}.png")
            tasks.append((member, dest, 'aptos', 'test', id_code, grade))

        print(f"Extracting {len(tasks)} images using {max_workers} worker threads...")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(safe_extract_member, zf, t[0], t[1]): t for t in tasks
            }
            for fut in tqdm(as_completed(futures), total=len(tasks), desc="Extracting APTOS"):
                fut.result()

        for t in tasks:
            records.append({
                'dataset': t[2],
                'split': t[3],
                'image_id': t[4],
                'grade': t[5],
                'path': t[1]
            })

    print(f"[SUCCESS] APTOS extracted and restructured into: {os.path.join(output_dir, 'aptos')}")
    return records


# =====================================================================
# 2. IDRiD (Indian DR Image Dataset)
# =====================================================================
def process_idrid(ssd_root, output_dir, max_workers=8):
    print("\n" + "="*60)
    print("Processing IDRiD Dataset...")
    print("="*60)
    
    zip_path = os.path.join(ssd_root, 'idrid-dataset', 'idrid-dataset.zip')
    csv_path = os.path.join(ssd_root, 'idrid-dataset', 'idrid_labels.csv')
    
    if not os.path.exists(zip_path) or not os.path.exists(csv_path):
        print(f"[ERROR] IDRiD zip or csv missing at: {zip_path}, {csv_path}")
        return []

    label_map = {}
    with open(csv_path, 'r', encoding='utf-8', errors='ignore') as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            if not row or not row[0].strip().startswith('IDRiD'):
                continue
            id_code = row[0].strip()
            grade = row[1].strip() if len(row) > 1 else ''
            if grade in {'0', '1', '2', '3', '4'}:
                label_map[id_code] = grade

    print(f"IDRiD labeled entries: {len(label_map)}")
    
    records = []
    tasks = []
    with zipfile.ZipFile(zip_path, 'r') as zf:
        namelist_set = set(zf.namelist())
        for id_code, grade in label_map.items():
            member = f"Imagenes/Imagenes/{id_code}.jpg"
            if member not in namelist_set:
                found = [m for m in namelist_set if f"{id_code}.jpg" in m]
                if found:
                    member = found[0]
                else:
                    continue

            is_test = 'test' in id_code.lower()
            split = 'test' if is_test else 'train'
            dest = os.path.join(output_dir, 'idrid', split, grade, f"{id_code}.jpg")
            tasks.append((member, dest, 'idrid', split, id_code, grade))

        print(f"Extracting {len(tasks)} IDRiD images...")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(safe_extract_member, zf, t[0], t[1]): t for t in tasks
            }
            for fut in tqdm(as_completed(futures), total=len(tasks), desc="Extracting IDRiD"):
                fut.result()

        for t in tasks:
            records.append({
                'dataset': t[2],
                'split': t[3],
                'image_id': t[4],
                'grade': t[5],
                'path': t[1]
            })

    print(f"[SUCCESS] IDRiD extracted and restructured into: {os.path.join(output_dir, 'idrid')}")
    return records


# =====================================================================
# 3. Messidor-2
# =====================================================================
def process_messidor2(ssd_root, output_dir, split_ratio=0.8, seed=42, max_workers=8):
    print("\n" + "="*60)
    print("Processing Messidor-2 Dataset...")
    print("="*60)
    
    img_zip_path = os.path.join(ssd_root, 'messidor-2-reference-grades', 'messidor-2-dataset.zip')
    grades_zip_path = os.path.join(ssd_root, 'messidor-2-reference-grades', 'messidor2-dr-grades.zip')
    
    if not os.path.exists(img_zip_path) or not os.path.exists(grades_zip_path):
        print(f"[ERROR] Messidor-2 zip files missing!")
        return []

    items_by_grade = defaultdict(list)
    with zipfile.ZipFile(grades_zip_path, 'r') as zf_grades:
        with zf_grades.open('messidor_data.csv') as f:
            reader = csv.DictReader(line.decode('utf-8', errors='ignore') for line in f)
            for row in reader:
                img_id = row['image_id'].strip()
                grade = row.get('adjudicated_dr_grade', '').strip()
                gradable = row.get('adjudicated_gradable', '1').strip()
                if gradable == '1' and grade in {'0', '1', '2', '3', '4'}:
                    items_by_grade[grade].append(img_id)

    total_valid = sum(len(v) for v in items_by_grade.values())
    print(f"Messidor-2 Adjudicated Gradable Images: {total_valid}")
    for g in sorted(items_by_grade.keys()):
        print(f"  Grade {g}: {len(items_by_grade[g])} images")

    train_set, test_set = stratified_split(items_by_grade, train_ratio=split_ratio, seed=seed)
    print(f"Stratified Split: Train={len(train_set)}, Test={len(test_set)}")

    records = []
    tasks = []
    with zipfile.ZipFile(img_zip_path, 'r') as zf_img:
        nl_set = set(zf_img.namelist())
        for img_id, grade in train_set:
            member = f"IMAGES/{img_id}"
            if member in nl_set:
                dest = os.path.join(output_dir, 'messidor2', 'train', grade, img_id)
                tasks.append((member, dest, 'messidor2', 'train', img_id, grade))
                
        for img_id, grade in test_set:
            member = f"IMAGES/{img_id}"
            if member in nl_set:
                dest = os.path.join(output_dir, 'messidor2', 'test', grade, img_id)
                tasks.append((member, dest, 'messidor2', 'test', img_id, grade))

        print(f"Extracting {len(tasks)} Messidor-2 images...")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(safe_extract_member, zf_img, t[0], t[1]): t for t in tasks
            }
            for fut in tqdm(as_completed(futures), total=len(tasks), desc="Extracting Messidor-2"):
                fut.result()

        for t in tasks:
            records.append({
                'dataset': t[2],
                'split': t[3],
                'image_id': t[4],
                'grade': t[5],
                'path': t[1]
            })

    print(f"[SUCCESS] Messidor-2 extracted and restructured into: {os.path.join(output_dir, 'messidor2')}")
    return records


# =====================================================================
# 4. DDR Dataset
# =====================================================================
def process_ddr(ssd_root, output_dir, max_workers=8):
    print("\n" + "="*60)
    print("Processing DDR Dataset (Official Train/Val/Test Splits)...")
    print("="*60)
    
    zip_path = os.path.join(ssd_root, 'ddr-dataset', 'ddr-dataset-credits-to-authors.zip')
    if not os.path.exists(zip_path):
        print(f"[ERROR] DDR zip not found at: {zip_path}")
        return []

    records = []
    tasks = []
    with zipfile.ZipFile(zip_path, 'r') as zf:
        splits = {
            'train': 'DDR-dataset/DR_grading/train.txt',
            'val': 'DDR-dataset/DR_grading/valid.txt',
            'test': 'DDR-dataset/DR_grading/test.txt'
        }
        
        for split, txt_path in splits.items():
            if txt_path not in zf.namelist():
                continue
            with zf.open(txt_path) as f:
                lines = [l.decode('utf-8', errors='ignore').strip().split() for l in f if l.strip()]
                grade_counts = Counter()
                for parts in lines:
                    if len(parts) >= 2:
                        img_name, grade = parts[0], parts[1]
                        if grade in {'0', '1', '2', '3', '4'}:
                            grade_counts[grade] += 1
                            folder_name = 'valid' if split == 'val' else split
                            member = f"DDR-dataset/DR_grading/{folder_name}/{img_name}"
                            dest = os.path.join(output_dir, 'ddr', split, grade, img_name)
                            tasks.append((member, dest, 'ddr', split, img_name, grade))
                        elif grade == '5':
                            dest = os.path.join(output_dir, 'ddr', f"{split}_ungradable", img_name)
                            folder_name = 'valid' if split == 'val' else split
                            member = f"DDR-dataset/DR_grading/{folder_name}/{img_name}"
                            tasks.append((member, dest, 'ddr', f"{split}_ungradable", img_name, '5'))

                print(f"DDR {split} Clinical Grades: {dict(sorted(grade_counts.items()))}")

        print(f"Extracting {len(tasks)} DDR images...")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(safe_extract_member, zf, t[0], t[1]): t for t in tasks
            }
            for fut in tqdm(as_completed(futures), total=len(tasks), desc="Extracting DDR"):
                fut.result()

        for t in tasks:
            records.append({
                'dataset': t[2],
                'split': t[3],
                'image_id': t[4],
                'grade': t[5],
                'path': t[1]
            })

    print(f"[SUCCESS] DDR extracted and restructured into: {os.path.join(output_dir, 'ddr')}")
    return records


# =====================================================================
# 5. EyePACS (Resized 512px Mirror)
# =====================================================================
def process_eyepacs(ssd_root, output_dir, max_per_grade=None, max_workers=8):
    print("\n" + "="*60)
    print("Processing EyePACS (Resized 512px Mirror)...")
    print("="*60)
    
    zip_path = os.path.join(ssd_root, 'resized-2015-2019-blindness-detection-images', 'resized-2015-2019-blindness-detection-images.zip')
    if not os.path.exists(zip_path):
        print(f"[ERROR] EyePACS zip not found at: {zip_path}")
        return []

    records = []
    tasks = []
    with zipfile.ZipFile(zip_path, 'r') as zf:
        if 'labels/trainLabels15.csv' in zf.namelist():
            with zf.open('labels/trainLabels15.csv') as f:
                reader = csv.DictReader(line.decode('utf-8', errors='ignore') for line in f)
                items_by_grade = defaultdict(list)
                for row in reader:
                    img = row['image'].strip()
                    lvl = row['level'].strip()
                    if lvl in {'0', '1', '2', '3', '4'}:
                        items_by_grade[lvl].append(img)

                print("EyePACS Train 15 Total per Grade:")
                for lvl, imgs in sorted(items_by_grade.items()):
                    print(f"  Grade {lvl}: {len(imgs)}")
                    if max_per_grade and len(imgs) > max_per_grade:
                        selected = random.sample(imgs, max_per_grade)
                    else:
                        selected = imgs
                    for img in selected:
                        member = f"resized train 15/{img}.jpg"
                        dest = os.path.join(output_dir, 'eyepacs', 'train', lvl, f"{img}.jpg")
                        tasks.append((member, dest, 'eyepacs', 'train', img, lvl))

        if 'labels/testLabels15.csv' in zf.namelist():
            with zf.open('labels/testLabels15.csv') as f:
                reader = csv.DictReader(line.decode('utf-8', errors='ignore') for line in f)
                test_by_grade = defaultdict(list)
                for row in reader:
                    img = row['image'].strip()
                    lvl = row['level'].strip()
                    if lvl in {'0', '1', '2', '3', '4'}:
                        test_by_grade[lvl].append(img)

                print("EyePACS Test 15 Total per Grade:")
                test_max = (max_per_grade // 4) if max_per_grade else None
                for lvl, imgs in sorted(test_by_grade.items()):
                    print(f"  Grade {lvl}: {len(imgs)}")
                    if test_max and len(imgs) > test_max:
                        selected = random.sample(imgs, test_max)
                    else:
                        selected = imgs
                    for img in selected:
                        member = f"resized test 15/{img}.jpg"
                        dest = os.path.join(output_dir, 'eyepacs', 'test', lvl, f"{img}.jpg")
                        tasks.append((member, dest, 'eyepacs', 'test', img, lvl))

        print(f"Extracting {len(tasks)} EyePACS images using {max_workers} threads...")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(safe_extract_member, zf, t[0], t[1]): t for t in tasks
            }
            for fut in tqdm(as_completed(futures), total=len(tasks), desc="Extracting EyePACS"):
                fut.result()

        for t in tasks:
            records.append({
                'dataset': t[2],
                'split': t[3],
                'image_id': t[4],
                'grade': t[5],
                'path': t[1]
            })

    print(f"[SUCCESS] EyePACS extracted and restructured into: {os.path.join(output_dir, 'eyepacs')}")
    return records


# =====================================================================
# Main CLI Runner
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description="DRISHYA Dataset Extraction and Restructuring")
    parser.add_argument("--ssd_root", type=str, default="/run/media/dhiru/1/data", help="Root path to downloaded datasets on SSD")
    parser.add_argument("--output_dir", type=str, default="/run/media/dhiru/1/data/prepared_datasets", help="Destination directory for restructured ImageFolder format")
    parser.add_argument("--dataset", type=str, choices=["aptos", "idrid", "messidor2", "ddr", "eyepacs", "all"], default="all", help="Dataset to process")
    parser.add_argument("--split_ratio", type=float, default=0.8, help="Train/test split ratio for un-split datasets (e.g. APTOS, Messidor-2)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible stratification")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel extraction threads")
    parser.add_argument("--eyepacs_limit", type=int, default=None, help="Max images per grade for EyePACS")
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    all_records = []

    print("="*60)
    print("Starting DRISHYA Data Restructuring")
    print(f"  SSD Root:   {args.ssd_root}")
    print(f"  Output Dir: {args.output_dir}")
    print(f"  Target:     {args.dataset}")
    print("="*60)

    if args.dataset in ["aptos", "all"]:
        all_records.extend(process_aptos(args.ssd_root, args.output_dir, args.split_ratio, args.seed, args.workers))

    if args.dataset in ["idrid", "all"]:
        all_records.extend(process_idrid(args.ssd_root, args.output_dir, args.workers))

    if args.dataset in ["messidor2", "all"]:
        all_records.extend(process_messidor2(args.ssd_root, args.output_dir, args.split_ratio, args.seed, args.workers))

    if args.dataset in ["ddr", "all"]:
        all_records.extend(process_ddr(args.ssd_root, args.output_dir, args.workers))

    if args.dataset in ["eyepacs", "all"]:
        all_records.extend(process_eyepacs(args.ssd_root, args.output_dir, args.eyepacs_limit, args.workers))

    # Save manifest
    manifest_path = os.path.join(args.output_dir, "dataset_manifest.csv")
    file_exists = os.path.exists(manifest_path)
    with open(manifest_path, 'a' if file_exists else 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['dataset', 'split', 'image_id', 'grade', 'path'])
        if not file_exists:
            writer.writeheader()
        for r in all_records:
            writer.writerow(r)

    print("\n" + "="*60)
    print(f"Restructuring Complete! Total records processed: {len(all_records)}")
    print(f"Manifest saved at: {manifest_path}")
    print("="*60)


if __name__ == '__main__':
    main()
