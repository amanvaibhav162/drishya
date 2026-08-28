import os
import cv2
import numpy as np
import pandas as pd
import time
import glob
import shutil
import warnings
from skimage.filters import frangi
from skimage.measure import regionprops, label
from tqdm import tqdm

warnings.filterwarnings("ignore")

# Define Paths
INPUT_ROOT = r"../data/formated_data_processed"
OUTPUT_ROOT = r"../data/final_processed"

DATASETS = ["aptos", "drive", "eyepacs", "idrid", "messidor2"]

def extract_landmarks(img):
    imgH, imgW = img.shape[:2]
    r_chan = img[:,:,2] 
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    r_closed = cv2.morphologyEx(r_chan, cv2.MORPH_CLOSE, kernel)
    
    circles = cv2.HoughCircles(r_closed, cv2.HOUGH_GRADIENT, dp=1, minDist=100,
                               param1=50, param2=30, minRadius=20, maxRadius=45)
                               
    od_mask = np.zeros((imgH, imgW), dtype=np.uint8)
    od_center = (imgW//2, imgH//2)
    od_radius = 30
    
    if circles is not None:
        circles = np.uint16(np.around(circles))
        best_circle = circles[0, 0] 
        od_center = (best_circle[0], best_circle[1])
        od_radius = best_circle[2]
    else:
        blurred = cv2.GaussianBlur(r_closed, (21, 21), 0)
        _, _, _, max_loc = cv2.minMaxLoc(blurred)
        od_center = max_loc
        
    cv2.circle(od_mask, od_center, od_radius, 255, -1)
    return od_mask > 0

def extract_vessels(img):
    g_chan = img[:,:,1]
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    g_enh = clahe.apply(g_chan)
    g_inv = cv2.bitwise_not(g_enh)
    
    vesselness = frangi(g_inv, sigmas=range(1, 8, 2), black_ridges=False)
    vesselness_norm = cv2.normalize(vesselness, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Adaptive threshold for robustness across different datasets
    thresh = cv2.threshold(vesselness_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]
    vessel_mask = vesselness_norm > (thresh * 0.5) 
    
    vessel_mask = vessel_mask.astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(vessel_mask, connectivity=8)
    clean_vessel_mask = np.zeros_like(vessel_mask)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= 30:
            clean_vessel_mask[labels == i] = 1
            
    return clean_vessel_mask > 0

def detect_lesions(img, vessel_mask, od_mask):
    g_chan = img[:,:,1]
    
    # 1. Microaneurysms
    kernel_ma = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    ma_candidates = cv2.morphologyEx(g_chan, cv2.MORPH_BLACKHAT, kernel_ma)
    
    ma_mask = ma_candidates > 15
    ma_mask[vessel_mask] = False
    ma_mask[od_mask] = False
    
    ma_mask_uint = ma_mask.astype(np.uint8)
    labeled_ma = label(ma_mask_uint)
    regions = regionprops(labeled_ma)
    
    clean_ma_mask = np.zeros_like(ma_mask_uint)
    for props in regions:
        area = props.area
        if 1 < area <= 30:
            if props.eccentricity < 0.85:
                coords = props.coords
                clean_ma_mask[coords[:, 0], coords[:, 1]] = 1
                
    ma_mask = clean_ma_mask > 0
    
    # 2. Hard Exudates
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L_chan, a_chan, b_chan = cv2.split(lab)
    
    L_chan[od_mask] = 0
    b_chan[od_mask] = 0
    
    L_mask = L_chan > 200
    b_mask = b_chan > 140
    ex_mask = L_mask & b_mask
    
    ex_mask_uint = ex_mask.astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(ex_mask_uint, connectivity=8)
    clean_ex_mask = np.zeros_like(ex_mask_uint)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if 3 < area < 5000:
            clean_ex_mask[labels == i] = 1
    ex_mask = clean_ex_mask > 0
    ex_mask[vessel_mask] = False
    
    return ma_mask, ex_mask

def classify_hemorrhages(img, vessel_mask, od_mask, ma_mask):
    g_chan = img[:,:,1]
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    g_enh = clahe.apply(g_chan)
    g_inv = cv2.bitwise_not(g_enh)
    
    r_chan = img[:,:,2]
    raw_retina_mask = (r_chan > 20).astype(np.uint8)
    kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35))
    retina_mask = cv2.erode(raw_retina_mask, kernel_erode) > 0
    
    all_dark_anomalies = g_inv > 180
    all_dark_anomalies[vessel_mask] = False
    all_dark_anomalies[od_mask] = False
    all_dark_anomalies[ma_mask] = False
    all_dark_anomalies[~retina_mask] = False
    
    all_dark_uint = all_dark_anomalies.astype(np.uint8)
    dot_blot_mask = np.zeros_like(all_dark_uint)
    flame_mask = np.zeros_like(all_dark_uint)
    preretinal_mask = np.zeros_like(all_dark_uint)
    
    labeled_img = label(all_dark_uint)
    regions = regionprops(labeled_img)
    
    for props in regions:
        area = props.area
        ecc = props.eccentricity
        coords = props.coords
        rows = coords[:, 0]
        cols = coords[:, 1]
        
        if 1000 < area < 20000:
            preretinal_mask[rows, cols] = 1
        elif ecc > 0.85 and 15 < area <= 1000:
            flame_mask[rows, cols] = 1
        elif ecc <= 0.85 and 15 < area <= 1000:
            dot_blot_mask[rows, cols] = 1
            
    return dot_blot_mask > 0, flame_mask > 0, preretinal_mask > 0

def process_dataset(dataset_name):
    print(f"\n[{dataset_name.upper()}] Starting processing...")
    input_ds_dir = os.path.join(INPUT_ROOT, dataset_name)
    output_ds_dir = os.path.join(OUTPUT_ROOT, dataset_name)
    
    input_img_dir = os.path.join(input_ds_dir, "images")
    output_img_dir = os.path.join(output_ds_dir, "images")
    
    if not os.path.exists(input_img_dir):
        print(f"Skipping {dataset_name}, no images found.")
        return
        
    os.makedirs(output_img_dir, exist_ok=True)
    
    # Load original labels
    input_csv_path = os.path.join(input_ds_dir, "labels.csv")
    if os.path.exists(input_csv_path):
        df_labels = pd.read_csv(input_csv_path)
    else:
        print(f"Warning: No labels.csv found for {dataset_name}")
        df_labels = None
        
    image_files = glob.glob(os.path.join(input_img_dir, "*.*"))
    features_data = []
    
    print(f"Found {len(image_files)} images.")
    
    # Iterate with progress tracking
    for img_path in tqdm(image_files, desc=f"Processing {dataset_name}"):
        try:
            filename = os.path.basename(img_path)
            img_id, ext = os.path.splitext(filename)
            
            # Setup image folder structure
            target_folder = os.path.join(output_img_dir, img_id)
            if os.path.exists(target_folder):
                continue # Skip if already processed (allows resuming)
                
            os.makedirs(target_folder, exist_ok=True)
            
            img = cv2.imread(img_path)
            if img is None: continue
            
            # Ensure 384x384 (already formatted, but just to be safe)
            if img.shape[:2] != (384, 384):
                img = cv2.resize(img, (384, 384))
                
            # Run Pipeline
            od_mask = extract_landmarks(img)
            vessel_mask = extract_vessels(img)
            ma_mask, ex_mask = detect_lesions(img, vessel_mask, od_mask)
            dot_blot_mask, flame_mask, preretinal_mask = classify_hemorrhages(img, vessel_mask, od_mask, ma_mask)
            
            # Save Raw Image
            shutil.copy(img_path, os.path.join(target_folder, filename))
            
            # Save Masks
            cv2.imwrite(os.path.join(target_folder, f"{img_id}_vessel.png"), vessel_mask.astype(np.uint8)*255)
            cv2.imwrite(os.path.join(target_folder, f"{img_id}_ma.png"), ma_mask.astype(np.uint8)*255)
            cv2.imwrite(os.path.join(target_folder, f"{img_id}_ex.png"), ex_mask.astype(np.uint8)*255)
            
            hemorrhage_mask = dot_blot_mask | flame_mask | preretinal_mask
            cv2.imwrite(os.path.join(target_folder, f"{img_id}_hemorrhage.png"), hemorrhage_mask.astype(np.uint8)*255)
            
            # Calculate Features
            r_chan = img[:,:,2]
            retina_area = np.sum(r_chan > 20)
            if retina_area == 0: retina_area = 1
                
            features_data.append({
                'image_id': img_id,
                'MA_Count': label(ma_mask).max(),
                'Exudate_Pct': round((np.sum(ex_mask) / retina_area) * 100, 4),
                'DotBlot_Pct': round((np.sum(dot_blot_mask) / retina_area) * 100, 4),
                'Flame_Pct': round((np.sum(flame_mask) / retina_area) * 100, 4),
                'PreRetinal_Pct': round((np.sum(preretinal_mask) / retina_area) * 100, 4)
            })
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            continue

    # Process and Merge CSV
    if len(features_data) > 0 and df_labels is not None:
        df_features = pd.DataFrame(features_data)
        
        # We assume the id column in df_labels is the first column
        id_col = df_labels.columns[0]
        
        # Ensure string type for reliable merging
        df_labels[id_col] = df_labels[id_col].astype(str)
        df_features['image_id'] = df_features['image_id'].astype(str)
        
        # Merge
        df_final = pd.merge(df_labels, df_features, left_on=id_col, right_on='image_id', how='left')
        
        # Drop redundant ID column if they have different names
        if id_col != 'image_id':
            df_final = df_final.drop(columns=['image_id'])
            
        output_csv = os.path.join(output_ds_dir, "labels.csv")
        df_final.to_csv(output_csv, index=False)
        print(f"[{dataset_name.upper()}] Features merged into {output_csv}")

if __name__ == "__main__":
    print(f"Starting massive multi-dataset processing from {INPUT_ROOT} to {OUTPUT_ROOT}")
    for ds in DATASETS:
        process_dataset(ds)
    print("ALL PROCESSING COMPLETE.")
