import os
import cv2
import numpy as np
import pandas as pd
import glob
import shutil
import warnings
from multiprocessing import Pool, cpu_count
from skimage.filters import frangi
from skimage.measure import regionprops, label
from tqdm import tqdm
from scipy import ndimage

warnings.filterwarnings("ignore")

# Define Paths
INPUT_ROOT = r"../data/formated_data_processed"
OUTPUT_ROOT = r"../data/final_processed"

DATASETS = ["idrid/train", "idrid/test"]

# --- Core AI Pipeline Functions ---
def extract_landmarks(img):
    r_chan = img[:,:,2] 
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    r_closed = cv2.morphologyEx(r_chan, cv2.MORPH_CLOSE, kernel_close)
    blurred = cv2.GaussianBlur(r_closed, (121, 121), 0)
    retina_mask = img[:,:,2] > 20
    blurred[~retina_mask] = 0
    _, _, _, max_loc = cv2.minMaxLoc(blurred)
        
    od_mask = np.zeros_like(r_chan)
    cv2.circle(od_mask, max_loc, 45, 255, -1)
    return od_mask > 0

def extract_fovea(img, od_mask, retina_mask):
    r_chan = img[:,:,2]
    retina_mask = retina_mask > 0
    od_coords = np.where(od_mask > 0)
    if len(od_coords[0]) == 0:
        return np.zeros_like(r_chan, dtype=bool)
    od_center_y, od_center_x = np.mean(od_coords[0]), np.mean(od_coords[1])
    imgW = img.shape[1]
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35))
    r_closed = cv2.morphologyEx(r_chan, cv2.MORPH_CLOSE, kernel_close)
    blurred = cv2.GaussianBlur(r_closed, (61, 61), 0)
    
    search_mask = np.zeros_like(r_chan, dtype=np.uint8)
    if od_center_x > imgW // 2:
        search_mask[:, :int(od_center_x - 30)] = 255
    else:
        search_mask[:, int(od_center_x + 30):] = 255
        
    search_mask[~retina_mask] = 0
    blurred_search = blurred.copy()
    blurred_search[search_mask == 0] = 255
    _, _, min_loc, _ = cv2.minMaxLoc(blurred_search)
    fovea_mask = np.zeros_like(r_chan)
    cv2.circle(fovea_mask, min_loc, 45, 255, -1)
    return fovea_mask > 0

def extract_vessels(img):
    g_chan = img[:,:,1]
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    g_enh = clahe.apply(g_chan)
    g_inv = cv2.bitwise_not(g_enh)
    vesselness = frangi(g_inv, sigmas=range(1, 8, 2), black_ridges=False)
    vesselness_norm = cv2.normalize(vesselness, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    thresh = cv2.threshold(vesselness_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]
    
    vessel_mask = vesselness_norm > (thresh * 0.5) 
    vessel_mask = vessel_mask.astype(np.uint8)
    
    suppression_mask = vesselness_norm > (thresh * 0.35)
    suppression_mask = suppression_mask.astype(np.uint8)
    
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(vessel_mask, connectivity=8)
    clean_vessel_mask = np.zeros_like(vessel_mask)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= 30:
            clean_vessel_mask[labels == i] = 1
            
    kernel_bridge = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    clean_vessel_mask = cv2.morphologyEx(clean_vessel_mask, cv2.MORPH_CLOSE, kernel_bridge)
    
    filled_vessels = ndimage.binary_fill_holes(clean_vessel_mask).astype(np.uint8)
    holes = filled_vessels - clean_vessel_mask
    num_labels_h, labels_h, stats_h, _ = cv2.connectedComponentsWithStats(holes, connectivity=8)
    for i in range(1, num_labels_h):
        if stats_h[i, cv2.CC_STAT_AREA] <= 150:
            clean_vessel_mask[labels_h == i] = 1
            
    raw_retina_mask = (img[:,:,2] > 20).astype(np.uint8)
    kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    retina_mask = cv2.erode(raw_retina_mask, kernel_erode) > 0
    clean_vessel_mask[~retina_mask] = 0
            
    return clean_vessel_mask > 0, suppression_mask > 0

def detect_lesions(img, visual_vessel_mask, suppression_vessel_mask, od_mask, retina_mask_eroded):
    g_chan = img[:,:,1]
    kernel_ma = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    ma_candidates = cv2.morphologyEx(g_chan, cv2.MORPH_BLACKHAT, kernel_ma)
    
    ma_mask = ma_candidates > 15
    ma_mask[suppression_vessel_mask > 0] = False
    ma_mask[od_mask] = False
    ma_mask[~retina_mask_eroded] = False
    
    labeled_ma = label(ma_mask.astype(np.uint8))
    regions = regionprops(labeled_ma)
    clean_ma_mask = np.zeros_like(ma_mask, dtype=np.uint8)
    for props in regions:
        if 1 < props.area <= 30 and props.eccentricity < 0.85:
            coords = props.coords
            clean_ma_mask[coords[:, 0], coords[:, 1]] = 1
    ma_mask = clean_ma_mask > 0
    
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L_chan, a_chan, b_chan = cv2.split(lab)
    L_chan[od_mask] = 0
    b_chan[od_mask] = 128
    
    thresh_L = cv2.threshold(L_chan, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]
    thresh_b = cv2.threshold(b_chan, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]
    
    ex_mask = (L_chan > thresh_L) & (b_chan > thresh_b)
    ex_mask_uint = ex_mask.astype(np.uint8)
    
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(ex_mask_uint, connectivity=8)
    clean_ex_mask = np.zeros_like(ex_mask_uint)
    for i in range(1, num_labels):
        if 3 < stats[i, cv2.CC_STAT_AREA] < 5000:
            clean_ex_mask[labels == i] = 1
    ex_mask = clean_ex_mask > 0
    ex_mask[suppression_vessel_mask > 0] = False
    
    return ma_mask, ex_mask

def classify_hemorrhages(img, visual_vessel_mask, suppression_vessel_mask, od_mask, fovea_mask, ma_mask):
    g_chan = img[:,:,1]
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    g_enh = clahe.apply(g_chan)
    g_inv = cv2.bitwise_not(g_enh)
    
    raw_retina_mask = (img[:,:,2] > 20).astype(np.uint8)
    kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35))
    retina_mask = cv2.erode(raw_retina_mask, kernel_erode) > 0
    
    all_dark_anomalies = g_inv > 180
    
    all_dark_anomalies[suppression_vessel_mask > 0] = False
    all_dark_anomalies[od_mask] = False
    all_dark_anomalies[ma_mask] = False
    all_dark_anomalies[~retina_mask] = False
    all_dark_anomalies[fovea_mask] = False
    
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


# --- Multiprocessing Target Function ---
def process_single_image(args):
    img_path, output_img_dir = args
    filename = os.path.basename(img_path)
    img_id, _ = os.path.splitext(filename)
    target_folder = os.path.join(output_img_dir, img_id)
    
    try:
        # Phase 1: Fast Recovery
        if os.path.exists(target_folder):
            ma_path = os.path.join(target_folder, f"{img_id}_ma.png")
            ex_path = os.path.join(target_folder, f"{img_id}_ex.png")
            hem_path = os.path.join(target_folder, f"{img_id}_hemorrhage.png")
            
            if os.path.exists(ma_path) and os.path.exists(ex_path) and os.path.exists(hem_path):
                img = cv2.imread(img_path)
                if img is None: return None
                
                ma_mask = cv2.imread(ma_path, cv2.IMREAD_GRAYSCALE) > 0
                ex_mask = cv2.imread(ex_path, cv2.IMREAD_GRAYSCALE) > 0
                hem_mask = cv2.imread(hem_path, cv2.IMREAD_GRAYSCALE) > 0
                
                retina_area = np.sum(img[:,:,2] > 20)
                if retina_area == 0: retina_area = 1
                
                return {
                    'image_id': img_id,
                    'MA_Count': label(ma_mask).max(),
                    'Exudate_Pct': round((np.sum(ex_mask) / retina_area) * 100, 4),
                    'DotBlot_Pct': 0.0, 
                    'Flame_Pct': 0.0,
                    'PreRetinal_Pct': round((np.sum(hem_mask) / retina_area) * 100, 4)
                }
                
        # Phase 2: Full Heavy Processing
        os.makedirs(target_folder, exist_ok=True)
        img = cv2.imread(img_path)
        if img is None: return None
        if img.shape[:2] != (384, 384):
            img = cv2.resize(img, (384, 384))
            
        od_mask = extract_landmarks(img)
        retina_mask = (img[:,:,2] > 20).astype(np.uint8)
        fovea_mask = extract_fovea(img, od_mask, retina_mask)
        
        visual_vessel_mask, suppression_vessel_mask = extract_vessels(img)
        
        kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        retina_mask_eroded = cv2.erode(retina_mask, kernel_erode) > 0
        
        ma_mask, ex_mask = detect_lesions(img, visual_vessel_mask, suppression_vessel_mask, od_mask, retina_mask_eroded)
        dot_blot_mask, flame_mask, preretinal_mask = classify_hemorrhages(img, visual_vessel_mask, suppression_vessel_mask, od_mask, fovea_mask, ma_mask)
        
        shutil.copy(img_path, os.path.join(target_folder, filename))
        cv2.imwrite(os.path.join(target_folder, f"{img_id}_vessel.png"), visual_vessel_mask.astype(np.uint8)*255)
        cv2.imwrite(os.path.join(target_folder, f"{img_id}_ma.png"), ma_mask.astype(np.uint8)*255)
        cv2.imwrite(os.path.join(target_folder, f"{img_id}_ex.png"), ex_mask.astype(np.uint8)*255)
        
        hemorrhage_mask = dot_blot_mask | flame_mask | preretinal_mask
        cv2.imwrite(os.path.join(target_folder, f"{img_id}_hemorrhage.png"), hemorrhage_mask.astype(np.uint8)*255)
        
        retina_area = np.sum(img[:,:,2] > 20)
        if retina_area == 0: retina_area = 1
            
        return {
            'image_id': img_id,
            'MA_Count': label(ma_mask).max(),
            'Exudate_Pct': round((np.sum(ex_mask) / retina_area) * 100, 4),
            'DotBlot_Pct': round((np.sum(dot_blot_mask) / retina_area) * 100, 4),
            'Flame_Pct': round((np.sum(flame_mask) / retina_area) * 100, 4),
            'PreRetinal_Pct': round((np.sum(preretinal_mask) / retina_area) * 100, 4)
        }
        
    except Exception as e:
        return None


def process_dataset(dataset_name):
    print(f"\n[{dataset_name.upper()}] Starting 20-Core processing...")
    input_ds_dir = os.path.join(INPUT_ROOT, dataset_name)
    output_ds_dir = os.path.join(OUTPUT_ROOT, dataset_name)
    
    input_img_dir = os.path.join(input_ds_dir, "images")
    output_img_dir = os.path.join(output_ds_dir, "images")
    
    if not os.path.exists(input_img_dir):
        print(f"Skipping {dataset_name}, no images found.")
        return
        
    os.makedirs(output_img_dir, exist_ok=True)
    
    input_csv_path = os.path.join(input_ds_dir, "labels.csv")
    df_labels = None
    if os.path.exists(input_csv_path):
        try:
            df_labels = pd.read_csv(input_csv_path)
        except pd.errors.EmptyDataError:
            print(f"Warning: {input_csv_path} is empty.")
            df_labels = None
        
    image_files = glob.glob(os.path.join(input_img_dir, "*.*"))
    print(f"Found {len(image_files)} images.")
    
    args = [(path, output_img_dir) for path in image_files]
    features_data = []
    
    num_cores = 20
    with Pool(processes=num_cores) as pool:
        for result in tqdm(pool.imap_unordered(process_single_image, args), total=len(args), desc=f"Processing {dataset_name}"):
            if result is not None:
                features_data.append(result)
                
    if len(features_data) > 0:
        df_features = pd.DataFrame(features_data)
        if df_labels is not None:
            id_col = df_labels.columns[0]
            df_labels[id_col] = df_labels[id_col].astype(str)
            df_features['image_id'] = df_features['image_id'].astype(str)
            df_final = pd.merge(df_labels, df_features, left_on=id_col, right_on='image_id', how='left')
            if id_col != 'image_id':
                df_final = df_final.drop(columns=['image_id'])
        else:
            df_final = df_features
            
        output_csv = os.path.join(output_ds_dir, "labels.csv")
        df_final.to_csv(output_csv, index=False)
        print(f"[{dataset_name.upper()}] Features saved into {output_csv}")

if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    
    print(f"Starting 20-Core processing from {INPUT_ROOT} to {OUTPUT_ROOT}")
    for ds in DATASETS:
        process_dataset(ds)
    print("ALL PROCESSING COMPLETE.")
