import os
import glob
import cv2
import numpy as np
import time
import pandas as pd
from skimage.filters import frangi, threshold_otsu
from skimage.measure import regionprops, label

# Setup paths
IMG_DIR = r"../data/processed/dr_images"
OUTPUT_MASK_DIR = r"../data/processed/stage2_test_python/masks"
CSV_OUTPUT_PATH = r"../data/processed/stage2_test_python/stage2_features_python.csv"

os.makedirs(OUTPUT_MASK_DIR, exist_ok=True)

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
    
    od_x, od_y = od_center
    od_diameter = od_radius * 2
    
    if od_x < (imgW / 2):
        fovea_search_x = int(od_x + 2.5 * od_diameter) 
    else:
        fovea_search_x = int(od_x - 2.5 * od_diameter) 
        
    fovea_search_y = od_y
    search_radius = int(1.0 * od_diameter)
    
    x_min = max(0, fovea_search_x - search_radius)
    x_max = min(imgW, fovea_search_x + search_radius)
    y_min = max(0, fovea_search_y - search_radius)
    y_max = min(imgH, fovea_search_y + search_radius)
    
    g_chan = img[:,:,1]
    g_blur = cv2.GaussianBlur(g_chan, (11, 11), 0)
    
    roi = g_blur[y_min:y_max, x_min:x_max]
    
    if roi.size > 0:
        _, _, min_loc, _ = cv2.minMaxLoc(roi)
        fovea_loc = (x_min + min_loc[0], y_min + min_loc[1])
    else:
        fovea_loc = (imgW//2, imgH//2)
        
    return od_mask > 0, fovea_loc

def extract_vessels(img, od_mask):
    g_chan = img[:,:,1]
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    g_enh = clahe.apply(g_chan)
    
    g_inv = cv2.bitwise_not(g_enh)
    
    vesselness = frangi(g_inv, sigmas=range(1, 8, 2), black_ridges=False)
    vesselness_norm = cv2.normalize(vesselness, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    thresh = threshold_otsu(vesselness_norm)
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
    
    # Hard threshold (tighter)
    ma_mask = ma_candidates > 15
    
    ma_mask[vessel_mask] = False
    ma_mask[od_mask] = False
    
    ma_mask_uint = ma_mask.astype(np.uint8)
    labeled_ma = label(ma_mask_uint)
    regions = regionprops(labeled_ma)
    
    clean_ma_mask = np.zeros_like(ma_mask_uint)
    for props in regions:
        area = props.area
        # Only regions with area >= 2 have a well-defined eccentricity
        if 1 < area <= 30:
            ecc = props.eccentricity
            # MAs are round dots. Vessel fragments are lines (ecc > 0.85)
            if ecc < 0.85:
                coords = props.coords
                clean_ma_mask[coords[:, 0], coords[:, 1]] = 1
                
    ma_mask = clean_ma_mask > 0
    
    # 2. Hard Exudates
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L_chan, a_chan, b_chan = cv2.split(lab)
    
    L_chan[od_mask] = 0
    b_chan[od_mask] = 0
    
    # Rigid Hard Thresholds (No Otsu)
    # L_chan is 0-255, extremely bright exudates > 200
    # b_chan is 0-255, 128 is neutral, definitively yellow is > 140
    L_mask = L_chan > 200
    b_mask = b_chan > 140
    
    ex_mask = L_mask & b_mask
    
    ex_mask_uint = ex_mask.astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(ex_mask_uint, connectivity=8)
    clean_ex_mask = np.zeros_like(ex_mask_uint)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        # Size filter: > 3 (noise), < 5000 (massive glare removal)
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
    
    # Erode retina mask aggressively to avoid dark edge vignetting
    kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35))
    retina_mask = cv2.erode(raw_retina_mask, kernel_erode) > 0
    
    # Hard threshold instead of Otsu. 
    # Since it's CLAHE enhanced, dark lesions in the original will be very bright in g_inv.
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
        
        # Upper bound area checks to prevent massive artifacts
        if 1000 < area < 20000:
            preretinal_mask[rows, cols] = 1
        elif ecc > 0.85 and 15 < area <= 1000:
            flame_mask[rows, cols] = 1
        elif ecc <= 0.85 and 15 < area <= 1000:
            dot_blot_mask[rows, cols] = 1
            
    return dot_blot_mask > 0, flame_mask > 0, preretinal_mask > 0

def main():
    files = glob.glob(os.path.join(IMG_DIR, "*.png"))
    test_files = files[:20]
    
    csv_data = []
    
    print(f"Starting Stage 2 Python Pipeline on {len(test_files)} images...")
    
    for i, file_path in enumerate(test_files):
        start_time = time.time()
        img_name = os.path.basename(file_path)
        print(f"Processing {i+1}/{len(test_files)}: {img_name}... ", end="", flush=True)
        
        img = cv2.imread(file_path)
        img = cv2.resize(img, (384, 384))
        
        od_mask, fovea_loc = extract_landmarks(img)
        vessel_mask = extract_vessels(img, od_mask)
        ma_mask, ex_mask = detect_lesions(img, vessel_mask, od_mask)
        dot_blot_mask, flame_mask, preretinal_mask = classify_hemorrhages(img, vessel_mask, od_mask, ma_mask)
        
        # Retina mask for percentage calculation
        r_chan = img[:,:,2]
        retina_mask = r_chan > 20
        retina_area = np.sum(retina_mask)
        if retina_area == 0:
            retina_area = 1
            
        # Hard Counts & Areas
        ma_count = label(ma_mask).max() # Hard count for MAs
        
        exudate_area = np.sum(ex_mask)
        dot_blot_area = np.sum(dot_blot_mask)
        flame_area = np.sum(flame_mask)
        preretinal_area = np.sum(preretinal_mask)
        
        # Percentages
        ex_pct = (exudate_area / retina_area) * 100
        dotblot_pct = (dot_blot_area / retina_area) * 100
        flame_pct = (flame_area / retina_area) * 100
        preretinal_pct = (preretinal_area / retina_area) * 100
        
        csv_data.append({
            'Image_Name': img_name,
            'MA_Count': ma_count,
            'Exudate_Pct': round(ex_pct, 4),
            'DotBlot_Pct': round(dotblot_pct, 4),
            'Flame_Pct': round(flame_pct, 4),
            'PreRetinal_Pct': round(preretinal_pct, 4)
        })
        
        # Save Masks
        base_name = os.path.splitext(img_name)[0]
        cv2.imwrite(os.path.join(OUTPUT_MASK_DIR, f"{base_name}_vessel.png"), vessel_mask.astype(np.uint8)*255)
        cv2.imwrite(os.path.join(OUTPUT_MASK_DIR, f"{base_name}_ma.png"), ma_mask.astype(np.uint8)*255)
        cv2.imwrite(os.path.join(OUTPUT_MASK_DIR, f"{base_name}_ex.png"), ex_mask.astype(np.uint8)*255)
        
        hemorrhage_mask = dot_blot_mask | flame_mask | preretinal_mask
        cv2.imwrite(os.path.join(OUTPUT_MASK_DIR, f"{base_name}_hemorrhage.png"), hemorrhage_mask.astype(np.uint8)*255)
        
        elapsed = time.time() - start_time
        print(f"Done! ({elapsed:.2f}s) MA Count: {ma_count}, Ex%: {ex_pct:.2f}%")

    df = pd.DataFrame(csv_data)
    df.to_csv(CSV_OUTPUT_PATH, index=False)
    print(f"\nPipeline complete. Features saved to {CSV_OUTPUT_PATH}")

if __name__ == "__main__":
    main()
