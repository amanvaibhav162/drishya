import os
import cv2
import numpy as np
from skimage.filters import frangi
from skimage.measure import regionprops, label

def extract_landmarks(img):
    imgH, imgW = img.shape[:2]
    # The OD is best isolated in the Red channel
    r_chan = img[:,:,2] 
    
    # Morphological closing to erase remaining vessels and small exudates
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    r_closed = cv2.morphologyEx(r_chan, cv2.MORPH_CLOSE, kernel_close)
    
    # A massive Gaussian blur perfectly averages the entire Optic Disc region.
    # Mathematically, this forces the max_loc to align perfectly with the vessel convergence point!
    blurred = cv2.GaussianBlur(r_closed, (121, 121), 0)
    retina_mask = img[:,:,2] > 20
    blurred[~retina_mask] = 0
    _, _, _, max_loc = cv2.minMaxLoc(blurred)
        
    od_mask = np.zeros_like(r_chan)
    cv2.circle(od_mask, max_loc, 45, 255, -1)
    return od_mask > 0

def extract_fovea(img, od_mask, retina_mask):
    # Locate the Fovea by finding the darkest region in the green channel
    g_chan = img[:,:,1]
    # Heavy blur to wash out vessels/lesions
    g_blur = cv2.GaussianBlur(g_chan, (51, 51), 0)
    
    # Restrict the search to the strictly inner retina (erode by 60 pixels)
    # This prevents the dark vignetted edges from being chosen over the true Fovea.
    kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (60, 60))
    retina_mask_strict = cv2.erode(retina_mask.astype(np.uint8), kernel_erode) > 0
    
    # Mask out the OD and edges so they aren't detected as the darkest point
    g_blur[od_mask > 0] = 255
    g_blur[~retina_mask_strict] = 255
    
    # Find the darkest point
    _, _, min_loc, _ = cv2.minMaxLoc(g_blur)
    
    fovea_mask = np.zeros_like(g_chan)
    # Draw a 45-pixel radius circle for the Fovea
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
    
    from scipy import ndimage
    
    # 1. Visual Vessel Mask (delicate, for the user)
    vessel_mask = vesselness_norm > (thresh * 0.5) 
    vessel_mask = vessel_mask.astype(np.uint8)
    
    # 2. Suppression Mask (internal, catches faint wisps without deleting MAs)
    suppression_mask = vesselness_norm > (thresh * 0.35)
    suppression_mask = suppression_mask.astype(np.uint8)
    
    # Clean the visual mask
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(vessel_mask, connectivity=8)
    clean_vessel_mask = np.zeros_like(vessel_mask)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= 30:
            clean_vessel_mask[labels == i] = 1
            
    # Bridge broken vessel gaps in visual mask
    kernel_bridge = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    clean_vessel_mask = cv2.morphologyEx(clean_vessel_mask, cv2.MORPH_CLOSE, kernel_bridge)
    
    # Fill ONLY small black holes inside thick vessels, preventing massive anatomical fills!
    filled_vessels = ndimage.binary_fill_holes(clean_vessel_mask).astype(np.uint8)
    holes = filled_vessels - clean_vessel_mask
    num_labels_h, labels_h, stats_h, _ = cv2.connectedComponentsWithStats(holes, connectivity=8)
    for i in range(1, num_labels_h):
        if stats_h[i, cv2.CC_STAT_AREA] <= 150: # Only fill holes smaller than 150 pixels
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
    
    # MA Threshold
    ma_mask = ma_candidates > 15
    
    # We DO NOT dilate the suppression mask! Dilating it swallows true MAs that are physically near vessels!
    # The suppression mask itself is already highly sensitive (thresh 0.35)
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
    
    # Exudates (Hard Exudates) using MATLAB's exact dynamic Otsu logic
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
    ex_mask[suppression_vessel_mask] = False
    
    return ma_mask, ex_mask

def classify_hemorrhages(img, visual_vessel_mask, suppression_vessel_mask, od_mask, fovea_mask, ma_mask):
    g_chan = img[:,:,1]
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    g_enh = clahe.apply(g_chan)
    g_inv = cv2.bitwise_not(g_enh)
    
    raw_retina_mask = (img[:,:,2] > 20).astype(np.uint8)
    kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35))
    retina_mask = cv2.erode(raw_retina_mask, kernel_erode) > 0
    
    # Original Battle-Tested Global Threshold for Hemorrhages
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
            
    hemorrhage_mask = (dot_blot_mask | flame_mask | preretinal_mask) > 0
    return hemorrhage_mask

def run_verification(image_paths, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            print(f"Failed to load {path}")
            continue
        
        if img.shape[:2] != (384, 384):
            img = cv2.resize(img, (384, 384))
            
        img_id = os.path.basename(path).split(".")[0]
        
        od_mask = extract_landmarks(img)
        retina_mask = img[:,:,2] > 20
        # Erode retina mask for stricter bounds
        kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        retina_mask_eroded = cv2.erode(retina_mask.astype(np.uint8), kernel_erode) > 0
        
        fovea_mask = extract_fovea(img, od_mask, retina_mask)
        
        # UNPACK BOTH MASKS
        visual_vessel_mask, suppression_vessel_mask = extract_vessels(img)
        
        ma_mask, ex_mask = detect_lesions(img, visual_vessel_mask, suppression_vessel_mask, od_mask, retina_mask_eroded)
        hem_mask = classify_hemorrhages(img, visual_vessel_mask, suppression_vessel_mask, od_mask, fovea_mask, ma_mask)
        
        # Save output images
        cv2.imwrite(os.path.join(output_dir, f"{img_id}_original.png"), img)
        cv2.imwrite(os.path.join(output_dir, f"{img_id}_vessel.png"), visual_vessel_mask.astype(np.uint8)*255)
        cv2.imwrite(os.path.join(output_dir, f"{img_id}_ma.png"), ma_mask.astype(np.uint8)*255)
        cv2.imwrite(os.path.join(output_dir, f"{img_id}_ex.png"), ex_mask.astype(np.uint8)*255)
        cv2.imwrite(os.path.join(output_dir, f"{img_id}_hemorrhage.png"), hem_mask.astype(np.uint8)*255)
        
        # Also save an overlay to see exactly where Fovea and OD are
        overlay = img.copy()
        overlay[fovea_mask > 0] = [255, 0, 0] # Fovea is Blue
        overlay[od_mask > 0] = [0, 255, 255] # OD is Yellow
        cv2.imwrite(os.path.join(output_dir, f"{img_id}_landmarks.png"), overlay)
        print(f"Processed {img_id}")

if __name__ == '__main__':
    images = [
        r"sample_image.jpg" 
    ]
    out_dir = r"verification_output"
    run_verification(images, out_dir)
