import os
import cv2
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import time

def extract_retinal_mask(rgbImg):
    gray = cv2.cvtColor(rgbImg, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels > 1:
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        cleanMask = (labels == largest_label).astype(np.uint8) * 255
    else:
        cleanMask = mask
        
    x, y, w, h = cv2.boundingRect(cleanMask)
    if w > 0 and h > 0:
        sideLength = int(max(w, h))
        xCenter = x + w/2.0
        yCenter = y + h/2.0
        
        x_new = int(round(xCenter - sideLength/2.0))
        y_new = int(round(yCenter - sideLength/2.0))
        
        pad_left = max(0, -x_new)
        pad_top = max(0, -y_new)
        pad_right = max(0, (x_new + sideLength) - rgbImg.shape[1])
        pad_bottom = max(0, (y_new + sideLength) - rgbImg.shape[0])
        
        padded_img = cv2.copyMakeBorder(rgbImg, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=[0,0,0])
        padded_mask = cv2.copyMakeBorder(cleanMask, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0)
        
        x_safe = x_new + pad_left
        y_safe = y_new + pad_top
        
        croppedImg = padded_img[y_safe:y_safe+sideLength, x_safe:x_safe+sideLength]
        cleanMask = padded_mask[y_safe:y_safe+sideLength, x_safe:x_safe+sideLength]
    else:
        croppedImg = rgbImg
        cleanMask = cleanMask
        
    return cleanMask, croppedImg

def assess_quality(rgbImg, mask):
    greenChannel = rgbImg[:,:,1]
    
    laplacian = cv2.Laplacian(greenChannel, cv2.CV_64F)
    laplacian[mask == 0] = 0
    valid_pixels = laplacian[mask > 0]
    focus_metric = np.var(valid_pixels) / 1000.0 if len(valid_pixels) > 0 else 0
    
    green_valid = greenChannel[mask > 0]
    contrast_metric = np.std(green_valid) / 255.0 if len(green_valid) > 0 else 0
    
    total_area = mask.shape[0] * mask.shape[1]
    fov_area = np.sum(mask > 0)
    fov_metric = fov_area / total_area
    
    mean_val = np.mean(green_valid) if len(green_valid) > 0 else 0
    std_val = np.std(green_valid) if len(green_valid) > 0 else 0
    illum_metric = 1.0 - (std_val / mean_val) if mean_val > 0 else 0
    
    return {
        'Focus': focus_metric,
        'Illumination': illum_metric,
        'FOV': fov_metric,
        'Contrast': contrast_metric
    }

def evaluate_iqa(metrics, thresholds):
    f_weight, i_weight, v_weight, c_weight = 0.35, 0.25, 0.20, 0.20
    
    f_score = min(metrics['Focus'] / thresholds['F_target'], 1.0)
    c_score = min(metrics['Contrast'] / thresholds['C_target'], 1.0)
    
    Q = (f_weight * f_score) + \
        (i_weight * metrics['Illumination']) + \
        (v_weight * metrics['FOV']) + \
        (c_weight * c_score)
        
    if Q < thresholds['Q_reject']:
        return "UNGRADABLE"
    elif Q < thresholds['Q_good']:
        return "BORDERLINE"
    else:
        return "ACCEPTABLE"

def process_single_image(src_path):
    thresholds = {
        'F_target': 0.0015,
        'C_target': 0.10,
        'Q_reject': 0.35,
        'Q_good': 0.55
    }
    try:
        rawImg = cv2.imread(src_path)
        if rawImg is None:
            return "FAILED"
            
        mask, croppedImg = extract_retinal_mask(rawImg)
        croppedImg = cv2.resize(croppedImg, (384, 384))
        mask = cv2.resize(mask, (384, 384))
        mask = (mask > 127).astype(np.uint8) * 255
        
        metrics = assess_quality(croppedImg, mask)
        status = evaluate_iqa(metrics, thresholds)
        return status
    except Exception as e:
        return "FAILED"

def main():
    src_dir = r"E:\data384"
    image_tasks = []
    
    for root, dirs, files in os.walk(src_dir):
        for file in files:
            src_file = os.path.join(root, file)
            ext = os.path.splitext(file)[1].lower()
            if ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']:
                image_tasks.append(src_file)
                
    acceptable_count = 0
    borderline_count = 0
    ungradable_count = 0
    failed_count = 0
    
    with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        for status in executor.map(process_single_image, image_tasks):
            if status == "ACCEPTABLE":
                acceptable_count += 1
            elif status == "BORDERLINE":
                borderline_count += 1
            elif status == "UNGRADABLE":
                ungradable_count += 1
            else:
                failed_count += 1
                
    print(f"Total Images Evaluated: {len(image_tasks)}")
    print(f"ACCEPTABLE (Clinically OK, bypassed enhancement): {acceptable_count}")
    print(f"BORDERLINE (CLAHE & Bilateral run): {borderline_count}")
    print(f"UNGRADABLE (Rejected, poor quality): {ungradable_count}")
    print(f"FAILED (Unreadable format): {failed_count}")

if __name__ == '__main__':
    main()
