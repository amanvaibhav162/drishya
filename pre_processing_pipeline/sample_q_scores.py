import os
import cv2
import numpy as np
import random
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

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

def calculate_q_score(src_path):
    thresholds = {
        'F_target': 0.0015,
        'C_target': 0.10
    }
    try:
        rawImg = cv2.imread(src_path)
        if rawImg is None:
            return None
            
        mask, croppedImg = extract_retinal_mask(rawImg)
        croppedImg = cv2.resize(croppedImg, (384, 384))
        mask = cv2.resize(mask, (384, 384))
        mask = (mask > 127).astype(np.uint8) * 255
        
        metrics = assess_quality(croppedImg, mask)
        
        f_weight, i_weight, v_weight, c_weight = 0.35, 0.25, 0.20, 0.20
        f_score = min(metrics['Focus'] / thresholds['F_target'], 1.0)
        c_score = min(metrics['Contrast'] / thresholds['C_target'], 1.0)
        
        Q = (f_weight * f_score) + \
            (i_weight * metrics['Illumination']) + \
            (v_weight * metrics['FOV']) + \
            (c_weight * c_score)
            
        return Q
    except Exception as e:
        return None

def main():
    src_dir = r"E:\data384"
    image_tasks = []
    
    for root, dirs, files in os.walk(src_dir):
        for file in files:
            src_file = os.path.join(root, file)
            ext = os.path.splitext(file)[1].lower()
            if ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']:
                image_tasks.append(src_file)
                
    # Sample 500 images randomly to get a good statistical distribution
    sample_size = min(500, len(image_tasks))
    sampled_tasks = random.sample(image_tasks, sample_size)
    
    q_scores = []
    
    with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        for q in executor.map(calculate_q_score, sampled_tasks):
            if q is not None:
                q_scores.append(q)
                
    if len(q_scores) > 0:
        q_scores.sort()
        p15 = np.percentile(q_scores, 15)
        p05 = np.percentile(q_scores, 5)
        median = np.percentile(q_scores, 50)
        
        print("=== Statistical Analysis of Q Scores ===")
        print(f"Sample Size: {len(q_scores)} images")
        print(f"Median Q Score: {median:.4f}")
        print(f"15th Percentile Q Score (Recommended Q_good): {p15:.4f}")
        print(f"5th Percentile Q Score (Recommended Q_reject): {p05:.4f}")
        print(f"Minimum Q Score: {min(q_scores):.4f}")
        print(f"Maximum Q Score: {max(q_scores):.4f}")
        
        # Save to a text file to easily read
        with open("q_score_stats.txt", "w") as f:
            f.write(f"{p15:.4f},{p05:.4f}")
    else:
        print("Failed to calculate Q scores.")

if __name__ == '__main__':
    main()
