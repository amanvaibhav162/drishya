import os
import cv2
import numpy as np
import shutil
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
        return "UNGRADABLE", "REJECT", Q
    elif Q < thresholds['Q_good']:
        return "BORDERLINE", "ENHANCE", Q
    else:
        return "ACCEPTABLE", "DIRECT_PASS", Q

def process_single_image(args):
    src_path, dst_path, thresholds = args
    try:
        rawImg = cv2.imread(src_path)
        if rawImg is None:
            return f"FAILED: {src_path} (Unreadable)"
            
        # Crop to square first
        mask, croppedImg = extract_retinal_mask(rawImg)
        
        # Resize to 384x384
        croppedImg = cv2.resize(croppedImg, (384, 384))
        mask = cv2.resize(mask, (384, 384))
        mask = (mask > 127).astype(np.uint8) * 255
        
        metrics = assess_quality(croppedImg, mask)
        status, feedback, Q = evaluate_iqa(metrics, thresholds)
        

        finalImg = croppedImg.copy()
        
        if status == "BORDERLINE":
            labImg = cv2.cvtColor(croppedImg, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(labImg)
            l_float = l.astype(np.float32) / 255.0
            
            if metrics['Illumination'] < 0.75:
                sigma = max(croppedImg.shape[0], croppedImg.shape[1]) / 30.0
                meanRetina = np.mean(l_float[mask > 0])
                l_filled = l_float.copy()
                l_filled[mask == 0] = meanRetina
                bg = cv2.GaussianBlur(l_filled, (0, 0), sigma)
                l_float = l_float - bg + meanRetina
                l_float = np.clip(l_float, 0, 1)
                
            l_uint8 = (l_float * 255).astype(np.uint8)
            clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8,8))
            l_clahe = clahe.apply(l_uint8)
            l_clahe = cv2.bilateralFilter(l_clahe, d=5, sigmaColor=25, sigmaSpace=25)
            
            if metrics['Focus'] < thresholds['F_target'] and metrics['Focus'] > (thresholds['F_target'] * 0.5):
                blur = cv2.GaussianBlur(l_clahe, (0,0), 1)
                l_clahe = cv2.addWeighted(l_clahe, 1.5, blur, -0.5, 0)
                
            labImg = cv2.merge((l_clahe, a, b))
            enhancedRgb = cv2.cvtColor(labImg, cv2.COLOR_LAB2BGR)
            
            # Apply mask to all channels
            finalImg = cv2.bitwise_and(enhancedRgb, enhancedRgb, mask=mask)
        else:
            finalImg = cv2.bitwise_and(finalImg, finalImg, mask=mask)
            
        cv2.imwrite(dst_path, finalImg)
        return f"SUCCESS ({status}): {src_path}"
    except Exception as e:
        return f"ERROR ({str(e)}): {src_path}"

def main():
    src_dir = r"E:\data384"
    dst_dir = r"E:\data384_processed"
    
    thresholds = {
        'F_target': 0.0015,
        'C_target': 0.10,
        'Q_reject': 0.76,
        'Q_good': 0.78
    }
    
    print(f"Starting batch process from {src_dir} to {dst_dir}")
    
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
        
    image_tasks = []
    
    # Walk the directory
    for root, dirs, files in os.walk(src_dir):
        # Create corresponding directories in dst
        rel_path = os.path.relpath(root, src_dir)
        target_dir = os.path.join(dst_dir, rel_path) if rel_path != "." else dst_dir
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
            
        for file in files:
            src_file = os.path.join(root, file)
            dst_file = os.path.join(target_dir, file)
            
            ext = os.path.splitext(file)[1].lower()
            if ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']:
                image_tasks.append((src_file, dst_file, thresholds))
            else:
                # Copy non-image files directly
                shutil.copy2(src_file, dst_file)
                
    print(f"Found {len(image_tasks)} images to process. Using {multiprocessing.cpu_count()} cores.")
    
    start_time = time.time()
    
    # Process images in parallel
    completed = 0
    with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        for result in executor.map(process_single_image, image_tasks):
            completed += 1
            if completed % 100 == 0 or completed == len(image_tasks):
                print(f"Progress: {completed} / {len(image_tasks)} processed. Time elapsed: {time.time() - start_time:.1f}s")
                
    print(f"Batch processing complete! Total time: {time.time() - start_time:.1f}s")

if __name__ == '__main__':
    main()
