import os
import cv2
import numpy as np
import argparse
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import time
import glob

def extract_retinal_mask(rgbImg):
    # Use Red channel (channel 2 in BGR) - highest reflectance across pigmentation levels
    r_chan = rgbImg[:, :, 2]
    
    # Adaptive thresholding on Red channel with guard limits
    otsu_val, _ = cv2.threshold(r_chan, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh_val = max(12, min(int(otsu_val * 0.5), 38))
    _, rawMask = cv2.threshold(r_chan, thresh_val, 255, cv2.THRESH_BINARY)
    
    # Clean up with morphological closing and fill holes
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    cleanMask = cv2.morphologyEx(rawMask, cv2.MORPH_CLOSE, kernel_close)
    
    contours, _ = cv2.findContours(cleanMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cv2.drawContours(cleanMask, contours, -1, 255, -1)
        
    numLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(cleanMask, connectivity=8)
    if numLabels > 1:
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        cleanMask = (labels == largest_label).astype(np.uint8) * 255
    else:
        cleanMask = np.zeros_like(cleanMask)
        
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    cleanMask = cv2.morphologyEx(cleanMask, cv2.MORPH_OPEN, kernel_open)
        
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
        mask = padded_mask[y_safe:y_safe+sideLength, x_safe:x_safe+sideLength]
    else:
        croppedImg = rgbImg
        mask = cleanMask
    return mask, croppedImg

def assess_quality(rgbImg, mask):
    metrics = {}
    gChan = rgbImg[:,:,1].astype(np.float32) / 255.0
    validPixels = gChan[mask > 0]
    
    # Focus
    lap = cv2.Laplacian(gChan, cv2.CV_32F)
    metrics['Focus'] = float(np.var(lap[mask > 0])) if len(validPixels) > 0 else 0
    
    # Illum
    meanIllum = float(np.mean(validPixels)) if len(validPixels) > 0 else 0
    metrics['Illumination'] = 1.0 - min(1.0, 2.0 * abs(meanIllum - 0.5))
    metrics['Contrast'] = float(np.std(validPixels)) if len(validPixels) > 0 else 0
    
    # FOV
    area = np.sum(mask > 0)
    if area > 0:
        equivDiameter = np.sqrt(4 * area / np.pi)
        expectedArea = np.pi * (equivDiameter / 2)**2
        metrics['FOV'] = min(1.0, area / expectedArea) if expectedArea > 0 else 0
    else:
        metrics['FOV'] = 0
    return metrics

def evaluate_iqa(metrics, thresholds):
    F_norm = min(1.0, metrics['Focus'] / thresholds['F_target'])
    I_norm = metrics['Illumination']
    V_norm = metrics['FOV']
    C_norm = min(1.0, metrics['Contrast'] / thresholds['C_target'])
    
    weights = [0.35, 0.25, 0.20, 0.20]
    Q = weights[0]*F_norm + weights[1]*I_norm + weights[2]*V_norm + weights[3]*C_norm
    
    # Two-category policy: Reject or Accept & Enhance
    if Q < thresholds['Q_reject'] or V_norm < 0.60:
        return "UNGRADABLE", "REJECT", Q
    else:
        return "ACCEPT_AND_ENHANCE", "STANDARDIZE", Q

def adaptive_enhance(rgbImg, mask, metrics, thresholds):
    img = rgbImg.copy()
    mask_bool = mask > 0
    
    # 1. Decouple into CIE L*a*b* color space
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    
    # 2. Illumination Homogenization strictly on Luminance L*
    l_float = l_channel.astype(np.float32)
    mean_retina_l = np.mean(l_float[mask_bool]) if np.any(mask_bool) else 128.0
    l_filled = l_float.copy()
    l_filled[~mask_bool] = mean_retina_l
    
    # Wide Gaussian blur captures low-frequency illumination manifold
    sigma = max(img.shape[0], img.shape[1]) / 30.0
    bg_l = cv2.GaussianBlur(l_filled, (0, 0), sigma)
    l_flat = l_float - bg_l + mean_retina_l
    l_flat = np.clip(l_flat, 0, 255).astype(np.uint8)
    
    # 3. Controlled Luminance-Only CLAHE
    clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l_flat)
    
    # 4. Calibrated edge-preserving grain suppression on L* only
    # Smooths CMOS sensor grain without blurring microaneurysms or vessel walls
    l_clean = cv2.bilateralFilter(l_clahe, d=5, sigmaColor=15, sigmaSpace=15)
    
    # 5. Recombine with UNTOUCHED chromatic channels a* and b* (Zero color drift)
    lab_out = cv2.merge((l_clean, a_channel, b_channel))
    enhancedRgb = cv2.cvtColor(lab_out, cv2.COLOR_LAB2BGR)
    enhancedRgb[~mask_bool] = 0
    return enhancedRgb

def process_single_image(args):
    src_path, dst_path, thresholds, target_size = args
    try:
        rawImg = cv2.imread(src_path)
        if rawImg is None:
            return f"FAILED: {src_path} (Unreadable)"
            
        # Crop to square first
        mask, croppedImg = extract_retinal_mask(rawImg)
        
        # Standardize on target resolution (512x512)
        croppedImg = cv2.resize(croppedImg, (target_size, target_size), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, (target_size, target_size), interpolation=cv2.INTER_NEAREST)
        mask = (mask > 127).astype(np.uint8) * 255
        
        metrics = assess_quality(croppedImg, mask)
        status, feedback, Q = evaluate_iqa(metrics, thresholds)
        
        if status == "ACCEPT_AND_ENHANCE":
            finalImg = adaptive_enhance(croppedImg, mask, metrics, thresholds)
        else:
            finalImg = None
            
        if finalImg is not None:
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            cv2.imwrite(dst_path, finalImg)
            return f"SUCCESS ({status}): {os.path.basename(src_path)}"
        else:
            return f"REJECTED ({status}): {os.path.basename(src_path)}"
    except Exception as e:
        return f"ERROR ({str(e)}): {src_path}"

def main():
    parser = argparse.ArgumentParser(description="Batch preprocess retinal fundus images at 512x512")
    parser.add_argument("--src", type=str, required=True, help="Source directory of images")
    parser.add_argument("--dst", type=str, required=True, help="Destination directory for processed images")
    parser.add_argument("--size", type=int, default=512, help="Target resolution (default: 512)")
    parser.add_argument("--workers", type=int, default=max(1, multiprocessing.cpu_count() - 1), help="Worker processes")
    args = parser.parse_args()

    thresholds = {
        'F_target': 0.0015,
        'C_target': 0.10,
        'Q_reject': 0.55,
        'Q_good': 0.80
    }

    os.makedirs(args.dst, exist_ok=True)
    exts = ('*.jpg', '*.jpeg', '*.png', '*.tif', '*.tiff')
    all_files = []
    for ext in exts:
        all_files.extend(glob.glob(os.path.join(args.src, ext)))
        all_files.extend(glob.glob(os.path.join(args.src, '**', ext), recursive=True))
    all_files = sorted(list(set(all_files)))

    print(f"Found {len(all_files)} images in {args.src}")
    print(f"Processing with {args.workers} workers to target size {args.size}x{args.size}...")

    task_args = []
    for f in all_files:
        rel_name = os.path.basename(f)
        out_f = os.path.join(args.dst, rel_name)
        task_args.append((f, out_f, thresholds, args.size))

    start_time = time.time()
    success_count = 0
    rejected_count = 0
    error_count = 0

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for i, res in enumerate(executor.map(process_single_image, task_args)):
            if "SUCCESS" in res:
                success_count += 1
            elif "REJECTED" in res:
                rejected_count += 1
            else:
                error_count += 1
            if (i + 1) % 100 == 0 or (i + 1) == len(all_files):
                elapsed = time.time() - start_time
                print(f"[{i+1}/{len(all_files)}] Success: {success_count}, Rejected: {rejected_count}, Errors: {error_count} ({elapsed:.1f}s)")

    print(f"Batch processing finished in {time.time() - start_time:.2f}s")

if __name__ == '__main__':
    main()
