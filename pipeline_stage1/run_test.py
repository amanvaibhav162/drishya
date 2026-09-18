import cv2
import numpy as np
import os
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
    
    if Q < thresholds['Q_reject'] or V_norm < 0.60:
        status = "UNGRADABLE"
        feedback = f"RECAPTURE: F={F_norm:.2f}, I={I_norm:.2f}, V={V_norm:.2f}"
    elif Q < thresholds['Q_good']:
        status = "BORDERLINE"
        feedback = "ENHANCE"
    else:
        status = "ACCEPTABLE"
        feedback = "DIRECT_PASS"
    return status, feedback, Q

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

def run_pipeline():
    outDir = '../pipeline_results'
    os.makedirs(outDir, exist_ok=True)
    imgDir = '../dr_images'
    files = glob.glob(os.path.join(imgDir, '*.png'))
    
    configParams = {
        'F_target': 0.0015,
        'C_target': 0.10,
        'Q_reject': 0.55,
        'Q_good': 0.80
    }
    
    reportFile = open(os.path.join(outDir, 'report.txt'), 'w')
    
    for i, imgPath in enumerate(files[:5]):
        imgName = os.path.basename(imgPath)
        rawImg = cv2.imread(imgPath)
        if rawImg is None: continue
        
        # Crop to square first, then resize to 512x512
        mask, croppedImg = extract_retinal_mask(rawImg)
        croppedImg = cv2.resize(croppedImg, (512, 512))
        mask = cv2.resize(mask, (512, 512))
        mask = (mask > 127).astype(np.uint8) * 255
        
        metrics = assess_quality(croppedImg, mask)
        status, feedback, Q = evaluate_iqa(metrics, configParams)
        
        print(f"[{i+1}/{len(files[:5])}] {imgName} -> Status: {status}, Q: {Q:.4f}, Focus: {metrics['Focus']:.6f}")
        reportFile.write(f"{imgName}: Status={status}, Q={Q:.4f}, Focus={metrics['Focus']:.6f}, Feedback={feedback}\\n")
        
        if status == "BORDERLINE":
            finalImg = adaptive_enhance(croppedImg, mask, metrics, configParams)
        elif status == "ACCEPTABLE":
            finalImg = cv2.bitwise_and(croppedImg, croppedImg, mask=mask)
        else:
            finalImg = None
            
        if finalImg is not None:
            cv2.imwrite(os.path.join(outDir, f"proc_{imgName}"), finalImg)
            
    reportFile.close()
    print("Pipeline complete. Results written to:", outDir)

if __name__ == '__main__':
    run_pipeline()
