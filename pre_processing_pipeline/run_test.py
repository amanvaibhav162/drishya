import cv2
import numpy as np
import os
import glob

def extract_retinal_mask(rgbImg):
    gray = cv2.cvtColor(rgbImg, cv2.COLOR_BGR2GRAY)
    _, rawMask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    cleanMask = cv2.morphologyEx(rawMask, cv2.MORPH_CLOSE, kernel)
    
    # keep largest connected component
    numLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(cleanMask, connectivity=8)
    if numLabels > 1:
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        cleanMask = (labels == largest_label).astype(np.uint8) * 255
    else:
        cleanMask = np.zeros_like(cleanMask)
        
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
    metrics['Illumination'] = 1 - min(1, 2 * abs(meanIllum - 0.5))
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
    F_norm = min(1, metrics['Focus'] / thresholds['F_target'])
    I_norm = metrics['Illumination']
    V_norm = metrics['FOV']
    C_norm = min(1, metrics['Contrast'] / thresholds['C_target'])
    
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
    lab = cv2.cvtColor(rgbImg, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    
    l_float = l_channel.astype(np.float32) / 255.0
    
    if metrics['Illumination'] < 0.75:
        sigma = max(rgbImg.shape[0], rgbImg.shape[1]) / 30.0
        
        # Zero-leakage fix: fill black background with mean retina intensity
        meanRetina = np.mean(l_float[mask > 0])
        l_filled = l_float.copy()
        l_filled[mask == 0] = meanRetina
        
        bg = cv2.GaussianBlur(l_filled, (0, 0), sigma)
        l_float = l_float - bg + meanRetina
        l_float = np.clip(l_float, 0, 1)
        
    if metrics['Contrast'] < thresholds['C_target'] and metrics['Focus'] < thresholds['F_target']:
        l_uint8 = (l_float * 255).astype(np.uint8)
        # Drastically reduced h parameter to avoid erasing capillaries (analogous to DegreeOfSmoothing = 0.002)
        l_uint8 = cv2.fastNlMeansDenoising(l_uint8, None, h=2, templateWindowSize=7, searchWindowSize=21)
        l_float = l_uint8.astype(np.float32) / 255.0
        
    # CLAHE
    l_uint8 = (l_float * 255).astype(np.uint8)
    # Reduced clipLimit to prevent color distortion
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8,8))
    l_clahe = clahe.apply(l_uint8)
    
    # Post-CLAHE Denoising (Bilateral Filter) to remove artificially introduced grain
    l_clahe = cv2.bilateralFilter(l_clahe, d=5, sigmaColor=25, sigmaSpace=25)
    
    # Conditional Sharpening
    if metrics['Focus'] < thresholds['F_target'] and metrics['Focus'] > (thresholds['F_target'] * 0.5):
        blur = cv2.GaussianBlur(l_clahe, (0,0), 1)
        l_clahe = cv2.addWeighted(l_clahe, 1.8, blur, -0.8, 0)
        
    l_channel = l_clahe
    lab_out = cv2.merge((l_channel, a_channel, b_channel))
    enhancedRgb = cv2.cvtColor(lab_out, cv2.COLOR_LAB2BGR)
    
    # Mask out
    enhancedRgb[mask == 0] = 0
    return enhancedRgb

def run_pipeline():
    outDir = '../pipeline_results'
    os.makedirs(outDir, exist_ok=True)
    imgDir = '../dr_images'
    files = glob.glob(os.path.join(imgDir, '*.png'))
    
    configParams = {
        'F_target': 0.0015,
        'C_target': 0.10,
        'Q_reject': 0.76,
        'Q_good': 0.78
    }
    
    reportFile = open(os.path.join(outDir, 'report.txt'), 'w')
    
    for i, imgPath in enumerate(files[:5]):
        imgName = os.path.basename(imgPath)
        rawImg = cv2.imread(imgPath)
        if rawImg is None: continue
        
        # Crop to square first, then resize!
        mask, croppedImg = extract_retinal_mask(rawImg)
        croppedImg = cv2.resize(croppedImg, (384, 384))
        mask = cv2.resize(mask, (384, 384))
        mask = (mask > 127).astype(np.uint8) * 255
        metrics = assess_quality(croppedImg, mask)
        status, feedback, Q = evaluate_iqa(metrics, configParams)
        
        reportFile.write(f"----------------------------------------\n")
        reportFile.write(f"Image: {imgName}\n")
        reportFile.write(f"Status: {status}\n")
        reportFile.write(f"QualityScore: {Q:.4f}\n")
        reportFile.write(f"Feedback: {feedback}\n")
        
        finalImg = None
        if status == "UNGRADABLE":
            pass
        elif status == "BORDERLINE":
            finalImg = adaptive_enhance(croppedImg, mask, metrics, configParams)
            enh_metrics = assess_quality(finalImg, mask)
            enh_status, enh_feedback, enh_Q = evaluate_iqa(enh_metrics, configParams)
            if enh_status == "UNGRADABLE":
                finalImg = None
        else:
            finalImg = croppedImg
            
        if finalImg is not None:
            outPath = os.path.join(outDir, 'processed_' + imgName)
            cv2.imwrite(outPath, finalImg)
            origPath = os.path.join(outDir, 'original_' + imgName)
            cv2.imwrite(origPath, rawImg)
            reportFile.write(f"Saved to: {outPath}\n")
        else:
            reportFile.write("Image rejected.\n")
            
    reportFile.close()

if __name__ == '__main__':
    run_pipeline()
