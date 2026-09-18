function enhancedRgb = adaptive_enhance(rgbImg, mask, metrics, thresholds)
    % ADAPTIVE_ENHANCE Final Clinically-Calibrated Luminance-Only Pipeline
    % Preserves 100% color constancy (untouched a*, b* chromatic channels)
    % Eliminates spatial chromatic drift (green/magenta cast) and suppresses CMOS noise floor.

    imgD = im2double(rgbImg);
    maskLog = mask > 0.5;
    
    % --- 1. Decouple into CIE L*a*b* Color Space ---
    labImg = rgb2lab(imgD);
    L = labImg(:,:,1) / 100; % Normalize L* to [0, 1]
    a_chan = labImg(:,:,2);   % Kept strictly untouched (Green-Red chromaticity)
    b_chan = labImg(:,:,3);   % Kept strictly untouched (Blue-Yellow chromaticity)
    
    % --- 2. Luminance-Domain Illumination Homogenization ---
    % Zero-leakage fix: fill background with mean retinal luminance before blurring
    meanRetinaL = mean(L(maskLog));
    L_filled = L;
    L_filled(~maskLog) = meanRetinaL;
    
    % Wide Gaussian blur captures the low-frequency illumination baseline
    sigma = max(size(rgbImg, 1), size(rgbImg, 2)) / 30;
    bgL = imgaussfilt(L_filled, sigma);
    L_flat = L - bgL + meanRetinaL;
    L_flat = min(max(L_flat, 0), 1);
    
    % --- 3. Controlled Luminance CLAHE ---
    % Uniform distribution with 0.01 clip limit prevents optic disc blowout
    L_clahe = adapthisteq(L_flat, 'ClipLimit', 0.01, 'Distribution', 'uniform');
    
    % --- 4. Calibrated Edge-Preserving Grain Suppression ---
    % Lightweight bilateral filter specifically removes CMOS sensor shot noise
    % without blurring capillary walls or faint microaneurysms
    L_clean = imbilatfilt(L_clahe, 0.015, 1.2);
    
    % --- 5. Recombine with UNTOUCHED Chromatic Channels ---
    labImg(:,:,1) = L_clean * 100;
    labImg(:,:,2) = a_chan; % Zero chromatic drift
    labImg(:,:,3) = b_chan; % Zero chromatic drift
    enhancedRgb = lab2rgb(labImg);
    
    % Ensure strict [0, 1] range and apply clean retinal mask
    enhancedRgb = min(max(enhancedRgb, 0), 1);
    enhancedRgb = enhancedRgb .* cast(repmat(maskLog, [1, 1, 3]), 'like', enhancedRgb);
end
