function enhancedRgb = adaptive_enhance(rgbImg, mask, metrics, thresholds)
    % Convert to Lab color space to operate strictly on Luminance (L)
    labImg = rgb2lab(rgbImg);
    L = labImg(:,:,1) / 100; % Normalize L to [0, 1]
    
    % 1. Illumination Homogenization (if illumination is non-uniform)
    if metrics.Illumination < 0.75
        % Ben Graham Zero-Leakage Fix: Fill the black background with the mean
        % retinal intensity before blurring so dark borders don't bleed inward (halos).
        meanRetina = mean(L(mask));
        L_filled = L;
        L_filled(~mask) = meanRetina;
        
        sigma = max(size(rgbImg, 1), size(rgbImg, 2)) / 30;
        bg = imgaussfilt(L_filled, sigma);
        L = L - bg + meanRetina;
        L = min(max(L, 0), 1);
    end
    
    % 2. Mild Denoising (Preserve microaneurysm point-spread functions)
    if metrics.Contrast < thresholds.C_target && metrics.Focus < thresholds.F_target
        % Drastically reduced DegreeOfSmoothing to prevent erasing fine capillaries
        L = imnlmfilt(L, 'DegreeOfSmoothing', 0.002);
    end
    
    % 3. Luminance-Only CLAHE
    % Changed to Uniform distribution and lowered ClipLimit to prevent Optic Disc color distortion
    L = adapthisteq(L, 'ClipLimit', 0.01, 'Distribution', 'uniform');
    
    % Post-CLAHE Denoising (Bilateral Filter) to remove artificially introduced grain
    L = imbilatfilt(L, 0.05, 1.5);
    
    % 4. Conditional Sharpening (Only if borderline blur is detected)
    if metrics.Focus < thresholds.F_target && metrics.Focus > (thresholds.F_target * 0.5)
        L = imsharpen(L, 'Radius', 1, 'Amount', 0.8, 'Threshold', 0.05);
    end
    
    % Reconstruct RGB
    labImg(:,:,1) = L * 100;
    enhancedRgb = lab2rgb(labImg);
    
    % Zero-out masked background
    % Ensure mask is logical and applied across all 3 color channels
    enhancedRgb = enhancedRgb .* cast(repmat(mask, [1, 1, 3]), 'like', enhancedRgb);
end
