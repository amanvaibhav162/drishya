function [ma_mask, ex_mask] = detect_lesions(I, vessel_mask, od_mask)
    % DETECT_LESIONS Extracts Microaneurysms (MAs) and Hard Exudates (EXs).
    % Inputs:
    %   I - RGB Fundus image
    %   vessel_mask - Binary mask of blood vessels
    %   od_mask - Binary mask of the Optic Disc
    % Outputs:
    %   ma_mask - Binary mask of Microaneurysms
    %   ex_mask - Binary mask of Hard Exudates

    [imgH, imgW, ~] = size(I);
    if nargin < 3
        od_mask = false(imgH, imgW);
    end
    if nargin < 2
        vessel_mask = false(imgH, imgW);
    end

    % --- 1. Microaneurysm (MA) Detection ---
    % MAs are dark red spots. We use the green channel where they have high contrast.
    I_green = I(:,:,2);
    
    % Use Bottom-Hat (imbothat) filter to extract small, dark circular anomalies
    % The structuring element 'disk' of radius 3 corresponds to typical MA size at 384x384
    se_ma = strel('disk', 3);
    ma_candidates = imbothat(I_green, se_ma);
    
    % Threshold the Bottom-Hat result to get binary MA candidates
    % (We can use a fixed threshold or a statistical one based on the image)
    % Since MAs are very faint, we use an empirical threshold, e.g. 5-10 out of 255.
    thresh_ma = 8;
    ma_mask = ma_candidates > thresh_ma;
    
    % Suppress false positives: Remove MAs detected inside vessels or Optic Disc
    ma_mask(vessel_mask | od_mask) = false;
    
    % Clean up noise: Remove 1-pixel artifacts
    ma_mask = bwareaopen(ma_mask, 2);

    % --- 2. Hard Exudate (EX) Detection ---
    % Exudates are high-intensity yellow-signature pixels. 
    % Best isolated in L*a*b* color space.
    I_lab = rgb2lab(I);
    L_chan = I_lab(:,:,1);
    b_chan = I_lab(:,:,3); % b* channel represents yellow-blue (positive is yellow)
    
    % Mask out the Optic Disc from the L* and b* channels to prevent false positives
    % because the OD is also bright yellow.
    L_chan(od_mask) = 0;
    b_chan(od_mask) = -128; % set to non-yellow
    
    % Apply Otsu's thresholding (multithresh) on the Luminance channel
    thresh_L = multithresh(L_chan);
    % Since exudates are the brightest lesions, we take the top tier
    L_mask = L_chan > thresh_L;
    
    % Apply Otsu's thresholding on the b* (yellow) channel
    % Sometimes b_chan doesn't have enough variance if there are no exudates,
    % so we need to be careful.
    try
        thresh_b = multithresh(b_chan);
        b_mask = b_chan > thresh_b;
    catch
        % If multithresh fails (e.g., flat channel), use a fixed empirical threshold
        b_mask = b_chan > 20; 
    end
    
    % Combine L* and b* masks: Pixels must be both bright AND yellow
    ex_mask = L_mask & b_mask;
    
    % Morphological cleaning: Remove small noisy pixels
    ex_mask = bwareaopen(ex_mask, 4);
    
    % Remove vessels just in case bright reflexes on vessels were caught
    ex_mask(vessel_mask) = false;
end
