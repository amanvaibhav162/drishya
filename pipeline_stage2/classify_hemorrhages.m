function [dot_blot_mask, flame_mask, preretinal_mask] = classify_hemorrhages(I, vessel_mask, od_mask, ma_mask)
    % CLASSIFY_HEMORRHAGES Detects and classifies hemorrhages into Dot-Blot, Flame, and Pre-retinal.
    % Inputs:
    %   I - RGB Fundus image
    %   vessel_mask - Binary mask of blood vessels
    %   od_mask - Binary mask of the Optic Disc
    %   ma_mask - Binary mask of Microaneurysms (to exclude from hemorrhages)
    % Outputs:
    %   dot_blot_mask - Mask of Dot-Blot hemorrhages
    %   flame_mask - Mask of Flame hemorrhages
    %   preretinal_mask - Mask of Pre-retinal / Vitreous hemorrhages

    [imgH, imgW, ~] = size(I);
    
    % Use green channel for dark lesion extraction
    I_green = I(:,:,2);
    
    % Enhance contrast and invert to make dark lesions bright
    I_enh = adapthisteq(I_green);
    I_inv = imcomplement(I_enh);
    
    % Create a background mask (we only want the retina)
    % Simple thresholding to find the bright circle of the retina
    retina_mask = I(:,:,1) > 20; 
    
    % We want to find dark anomalies that are NOT vessels and NOT OD and NOT MAs
    % Threshold the inverted green channel to get all dark pixels
    thresh = multithresh(I_inv);
    all_dark_anomalies = I_inv > thresh;
    
    % Suppress known structures
    all_dark_anomalies(vessel_mask) = false;
    all_dark_anomalies(od_mask) = false;
    all_dark_anomalies(ma_mask) = false;
    all_dark_anomalies(~retina_mask) = false; % Ignore pixels outside the retina
    
    % Clean up noise (very small pixels should be ignored)
    all_dark_anomalies = bwareaopen(all_dark_anomalies, 15);
    
    % Initialize output masks
    dot_blot_mask = false(imgH, imgW);
    flame_mask = false(imgH, imgW);
    preretinal_mask = false(imgH, imgW);
    
    % Measure geometric properties
    stats = regionprops(all_dark_anomalies, 'PixelIdxList', 'Area', 'Eccentricity');
    
    for i = 1:length(stats)
        area = stats(i).Area;
        ecc = stats(i).Eccentricity; % 0 is a circle, 1 is a line
        
        if area > 1000 % Extremely large areas are pre-retinal or vitreous hemorrhages
            preretinal_mask(stats(i).PixelIdxList) = true;
        elseif ecc > 0.85 % Elongated shape
            flame_mask(stats(i).PixelIdxList) = true;
        elseif ecc <= 0.85 && area > 15 % Circular/compact shape (larger than MAs)
            dot_blot_mask(stats(i).PixelIdxList) = true;
        end
    end
end
