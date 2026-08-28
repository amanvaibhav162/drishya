function [vessel_mask] = extract_vessels(I, od_mask)
    % EXTRACT_VESSELS Extracts the vascular network using Frangi filter.
    % Inputs:
    %   I - RGB Fundus image
    %   od_mask - Binary mask of the optic disc (optional, to suppress OD artifacts)
    % Outputs:
    %   vessel_mask - Binary mask of the blood vessels
    
    if nargin < 2
        od_mask = false(size(I, 1), size(I, 2));
    end

    % Use green channel for highest contrast
    I_green = I(:,:,2);
    
    % Enhance contrast using CLAHE
    I_enh = adapthisteq(I_green, 'NumTiles', [8 8], 'ClipLimit', 0.01);
    
    % Invert image (Frangi filter / fibermetric usually expects bright vessels on dark background)
    I_inv = imcomplement(I_enh);
    
    % Apply Frangi filter (fibermetric in MATLAB)
    % 'Thickness' parameter depends on vessel width in pixels
    % For 384x384, vessels are roughly 1 to 8 pixels thick
    vessel_enhanced = fibermetric(I_inv, [1:8], 'StructureSensitivity', 1.0, 'ObjectPolarity', 'bright');
    
    % Binarize using a global threshold (or adaptive)
    % We use an empirically determined threshold, or Otsu's method
    thresh = multithresh(vessel_enhanced);
    vessel_mask = vessel_enhanced > (thresh * 0.5); % Slightly lower threshold to keep thin vessels
    
    % Clean up noise
    vessel_mask = bwareaopen(vessel_mask, 30); % Remove small connected components (noise)
    
    % Suppress vessels detected inside the Optic Disc, if needed
    % Wait, vessels DO exist inside the OD. The user wants the vascular network extracted.
    % But we usually don't want the OD boundary itself to be classified as a vessel.
    % We will keep the vessels inside OD, but we can dilate the OD mask slightly and remove the OD edge.
    % Actually, for MA/Exudate suppression, keeping vessels inside OD is good so we don't mistakenly flag them as lesions.
end
