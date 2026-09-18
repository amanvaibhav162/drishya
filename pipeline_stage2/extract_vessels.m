function [visual_mask, supp_mask, vessel_skeleton] = extract_vessels(I, od_mask, target_res)
    % EXTRACT_VESSELS Extracts the vascular network using Frangi vesselness filter.
    % Inputs:
    %   I          - RGB Fundus image (uint8, default 512x512)
    %   od_mask    - Binary mask of optic disc (optional)
    %   target_res - Target resolution (default: 512)
    % Outputs:
    %   visual_mask     - Clean binary mask of blood vessels for visualization/metrics
    %   supp_mask       - Sensitive binary mask for lesion suppression
    %   vessel_skeleton - 1-pixel wide skeleton of vascular tree

    if nargin < 3
        target_res = 512;
    end
    [imgH, imgW, ~] = size(I);
    if imgH ~= target_res || imgW ~= target_res
        I = imresize(I, [target_res, target_res]);
        [imgH, imgW, ~] = size(I);
    end

    if nargin < 2 || isempty(od_mask)
        od_mask = false(imgH, imgW);
    end

    scale_factor = target_res / 384.0;

    % Use green channel for highest contrast
    I_green = I(:,:,2);

    % CLAHE enhancement
    I_enh = adapthisteq(I_green, 'NumTiles', [8 8], 'ClipLimit', 0.02);

    % Invert: vessels become bright ridges on dark background
    I_inv = imcomplement(I_enh);

    % Frangi filter (fibermetric in MATLAB)
    max_thickness = max(8, round(8 * scale_factor));
    thicknesses = 1:1:max_thickness;
    try
        vessel_enhanced = fibermetric(I_inv, thicknesses, 'StructureSensitivity', 1.0, 'ObjectPolarity', 'bright');
    catch
        % Fallback for environments where fibermetric expects single scale
        vessel_enhanced = fibermetric(I_inv, round(max_thickness / 2));
    end

    % Normalize to 0..255
    v_norm = double(vessel_enhanced);
    v_min = min(v_norm(:));
    v_max = max(v_norm(:));
    if v_max > v_min
        v_norm = (v_norm - v_min) / (v_max - v_min) * 255.0;
    end

    % Otsu thresholding
    try
        thresh = multithresh(uint8(v_norm));
    catch
        thresh = graythresh(uint8(v_norm)) * 255.0;
    end

    % Hysteresis thresholding
    t_high = thresh * 0.45;
    t_low = thresh * 0.18;
    
    strong_edges = v_norm > t_high;
    weak_edges = v_norm > t_low;
    
    % Hysteresis tracking
    visual_raw = bwselect(weak_edges, find(strong_edges), 8);
    
    % Suppression mask: Dilate visual mask (7x7 approx)
    se_supp = strel('disk', 3);
    supp_mask = imdilate(visual_raw, se_supp);

    % Remove small noise components
    min_vessel_area = max(15, round(30 * scale_factor));
    visual_mask = bwareaopen(visual_raw, min_vessel_area);

    % Small bridge closing (3x3 disk) — NO large closing, NO hole filling
    se_bridge = strel('disk', 1);
    visual_mask = imclose(visual_mask, se_bridge);

    % Apply retina FOV mask with 5px erosion to remove circular aperture ring
    retina_raw = I(:,:,1) > 20;
    se_fov = strel('disk', max(1, round(5 * scale_factor)));
    retina_fov = imerode(retina_raw, se_fov);

    visual_mask(~retina_fov) = false;
    supp_mask(~retina_fov) = false;

    % Note: Vessels naturally originate in the OD, so we preserve vessels inside OD

    % Generate vessel skeleton (1-pixel wide)
    try
        vessel_skeleton = bwskel(visual_mask);
    catch
        vessel_skeleton = bwmorph(visual_mask, 'skel', Inf);
    end

    % Vessel width validation: filter out large non-vessel blobs
    if any(visual_mask(:))
        dist_map = bwdist(~visual_mask);
        max_width = round(12 * scale_factor);
        cc = bwconncomp(visual_mask);
        stats = regionprops(cc, 'PixelIdxList');

        for k = 1:cc.NumObjects
            idx = stats(k).PixelIdxList;
            skel_idx = idx(vessel_skeleton(idx));
            if ~isempty(skel_idx)
                mean_w = mean(dist_map(skel_idx));
                if mean_w > max_width
                    visual_mask(idx) = false;
                end
            end
        end

        % Re-skeletonize after width filtering
        try
            vessel_skeleton = bwskel(visual_mask);
        catch
            vessel_skeleton = bwmorph(visual_mask, 'skel', Inf);
        end
    end
end
