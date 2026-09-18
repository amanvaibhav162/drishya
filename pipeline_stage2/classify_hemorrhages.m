function [dot_blot_mask, flame_mask, preretinal_mask, hemorrhage_mask] = classify_hemorrhages(I, supp_vessel_mask, od_mask, fovea_mask, ma_mask, target_res)
    % CLASSIFY_HEMORRHAGES Detects and classifies hemorrhages into Dot-Blot, Flame, and Pre-retinal.
    % Inputs:
    %   I                - RGB Fundus image (uint8, default 512x512)
    %   supp_vessel_mask - Binary vessel suppression mask
    %   od_mask          - Binary mask of Optic Disc
    %   fovea_mask       - Binary mask of Fovea exclusion zone
    %   ma_mask          - Binary mask of Microaneurysms
    %   target_res       - Target resolution (default: 512)
    % Outputs:
    %   dot_blot_mask   - Mask of Dot-Blot hemorrhages
    %   flame_mask      - Mask of Flame hemorrhages
    %   preretinal_mask - Mask of Pre-retinal / Vitreous hemorrhages
    %   hemorrhage_mask - Union of all hemorrhage masks

    if nargin < 6
        target_res = 512;
    end
    [imgH, imgW, ~] = size(I);
    if imgH ~= target_res || imgW ~= target_res
        I = imresize(I, [target_res, target_res]);
        [imgH, imgW, ~] = size(I);
    end

    if nargin < 5 || isempty(ma_mask), ma_mask = false(imgH, imgW); end
    if nargin < 4 || isempty(fovea_mask), fovea_mask = false(imgH, imgW); end
    if nargin < 3 || isempty(od_mask), od_mask = false(imgH, imgW); end
    if nargin < 2 || isempty(supp_vessel_mask), supp_vessel_mask = false(imgH, imgW); end

    scale_factor = target_res / 384.0;

    % Deep retina mask with margin
    retina_raw = I(:,:,1) > 20;
    se_ret = strel('disk', max(1, round(35 * scale_factor)));
    retina_mask = imerode(retina_raw, se_ret);

    % Enhanced inverted green channel
    I_green = I(:,:,2);
    I_enh = adapthisteq(I_green, 'NumTiles', [8 8], 'ClipLimit', 0.02);
    I_inv = double(imcomplement(I_enh));

    % Adaptive thresholding within retina mask
    retina_vals = I_inv(retina_mask);
    if isempty(retina_vals)
        dot_blot_mask = false(imgH, imgW);
        flame_mask = false(imgH, imgW);
        preretinal_mask = false(imgH, imgW);
        hemorrhage_mask = false(imgH, imgW);
        return;
    end

    mean_val = mean(retina_vals);
    std_val = std(retina_vals);
    adaptive_thresh = min(mean_val + 1.2 * std_val, 190);

    candidates = I_inv > adaptive_thresh;

    % Suppress anatomical landmarks and known structures
    candidates(supp_vessel_mask | od_mask | fovea_mask | ma_mask | ~retina_mask) = false;

    % Green/Red Hemoglobin Attenuation Validation
    % Hemoglobin absorbs green light (~540-577nm) but reflects red light:
    % G_cand / G_bg < 0.85 AND R_cand / R_bg > 0.90
    blur_k = max(1, round(75 * scale_factor));
    G_bg = max(imgaussfilt(double(I(:,:,2)), blur_k), 1.0);
    R_bg = max(imgaussfilt(double(I(:,:,1)), blur_k), 1.0);

    G_chan = double(I(:,:,2));
    R_chan = double(I(:,:,1));

    cc_cand = bwconncomp(candidates);
    cand_props = regionprops(cc_cand, 'PixelIdxList');

    validated_mask = false(imgH, imgW);
    for k = 1:cc_cand.NumObjects
        idx = cand_props(k).PixelIdxList;
        g_cand = mean(G_chan(idx));
        r_cand = mean(R_chan(idx));
        g_bg_local = mean(G_bg(idx));
        r_bg_local = mean(R_bg(idx));

        g_ratio = g_cand / g_bg_local;
        r_ratio = r_cand / r_bg_local;

        % Hemoglobin spectral signature validation (dynamic for massive bleeds)
        area = length(idx);
        if area > 150 * (scale_factor^2)
            g_thresh = 0.95;
            r_thresh = 0.65;
        else
            g_thresh = 0.88;
            r_thresh = 0.85;
        end
        if g_ratio < g_thresh && r_ratio > r_thresh
            validated_mask(idx) = true;
        end
    end

    % Morphological cleaning
    se_clean = strel('disk', 1);
    validated_mask = imopen(validated_mask, se_clean);

    % Size and shape classification
    min_hem_area = max(15, round(25 * scale_factor));
    preretinal_min = max(500, round(1000 * scale_factor));
    preretinal_max = round(20000 * scale_factor);
    flame_max = preretinal_min;

    cc_hem = bwconncomp(validated_mask);
    hem_props = regionprops(cc_hem, 'Area', 'Eccentricity', 'PixelIdxList');

    dot_blot_mask = false(imgH, imgW);
    flame_mask = false(imgH, imgW);
    preretinal_mask = false(imgH, imgW);

    for k = 1:cc_hem.NumObjects
        area = hem_props(k).Area;
        ecc = hem_props(k).Eccentricity;
        idx = hem_props(k).PixelIdxList;

        if area < min_hem_area
            continue;
        end

        if area >= preretinal_min && area <= preretinal_max
            preretinal_mask(idx) = true;
        elseif ecc > 0.85 && area <= flame_max
            flame_mask(idx) = true;
        elseif ecc <= 0.85 && area <= flame_max
            dot_blot_mask(idx) = true;
        end
    end

    hemorrhage_mask = dot_blot_mask | flame_mask | preretinal_mask;
end
