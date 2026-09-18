function [ma_mask, ex_mask] = detect_lesions(I, supp_vessel_mask, vessel_skeleton, od_mask, target_res)
    % DETECT_LESIONS Extracts Microaneurysms (MAs) and Hard Exudates (EXs).
    % Inputs:
    %   I                  - RGB Fundus image (uint8, default 512x512)
    %   supp_vessel_mask   - Sensitive binary vessel mask for suppression
    %   vessel_skeleton    - 1-pixel wide vessel skeleton
    %   od_mask            - Binary mask of Optic Disc
    %   target_res         - Target resolution (default: 512)
    % Outputs:
    %   ma_mask - Binary mask of Microaneurysms
    %   ex_mask - Binary mask of Hard Exudates

    if nargin < 5
        target_res = 512;
    end
    [imgH, imgW, ~] = size(I);
    if imgH ~= target_res || imgW ~= target_res
        I = imresize(I, [target_res, target_res]);
        [imgH, imgW, ~] = size(I);
    end

    if nargin < 4 || isempty(od_mask)
        od_mask = false(imgH, imgW);
    end
    if nargin < 3 || isempty(vessel_skeleton)
        vessel_skeleton = false(imgH, imgW);
    end
    if nargin < 2 || isempty(supp_vessel_mask)
        supp_vessel_mask = false(imgH, imgW);
    end

    scale_factor = target_res / 384.0;

    % Retina field of view mask
    retina_raw = I(:,:,1) > 20;
    se_ret = strel('disk', max(1, round(15 * scale_factor)));
    retina_mask = imerode(retina_raw, se_ret);

    % --- 1. Microaneurysm (MA) Detection ---
    I_green = I(:,:,2);

    % Bottom-Hat filter with disk kernel calibrated to MA scale
    se_r = max(1, round(3 * scale_factor));
    se_ma = strel('disk', se_r);
    ma_candidates = imbothat(I_green, se_ma);

    % Tighter threshold: 20 out of 255
    ma_raw = ma_candidates > 20;

    % Suppress vessels, OD, and non-retinal margins
    ma_raw(supp_vessel_mask | od_mask | ~retina_mask) = false;

    % Distance transform from vessel skeleton (Gemini refinement)
    if any(vessel_skeleton(:))
        dist_to_skeleton = bwdist(vessel_skeleton);
    else
        dist_to_skeleton = ones(imgH, imgW) * 999;
    end

    % Morphological and intensity validation
    ma_min_area = max(2, round(2 * scale_factor));
    ma_max_area = round(25 * scale_factor);

    cc_ma = bwconncomp(ma_raw);
    props_ma = regionprops(cc_ma, 'Area', 'Eccentricity', 'Centroid', 'PixelIdxList');

    ma_mask = false(imgH, imgW);
    ring_outer = max(2, round(5 * scale_factor));

    for k = 1:cc_ma.NumObjects
        area = props_ma(k).Area;
        ecc = props_ma(k).Eccentricity;
        centroid = round(props_ma(k).Centroid); % [x, y]
        cx = centroid(1);
        cy = centroid(2);
        idx = props_ma(k).PixelIdxList;

        % Area and circularity criteria (MAs are round, ecc < 0.75)
        if area < ma_min_area || area > ma_max_area || ecc >= 0.65
            continue;
        end

        % Skeleton-based proximity check: reject if centroid < 2px from vessel skeleton
        if cx >= 1 && cx <= imgW && cy >= 1 && cy <= imgH
            if dist_to_skeleton(cy, cx) < 4
                continue;
            end
        end

        % Local intensity validation: candidate green channel must be darker than local bg
        [rows, cols] = ind2sub([imgH, imgW], idx);
        r_min = max(1, min(rows) - ring_outer);
        r_max = min(imgH, max(rows) + ring_outer);
        c_min = max(1, min(cols) - ring_outer);
        c_max = min(imgW, max(cols) + ring_outer);

        roi_g = double(I_green(r_min:r_max, c_min:c_max));
        roi_mask = false(size(roi_g));
        for p = 1:length(rows)
            roi_mask(rows(p) - r_min + 1, cols(p) - c_min + 1) = true;
        end

        bg_mask = ~roi_mask & (roi_g > 0);
        if sum(bg_mask(:)) > 5
            cand_mean = mean(roi_g(roi_mask));
            bg_mean = mean(roi_g(bg_mask));
            if cand_mean > bg_mean * 0.88
                continue;
            end
        end

        ma_mask(idx) = true;
    end

    % --- 2. Hard Exudate (EX) Detection ---
    % Convert to L*a*b* color space
    I_lab = rgb2lab(I);
    L_chan = I_lab(:,:,1);
    b_chan = I_lab(:,:,3);

    % Suppress Optic Disc to prevent false positive exudates on bright OD
    L_chan_masked = L_chan;
    b_chan_masked = b_chan;
    L_chan_masked(od_mask) = 0;
    b_chan_masked(od_mask) = -128;

    % Otsu thresholding on L* and b* channels
    try
        thresh_L = multithresh(L_chan_masked);
        thresh_b = multithresh(b_chan_masked);
    catch
        thresh_L = mean(L_chan_masked(retina_mask)) + std(L_chan_masked(retina_mask));
        thresh_b = 10;
    end

    ex_raw = (L_chan_masked > thresh_L) & (b_chan_masked > thresh_b);
    ex_raw(supp_vessel_mask | ~retina_mask) = false;

    % Area filtering
    ex_min_area = max(2, round(3 * scale_factor));
    ex_max_area = round(5000 * scale_factor);

    cc_ex = bwconncomp(ex_raw);
    props_ex = regionprops(cc_ex, 'Area', 'PixelIdxList');

    ex_mask = false(imgH, imgW);
    for k = 1:cc_ex.NumObjects
        area = props_ex(k).Area;
        if area >= ex_min_area && area <= ex_max_area
            ex_mask(props_ex(k).PixelIdxList) = true;
        end
    end
end
