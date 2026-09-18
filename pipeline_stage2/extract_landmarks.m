function [od_mask, fovea_loc, od_center, od_radius, fovea_mask] = extract_landmarks(I, target_res)
    % EXTRACT_LANDMARKS Locates the Optic Disc (OD) and Fovea in a retinal fundus image.
    % Inputs:
    %   I          - RGB Fundus image (uint8, default 512x512)
    %   target_res - Target resolution (default: 512)
    % Outputs:
    %   od_mask    - Binary mask of the optic disc
    %   fovea_loc  - [x, y] coordinates of the fovea
    %   od_center  - [x, y] coordinates of the optic disc center
    %   od_radius  - Estimated OD radius in pixels
    %   fovea_mask - Binary mask of fovea exclusion zone

    if nargin < 2
        target_res = 512;
    end

    [imgH, imgW, ~] = size(I);
    if imgH ~= target_res || imgW ~= target_res
        I = imresize(I, [target_res, target_res]);
        [imgH, imgW, ~] = size(I);
    end

    scale_factor = target_res / 384.0;

    % --- 1. Optic Disc Detection ---
    % Red channel isolates OD best (vessels are dark in red)
    I_red = I(:,:,1);

    % Morphological closing erases remaining dark vessels
    close_r = max(1, round(15 * scale_factor));
    se_close = strel('disk', close_r);
    I_red_closed = imclose(I_red, se_close);

    % Heavy Gaussian blur forces max to align with true disc center
    blur_sigma = max(1, round(20 * scale_factor));
    I_blur = imgaussfilt(double(I_red_closed), blur_sigma);

    % Mask out non-retinal background pixels
    retina_mask = I_red > 20;
    I_blur(~retina_mask) = 0;

    % Find center: maximum intensity of heavily blurred red channel
    [~, max_idx] = max(I_blur(:));
    [od_y, od_x] = ind2sub([imgH, imgW], max_idx);
    od_center = [od_x, od_y];

    % --- Adaptive Radius via Radial Intensity Profiling ---
    r_min = max(5, round(15 * scale_factor));
    r_max = round(60 * scale_factor);
    radii = r_min:2:r_max;
    mean_intensities = zeros(size(radii));
    angles = linspace(0, 2*pi, 60);

    for k = 1:length(radii)
        r = radii(k);
        xs = round(od_x + r * cos(angles));
        ys = round(od_y + r * sin(angles));

        valid = (xs >= 1) & (xs <= imgW) & (ys >= 1) & (ys <= imgH);
        xs = xs(valid);
        ys = ys(valid);

        if isempty(xs)
            mean_intensities(k) = 0;
            continue;
        end

        lin_idx = sub2ind([imgH, imgW], ys, xs);
        vals = double(I_red_closed(lin_idx));
        ret_valid = retina_mask(lin_idx);
        vals = vals(ret_valid);

        if ~isempty(vals)
            mean_intensities(k) = mean(vals);
        else
            mean_intensities(k) = 0;
        end
    end

    % Boundary is where intensity drops sharpest (minimum gradient)
    if length(mean_intensities) >= 3
        grad = gradient(mean_intensities);
        [~, min_g_idx] = min(grad);
        od_radius = radii(min_g_idx);
    else
        od_radius = round(35 * scale_factor);
    end

    % Clamp to plausible clinical bounds
    od_radius = max(round(20 * scale_factor), min(round(55 * scale_factor), od_radius));

    % Construct OD binary mask
    [X, Y] = meshgrid(1:imgW, 1:imgH);
    dist_from_center = sqrt((X - od_center(1)).^2 + (Y - od_center(2)).^2);
    od_mask = dist_from_center <= od_radius;

    % --- 2. Fovea Localization ---
    % Fovea is the darkest region ~2.5 OD diameters away from OD center
    od_diameter = od_radius * 2;
    if od_center(1) < (imgW / 2)
        % Right eye (OD on left side): fovea is to the right
        fovea_search_x = od_center(1) + round(2.5 * od_diameter);
    else
        % Left eye (OD on right side): fovea is to the left
        fovea_search_x = od_center(1) - round(2.5 * od_diameter);
    end
    fovea_search_y = od_center(2);

    search_radius = round(1.0 * od_diameter);
    x_min = max(1, round(fovea_search_x - search_radius));
    x_max = min(imgW, round(fovea_search_x + search_radius));
    y_min = max(1, round(fovea_search_y - search_radius));
    y_max = min(imgH, round(fovea_search_y + search_radius));

    % Use green channel for fovea (darkest retinal pigment center)
    I_green = I(:,:,2);
    I_g_blur = imgaussfilt(double(I_green), max(1, round(5 * scale_factor)));

    roi = I_g_blur(y_min:y_max, x_min:x_max);
    if ~isempty(roi)
        [~, min_idx] = min(roi(:));
        [roi_y, roi_x] = ind2sub(size(roi), min_idx);
        fovea_loc = [x_min + roi_x - 1, y_min + roi_y - 1];
    else
        fovea_loc = [round(imgW / 2), round(imgH / 2)];
    end

    % Fovea exclusion zone mask (~80px radius at 512px)
    fovea_excl_r = max(1, round(60 * scale_factor));
    dist_from_fovea = sqrt((X - fovea_loc(1)).^2 + (Y - fovea_loc(2)).^2);
    fovea_mask = dist_from_fovea <= fovea_excl_r;
end
