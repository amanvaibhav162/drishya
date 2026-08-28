function [od_mask, fovea_loc, od_center] = extract_landmarks(I)
    % EXTRACT_LANDMARKS Locates the Optic Disc (OD) and Fovea in a retinal image.
    % Inputs:
    %   I - RGB Fundus image (assumed to be 384x384 or similar)
    % Outputs:
    %   od_mask - Binary mask of the optic disc
    %   fovea_loc - [x, y] coordinates of the fovea
    %   od_center - [x, y] coordinates of the optic disc

    [imgH, imgW, ~] = size(I);
    
    % --- 1. Optic Disc Detection ---
    % Use the Red channel to avoid vessels as much as possible
    I_red = I(:,:,1);
    
    % Morphological closing to erase remaining dark vessels
    se = strel('disk', 15);
    I_red_closed = imclose(I_red, se);
    
    % Find circles using Circular Hough Transform
    % For a 384x384 image, OD radius is typically between 20 and 45 pixels
    Rmin = 20;
    Rmax = 45;
    [centers, radii, metric] = imfindcircles(I_red_closed, [Rmin, Rmax], 'ObjectPolarity', 'bright', 'Sensitivity', 0.90, 'EdgeThreshold', 0.1);
    
    od_mask = false(imgH, imgW);
    od_center = [NaN, NaN];
    od_radius = 0;
    
    if ~isempty(centers)
        % Take the strongest circle
        od_center = centers(1, :);
        od_radius = radii(1);
        
        % Create the mask
        [X, Y] = meshgrid(1:imgW, 1:imgH);
        dist_from_center = sqrt((X - od_center(1)).^2 + (Y - od_center(2)).^2);
        od_mask(dist_from_center <= od_radius) = true;
    else
        % Fallback: Find the brightest spot if imfindcircles fails
        % Apply strong Gaussian blur
        I_blur = imgaussfilt(double(I_red_closed), 10);
        [~, max_idx] = max(I_blur(:));
        [y, x] = ind2sub(size(I_blur), max_idx);
        od_center = [x, y];
        od_radius = 30; % Default fallback radius
        
        [X, Y] = meshgrid(1:imgW, 1:imgH);
        dist_from_center = sqrt((X - od_center(1)).^2 + (Y - od_center(2)).^2);
        od_mask(dist_from_center <= od_radius) = true;
    end
    
    % --- 2. Fovea Localization ---
    % Fovea is the darkest region ~2.5 OD diameters away from OD center.
    % Laterality check:
    od_center_x = od_center(1);
    od_center_y = od_center(2);
    od_diameter = od_radius * 2;
    
    if od_center_x < (imgW / 2)
        % Right Eye (OD is on the left side of image)
        % Fovea is to the RIGHT (+x)
        fovea_search_x = od_center_x + round(2.5 * od_diameter);
    else
        % Left Eye (OD is on the right side of image)
        % Fovea is to the LEFT (-x)
        fovea_search_x = od_center_x - round(2.5 * od_diameter);
    end
    
    fovea_search_y = od_center_y; % Fovea is roughly on the same horizontal plane
    
    % Ensure search window is within bounds
    search_radius = round(1.0 * od_diameter);
    x_min = max(1, round(fovea_search_x - search_radius));
    x_max = min(imgW, round(fovea_search_x + search_radius));
    y_min = max(1, round(fovea_search_y - search_radius));
    y_max = min(imgH, round(fovea_search_y + search_radius));
    
    % Use green channel for fovea (darkest spot)
    I_green = I(:,:,2);
    
    % Apply slight blur to green channel to avoid local noise minimums
    I_g_blur = imgaussfilt(double(I_green), 5);
    
    % Extract ROI
    roi = I_g_blur(y_min:y_max, x_min:x_max);
    
    % Find minimum in ROI
    [~, min_idx] = min(roi(:));
    [roi_y, roi_x] = ind2sub(size(roi), min_idx);
    
    % Global fovea coordinates
    fovea_loc = [x_min + roi_x - 1, y_min + roi_y - 1];
end
