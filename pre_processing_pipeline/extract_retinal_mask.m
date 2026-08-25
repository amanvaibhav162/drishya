function [mask, croppedImg] = extract_retinal_mask(rgbImg)
    % Convert image to grayscale for contrast
    gray = rgb2gray(rgbImg);
    
    % Threshold out the black margins (empirical threshold I > 10/255)
    rawMask = gray > (10/255); 
    
    % Clean up small holes/artifacts
    cleanMask = imclose(rawMask, strel('disk', 7));
    mask = bwareafilt(cleanMask, 1);
    
    % Crop to valid bounding box
    stats = regionprops(mask, 'BoundingBox');
    if ~isempty(stats)
        bbox = stats(1).BoundingBox;
        
        % Force the bounding box to be a perfect square
        sideLength = ceil(max(bbox(3), bbox(4))); % Take the longest side
        xCenter = bbox(1) + bbox(3)/2;
        yCenter = bbox(2) + bbox(4)/2;
        
        xMin = round(xCenter - sideLength/2);
        yMin = round(yCenter - sideLength/2);
        xMax = xMin + sideLength - 1;
        yMax = yMin + sideLength - 1;
        
        % Pad image if necessary to prevent out-of-bounds crop
        padTop = max(0, 1 - yMin);
        padLeft = max(0, 1 - xMin);
        padBottom = max(0, yMax - size(rgbImg, 1));
        padRight = max(0, xMax - size(rgbImg, 2));
        
        if padTop > 0 || padLeft > 0 || padBottom > 0 || padRight > 0
            rgbImg = padarray(rgbImg, [padTop, padLeft], 0, 'pre');
            rgbImg = padarray(rgbImg, [padBottom, padRight], 0, 'post');
            mask = padarray(mask, [padTop, padLeft], 0, 'pre');
            mask = padarray(mask, [padBottom, padRight], 0, 'post');
            xMin = xMin + padLeft;
            yMin = yMin + padTop;
            xMax = xMax + padLeft;
            yMax = yMax + padTop;
        end
        
        croppedImg = rgbImg(yMin:yMax, xMin:xMax, :);
        mask = mask(yMin:yMax, xMin:xMax);
    else
        croppedImg = rgbImg;
    end
end
