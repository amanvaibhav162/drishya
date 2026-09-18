function metrics = assess_quality(rgbImg, mask)
    % Work primarily with double precision [0, 1]
    imgD = im2double(rgbImg);
    gChan = imgD(:,:,2);
    validPixels = gChan(mask);
    
    % 1. Focus Metric (Variance of Laplacian on Green channel)
    lapFilter = fspecial('laplacian', 0.2);
    lapResp = imfilter(gChan, lapFilter, 'replicate');
    metrics.Focus = var(lapResp(mask));
    
    % 2. Illumination Metric (Centered at optimal mean ~0.45-0.55)
    if isempty(validPixels)
        meanIllum = 0;
        metrics.Contrast = 0;
    else
        meanIllum = mean(validPixels);
        metrics.Contrast = std(validPixels);
    end
    metrics.Illumination = 1 - min(1, 2 * abs(meanIllum - 0.5));
    
    % 3. Field of View Metric
    props = regionprops(mask, 'Area', 'EquivDiameter');
    if isempty(props)
        metrics.FOV = 0;
    else
        expectedArea = pi * (props.EquivDiameter / 2)^2;
        metrics.FOV = min(1.0, props.Area / expectedArea);
    end
end
