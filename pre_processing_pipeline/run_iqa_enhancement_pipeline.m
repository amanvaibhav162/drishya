function [finalImg, reportStruct] = run_iqa_enhancement_pipeline(rawImgPath, configParams)
    % Default thresholds if not provided
    if nargin < 2
        configParams.F_target = 0.0015; % Relaxed focus threshold (FIQA standards)
        configParams.C_target = 0.10;   % Relaxed contrast threshold
        configParams.Q_reject = 0.76;   % 5th percentile (Reject)
        configParams.Q_good = 0.78;     % 15th percentile (Enhance)
    end
    
    % Read the image
    rawImg = imread(rawImgPath);
    
    % Stage 1: Pre-Masking (Crop to perfect square FIRST to maintain aspect ratio)
    [mask, croppedImg] = extract_retinal_mask(rawImg);
    
    % Resize to 384x384 as requested for model training
    croppedImg = imresize(croppedImg, [384, 384]);
    mask = imresize(mask, [384, 384]);
    mask = mask > 0.5; % Ensure it remains logical after resize
    
    % Stage 2: Metric Suite
    metrics = assess_quality(croppedImg, mask);
    
    % Stage 3: Decision Engine
    [status, feedback, weights, Q] = evaluate_iqa(metrics, configParams);
    
    reportStruct.OriginalMetrics = metrics;
    reportStruct.QualityScore = Q;
    reportStruct.Status = status;
    reportStruct.Feedback = feedback;
    
    % Stage 4: Processing based on decision
    if status == "UNGRADABLE"
        finalImg = []; % Reject
        disp('Image is UNGRADABLE. Halting pipeline. ' + feedback);
    elseif status == "BORDERLINE"
        disp('Image is BORDERLINE. Applying enhancement...');
        finalImg = adaptive_enhance(croppedImg, mask, metrics, configParams);
        
        % Re-evaluate enhanced image
        enhancedMetrics = assess_quality(finalImg, mask);
        [enhStatus, enhFeedback, ~, enhQ] = evaluate_iqa(enhancedMetrics, configParams);
        
        reportStruct.EnhancedMetrics = enhancedMetrics;
        reportStruct.EnhancedQualityScore = enhQ;
        reportStruct.EnhancedStatus = enhStatus;
        
        if enhStatus == "UNGRADABLE"
            finalImg = [];
            disp('Image failed verification after enhancement. Rejecting. ' + enhFeedback);
        else
            disp('Image successfully enhanced and passed to downstream.');
        end
    else
        disp('Image is ACCEPTABLE. Passing to downstream.');
        finalImg = croppedImg; % Direct Pass
    end
end
