function [finalImg, reportStruct] = run_iqa_enhancement_pipeline(rawImgPath, configParams)
    % Default thresholds if not provided
    if nargin < 2
        configParams.F_target = 0.0015; % Relaxed focus threshold (FIQA standards)
        configParams.C_target = 0.10;   % Relaxed contrast threshold
        configParams.Q_reject = 0.55;   % Truly ungradable/unreadable cutoff
    end
    
    % Read the image
    rawImg = imread(rawImgPath);
    
    % Stage 1: Pre-Masking (Red-channel robust mask + square crop)
    [mask, croppedImg] = extract_retinal_mask(rawImg);
    
    % Resize to 512x512 to preserve fine microaneurysm Nyquist spatial resolution
    croppedImg = imresize(croppedImg, [512, 512]);
    mask = imresize(mask, [512, 512]);
    mask = mask > 0.5; % Ensure it remains logical after resize
    
    % Stage 2: Metric Suite
    metrics = assess_quality(croppedImg, mask);
    
    % Stage 3: Decision Engine (Two categories: UNGRADABLE vs ACCEPT_AND_ENHANCE)
    [status, feedback, weights, Q] = evaluate_iqa(metrics, configParams);
    
    reportStruct.OriginalMetrics = metrics;
    reportStruct.QualityScore = Q;
    reportStruct.Status = status;
    reportStruct.Feedback = feedback;
    
    % Stage 4: Processing based on decision
    if status == "UNGRADABLE"
        finalImg = []; % Reject
        disp('Image is UNGRADABLE. Halting pipeline. ' + feedback);
    else
        % Uniform enhancement across 100% of accepted training images
        finalImg = adaptive_enhance(croppedImg, mask, metrics, configParams);
        enhancedMetrics = assess_quality(finalImg, mask);
        reportStruct.EnhancedMetrics = enhancedMetrics;
        disp('Image ACCEPTED and standardized for downstream training.');
    end
end
