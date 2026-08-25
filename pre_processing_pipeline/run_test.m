% Create output folder
outDir = '..\pipeline_results';
if ~exist(outDir, 'dir')
    mkdir(outDir);
end

% Get list of PNG files in dr_images
imgDir = '..\dr_images';
files = dir(fullfile(imgDir, '*.png'));

% Open a file to write reports
reportFile = fopen(fullfile(outDir, 'report.txt'), 'w');

% Process first 10
numToProcess = min(10, length(files));
for i = 1:numToProcess
    imgName = files(i).name;
    imgPath = fullfile(imgDir, imgName);
    
    fprintf('Processing %s...\n', imgName);
    fprintf(reportFile, '----------------------------------------\n');
    fprintf(reportFile, 'Image: %s\n', imgName);
    
    try
        % Run pipeline
        configParams.F_target = 0.005;
        configParams.C_target = 0.15;
        configParams.Q_reject = 0.40;
        configParams.Q_good = 0.70;
        
        [finalImg, reportStruct] = run_iqa_enhancement_pipeline(imgPath, configParams);
        
        % Write report
        fprintf(reportFile, 'Status: %s\n', reportStruct.Status);
        fprintf(reportFile, 'QualityScore: %.4f\n', reportStruct.QualityScore);
        fprintf(reportFile, 'Feedback: %s\n', reportStruct.Feedback);
        
        if isfield(reportStruct, 'EnhancedStatus')
            fprintf(reportFile, 'EnhancedStatus: %s\n', reportStruct.EnhancedStatus);
            fprintf(reportFile, 'EnhancedQualityScore: %.4f\n', reportStruct.EnhancedQualityScore);
        end
        
        % Save final image if it wasn't rejected
        if ~isempty(finalImg)
            outPath = fullfile(outDir, ['processed_' imgName]);
            imwrite(finalImg, outPath);
            fprintf(reportFile, 'Saved to: %s\n', outPath);
        else
            fprintf(reportFile, 'Image rejected.\n');
        end
    catch ME
        fprintf(reportFile, 'Error: %s\n', ME.message);
    end
end
fclose(reportFile);
disp('Test completed.');
