% run_stage2_test.m
% Wrapper script to test the Stage 2 Mathematical Extraction Pipeline

% 1. Setup paths
img_dir = '../data/processed/dr_images';
output_mask_dir = '../data/processed/stage2_test/masks';
csv_output_path = '../data/processed/stage2_test/stage2_features.csv';

if ~exist(output_mask_dir, 'dir')
    mkdir(output_mask_dir);
end

% Get list of image files (test on first 2)
files = dir(fullfile(img_dir, '*.png'));
num_test = min(2, length(files));

% Initialize cell array to store CSV data
% Headers: Image_Name, MA_Count, Exudate_Area, DotBlot_Count, Flame_Count, PreRetinal_Area
csv_data = cell(num_test + 1, 6);
csv_data(1,:) = {'Image_Name', 'MA_Count', 'Exudate_Area', 'DotBlot_Count', 'Flame_Count', 'PreRetinal_Area'};

fprintf('Starting Stage 2 Mathematical Pipeline on %d images...\n', num_test);

for i = 1:num_test
    tic;
    img_name = files(i).name;
    img_path = fullfile(img_dir, img_name);
    
    fprintf('Processing %d/%d: %s... \n', i, num_test, img_name);
    
    % Read image
    I = imread(img_path);
    
    % Ensure it's 384x384 just in case
    I = imresize(I, [384 384]);
    
    % --- Module 1: Landmarks ---
    [od_mask, fovea_loc, od_center] = extract_landmarks(I);
    
    % --- Module 2: Vascular Network ---
    vessel_mask = extract_vessels(I, od_mask);
    
    % --- Module 3: Early Lesions ---
    [ma_mask, ex_mask] = detect_lesions(I, vessel_mask, od_mask);
    
    % --- Module 4: Hemorrhages ---
    [dot_blot_mask, flame_mask, preretinal_mask] = classify_hemorrhages(I, vessel_mask, od_mask, ma_mask);
    
    % --- Calculate Features ---
    % MAs are typically disconnected tiny dots, so we count connected components
    cc_ma = bwconncomp(ma_mask);
    ma_count = cc_ma.NumObjects;
    
    % Exudates: total area is more reliable than count due to clustering
    exudate_area = sum(ex_mask(:));
    
    % Hemorrhages: count individual shapes
    cc_dotblot = bwconncomp(dot_blot_mask);
    dotblot_count = cc_dotblot.NumObjects;
    
    cc_flame = bwconncomp(flame_mask);
    flame_count = cc_flame.NumObjects;
    
    % Pre-retinal hemorrhages: measure total area
    preretinal_area = sum(preretinal_mask(:));
    
    % Store in CSV data
    csv_data(i+1, :) = {img_name, num2str(ma_count), num2str(exudate_area), ...
                        num2str(dotblot_count), num2str(flame_count), num2str(preretinal_area)};
                    
    % --- Save Masks (Multi-Channel Stacking Option) ---
    [~, base_name, ~] = fileparts(img_name);
    imwrite(vessel_mask, fullfile(output_mask_dir, [base_name, '_vessel.png']));
    imwrite(ma_mask, fullfile(output_mask_dir, [base_name, '_ma.png']));
    imwrite(ex_mask, fullfile(output_mask_dir, [base_name, '_ex.png']));
    imwrite(dot_blot_mask | flame_mask | preretinal_mask, fullfile(output_mask_dir, [base_name, '_hemorrhage.png']));
    
    fprintf('Done! (MAs: %d, ExArea: %d)\n', ma_count, exudate_area);
end

% Write to CSV
T = cell2table(csv_data(2:end,:), 'VariableNames', csv_data(1,:));
writetable(T, csv_output_path);

fprintf('Pipeline complete. Features saved to %s\n', csv_output_path);
