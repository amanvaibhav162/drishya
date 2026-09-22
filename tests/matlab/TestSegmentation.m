classdef TestSegmentation < matlab.unittest.TestCase
    % TESTSEGMENTATION Unit tests for Anatomical & Lesion Segmentation algorithms.

    properties
        TestScan
    end

    methods (TestMethodSetup)
        function loadScan(testCase)
            baseDir = fullfile(fileparts(mfilename('fullpath')), '..', '..');
            imgPath = fullfile(baseDir, 'data', 'aptos', 'images', 's00r00000', 's00r00000.jpg');
            testCase.TestScan = imresize(imread(imgPath), [384, 384]);
        end
    end

    methods (Test)
        function testLandmarksExtraction(testCase)
            [od_mask, fovea_loc, od_center, od_radius] = extract_landmarks(testCase.TestScan);
            testCase.verifyTrue(islogical(od_mask));
            testCase.verifyGreaterThan(od_radius, 15);
            testCase.verifyLessThan(od_radius, 65);
            testCase.verifyGreaterThanOrEqual(od_center(1), 1);
            testCase.verifyLessThanOrEqual(od_center(1), 384);
            testCase.verifyGreaterThanOrEqual(fovea_loc(1), 1);
            testCase.verifyLessThanOrEqual(fovea_loc(1), 384);
        end

        function testVesselSegmentation(testCase)
            [v_mask, v_prob] = extract_vessels(testCase.TestScan);
            testCase.verifyTrue(islogical(v_mask));
            testCase.verifyEqual(size(v_mask), [384, 384]);
            testCase.verifyGreaterThan(sum(v_mask(:)), 500); % Meaningful vessel tree
            testCase.verifyLessThan(sum(v_mask(:)), 384*384*0.40); % Not saturated
        end

        function testSubPixelMADetection(testCase)
            [od_m, ~, ~, ~] = extract_landmarks(testCase.TestScan);
            [v_m, ~] = extract_vessels(testCase.TestScan, od_m);
            [ma_mask, ma_table, num_ma] = detect_subpixel_ma(testCase.TestScan, v_m, od_m);
            
            testCase.verifyTrue(islogical(ma_mask));
            testCase.verifyEqual(num_ma, height(ma_table));
            if num_ma > 0
                % Verify sub-pixel coordinates are floating point numbers
                testCase.verifyTrue(isa(ma_table.X_SubPixel, 'double'));
                testCase.verifyTrue(isa(ma_table.Y_SubPixel, 'double'));
            end
        end

        function testExudatesSegmentation(testCase)
            [od_m, ~, ~, ~] = extract_landmarks(testCase.TestScan);
            [v_m, ~] = extract_vessels(testCase.TestScan, od_m);
            [hard_m, soft_m, total_m, stats] = segment_exudates(testCase.TestScan, v_m, od_m);
            
            testCase.verifyTrue(islogical(hard_m));
            testCase.verifyTrue(islogical(soft_m));
            testCase.verifyTrue(islogical(total_m));
            testCase.verifyEqual(total_m, hard_m | soft_m);
            testCase.verifyGreaterThanOrEqual(stats.AreaPct, 0);
        end

        function testHemorrhageClassification(testCase)
            [od_m, ~, ~, ~] = extract_landmarks(testCase.TestScan);
            [v_m, ~] = extract_vessels(testCase.TestScan, od_m);
            [dot_m, flame_m, preret_m, total_m, stats] = classify_hemorrhages(testCase.TestScan, v_m, od_m);
            
            testCase.verifyTrue(islogical(dot_m));
            testCase.verifyTrue(islogical(flame_m));
            testCase.verifyTrue(islogical(preret_m));
            % Mutually exclusive
            testCase.verifyFalse(any(dot_m(:) & flame_m(:)));
            testCase.verifyEqual(length(stats.QuadrantCounts), 4);
        end

        function testNeovascularizationOutputs(testCase)
            [od_m, ~, od_c, od_r] = extract_landmarks(testCase.TestScan);
            [v_m, ~] = extract_vessels(testCase.TestScan, od_m);
            [nvd_m, nve_m, has_nvd, has_nve, pdr_score, ~] = ...
                detect_neovascularization(testCase.TestScan, v_m, od_m, od_c, od_r);
            
            testCase.verifyTrue(islogical(nvd_m));
            testCase.verifyTrue(islogical(nve_m));
            testCase.verifyTrue(islogical(has_nvd));
            testCase.verifyTrue(islogical(has_nve));
            testCase.verifyGreaterThanOrEqual(pdr_score, 0);
            testCase.verifyLessThanOrEqual(pdr_score, 1.0);
        end
    end
end
