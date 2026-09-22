classdef TestEdgeCasesAndStress < matlab.unittest.TestCase
    % TESTEDGECASESANDSTRESS Hostile inputs, camera failures, extreme noise, and dimension robustness.

    methods (Test)
        function testPitchBlackImage(testCase)
            % Camera shutter closed or sensor disconnect
            black_img = zeros(384, 384, 3, 'uint8');
            metrics = assess_quality(black_img);
            [status, feedback, ~, Q] = evaluate_iqa(metrics);
            
            testCase.verifyEqual(status, "UNGRADEABLE");
            testCase.verifyLessThan(Q, 0.25);
            testCase.verifyTrue(contains(feedback, "illumination", 'IgnoreCase', true) || ...
                                 contains(feedback, "FOV", 'IgnoreCase', true) || ...
                                 contains(feedback, "dark", 'IgnoreCase', true));
        end

        function testWhiteSaturatedImage(testCase)
            % Overexposed flash / sensor saturation
            white_img = 255 * ones(384, 384, 3, 'uint8');
            metrics = assess_quality(white_img);
            [status, feedback, ~, Q] = evaluate_iqa(metrics);
            
            testCase.verifyEqual(status, "UNGRADEABLE");
            testCase.verifyLessThan(Q, 0.40);
            testCase.verifyTrue(contains(feedback, "illumination", 'IgnoreCase', true) || ...
                                 contains(feedback, "flash", 'IgnoreCase', true) || ...
                                 contains(feedback, "overexposed", 'IgnoreCase', true));
        end

        function testPureNoiseImage(testCase)
            % Dead camera analog noise (uncorrelated RGB)
            rng(42);
            noise_img = randi([0, 255], [384, 384, 3], 'uint8');
            metrics = assess_quality(noise_img);
            [status, ~, ~, ~] = evaluate_iqa(metrics);
            
            % Pure noise fails clinical retinal adequacy
            testCase.verifyEqual(status, "UNGRADEABLE");
        end

        function testExtremeBlur(testCase)
            % Severe patient defocus / movement
            sz = 384;
            [X, Y] = meshgrid(1:sz, 1:sz);
            R = sqrt((X - sz/2).^2 + (Y - sz/2).^2);
            fundus = zeros(sz, sz, 3, 'uint8');
            retina_mask = R <= (sz/2 - 16);
            fundus(:,:,1) = uint8(retina_mask * 180);
            fundus(:,:,2) = uint8(retina_mask * 90);
            fundus(:,:,3) = uint8(retina_mask * 25);
            
            % Apply severe Gaussian blur
            blurred_fundus = imgaussfilt(fundus, 16);
            metrics = assess_quality(blurred_fundus);
            [status, feedback, ~, ~] = evaluate_iqa(metrics);
            
            testCase.verifyEqual(status, "UNGRADEABLE");
            testCase.verifyTrue(contains(feedback, "blur", 'IgnoreCase', true) || ...
                                 contains(feedback, "focus", 'IgnoreCase', true) || ...
                                 contains(feedback, "sharpness", 'IgnoreCase', true));
        end

        function testNonSquareDimensionsCropPad(testCase)
            % Rectangular camera sensors (e.g. 512x768) with valid retinal cone
            rect_img = zeros(512, 768, 3, 'uint8');
            [X, Y] = meshgrid(1:768, 1:512);
            R = sqrt((X - 384).^2 + (Y - 256).^2);
            cone = R <= 230;
            rect_img(:,:,1) = uint8(cone * 180);
            rect_img(:,:,2) = uint8(cone * 90);
            rect_img(:,:,3) = uint8(cone * 30);
            
            [mask, croppedImg] = extract_retinal_mask(rect_img);
            
            testCase.verifyNotEmpty(croppedImg);
            % Aspect-preserving square crop
            testCase.verifyEqual(size(croppedImg, 1), size(croppedImg, 2));
            testCase.verifyEqual(size(croppedImg, 3), 3);
            testCase.verifyEqual(size(mask, 1), size(croppedImg, 1));
            testCase.verifyEqual(class(croppedImg), 'uint8');
        end

        function testLowContrastAdaptiveEnhancement(testCase)
            % Low contrast underexposed image
            sz = 384;
            [X, Y] = meshgrid(1:sz, 1:sz);
            R = sqrt((X - sz/2).^2 + (Y - sz/2).^2);
            dim_img = zeros(sz, sz, 3, 'uint8');
            retina_mask = R <= (sz/2 - 20);
            dim_img(:,:,1) = uint8(retina_mask * 55);
            dim_img(:,:,2) = uint8(retina_mask * 35);
            dim_img(:,:,3) = uint8(retina_mask * 15);
            
            mask = extract_retinal_mask(dim_img);
            enhanced = adaptive_enhance(dim_img, mask);
            
            % Enhanced image should have higher contrast/standard deviation
            testCase.verifyGreaterThan(std(double(enhanced(:))), std(double(dim_img(:))));
        end

        function testEmptySegmentationSafety(testCase)
            % Blank dark image should run without runtime errors
            blank_img = zeros(384, 384, 3, 'uint8');
            [segResults, overlay] = run_segmentation_pipeline(blank_img);
            
            testCase.verifyTrue(isstruct(segResults));
            testCase.verifyEqual(segResults.NumMA, 0);
            testCase.verifyEqual(segResults.HemorrhageStats.TotalCount, 0);
            testCase.verifyEqual(segResults.ExudateStats.Count, 0);
            testCase.verifyFalse(segResults.HasNVD);
            testCase.verifyFalse(segResults.HasNVE);
            testCase.verifyEqual(size(overlay), [384, 384, 3]);
        end
    end
end
