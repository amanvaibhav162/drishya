classdef TestIQA < matlab.unittest.TestCase
    % TESTIQA Unit tests for Image Quality Assessment & Enhancement suite.

    properties
        SampleImage
        RetinalMask
        ConfigParams
    end

    methods (TestMethodSetup)
        function loadAssets(testCase)
            baseDir = fullfile(fileparts(mfilename('fullpath')), '..', '..');
            imgPath = fullfile(baseDir, 'data', 'aptos', 'images', 's00r00000', 's00r00000.jpg');
            testCase.SampleImage = imresize(imread(imgPath), [384, 384]);
            [testCase.RetinalMask, ~] = extract_retinal_mask(testCase.SampleImage);
            testCase.ConfigParams.F_target = 0.0015;
            testCase.ConfigParams.C_target = 0.10;
            testCase.ConfigParams.Q_reject = 0.62;
            testCase.ConfigParams.Q_good   = 0.76;
        end
    end

    methods (Test)
        function testQualityMetricsExtraction(testCase)
            metrics = assess_quality(testCase.SampleImage, testCase.RetinalMask);
            testCase.verifyGreaterThan(metrics.Focus, 0);
            testCase.verifyGreaterThanOrEqual(metrics.Illumination, 0);
            testCase.verifyLessThanOrEqual(metrics.Illumination, 1.0);
            testCase.verifyGreaterThan(metrics.FOV, 0.5);
            testCase.verifyGreaterThan(metrics.Contrast, 0);
        end

        function testBlurryImageRejection(testCase)
            % Convolve with heavy blur kernel
            blurryImg = imgaussfilt(testCase.SampleImage, 12);
            metrics = assess_quality(blurryImg, testCase.RetinalMask);
            [status, feedback, ~, Q] = evaluate_iqa(metrics, testCase.ConfigParams);
            
            testCase.verifyEqual(status, "UNGRADEABLE");
            testCase.verifyTrue(contains(feedback, "blur", 'IgnoreCase', true));
        end

        function testUnderExposedImageFeedback(testCase)
            % Severely underexpose image
            darkImg = testCase.SampleImage * 0.15;
            metrics = assess_quality(darkImg, testCase.RetinalMask);
            [status, feedback, ~, ~] = evaluate_iqa(metrics, testCase.ConfigParams);
            
            testCase.verifyEqual(status, "UNGRADEABLE");
            testCase.verifyTrue(contains(feedback, "illumination", 'IgnoreCase', true));
        end

        function testAdaptiveEnhancementImprovesScore(testCase)
            metrics = assess_quality(testCase.SampleImage, testCase.RetinalMask);
            [~, ~, ~, Q_init] = evaluate_iqa(metrics, testCase.ConfigParams);
            
            enhanced = adaptive_enhance(testCase.SampleImage, testCase.RetinalMask, metrics, testCase.ConfigParams);
            testCase.verifyEqual(size(enhanced), size(testCase.SampleImage));
            
            enhMetrics = assess_quality(enhanced, testCase.RetinalMask);
            [~, ~, ~, Q_enh] = evaluate_iqa(enhMetrics, testCase.ConfigParams);
            testCase.verifyGreaterThanOrEqual(Q_enh, Q_init - 0.05);
        end

        function testAspectPreservingCropping(testCase)
            [mask, cropped] = extract_retinal_mask(testCase.SampleImage);
            testCase.verifyEqual(size(cropped, 1), size(cropped, 2)); % Must be square 1:1
            testCase.verifyTrue(islogical(mask));
        end
    end
end
