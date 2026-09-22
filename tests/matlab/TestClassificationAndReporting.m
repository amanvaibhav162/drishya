classdef TestClassificationAndReporting < matlab.unittest.TestCase
    % TESTCLASSIFICATIONANDREPORTING Unit tests for ICDR grading, Grad-CAM, and reporting.

    properties
        ScanImage
        SegResults
    end

    methods (TestMethodSetup)
        function setupData(testCase)
            baseDir = fullfile(fileparts(mfilename('fullpath')), '..', '..');
            imgPath = fullfile(baseDir, 'data', 'aptos', 'images', 's00r00000', 's00r00000.jpg');
            testCase.ScanImage = imresize(imread(imgPath), [384, 384]);
            testCase.SegResults = run_segmentation_pipeline(testCase.ScanImage);
        end
    end

    methods (Test)
        function testPredictGradeRange(testCase)
            diag = predict_dr_grade(testCase.ScanImage, testCase.SegResults);
            testCase.verifyGreaterThanOrEqual(diag.Grade, 0);
            testCase.verifyLessThanOrEqual(diag.Grade, 4);
            testCase.verifyTrue(islogical(diag.IsReferable));
            testCase.verifyGreaterThanOrEqual(diag.Confidence, 0);
            testCase.verifyLessThanOrEqual(diag.Confidence, 100);
        end

        function testCalibrateConfidence(testCase)
            dummyLogits = [2.5, 0.1, -0.5, -1.2, -2.0];
            [probs, conf, ci] = calibrate_confidence(dummyLogits, 1.15);
            
            testCase.verifyEqual(length(probs), 5);
            testCase.verifyEqual(sum(probs), 1.0, 'RelTol', 1e-4);
            testCase.verifyGreaterThanOrEqual(ci(1), 0);
            testCase.verifyLessThanOrEqual(ci(2), 100);
            testCase.verifyGreaterThanOrEqual(ci(2), ci(1));
        end

        function testGradCAMGeneration(testCase)
            diag = predict_dr_grade(testCase.ScanImage, testCase.SegResults);
            [overlay, heat] = explain_gradcam(testCase.ScanImage, testCase.SegResults, diag);
            
            testCase.verifyEqual(size(overlay), size(testCase.ScanImage));
            testCase.verifyEqual(size(heat), [384, 384]);
            testCase.verifyGreaterThanOrEqual(min(heat(:)), 0);
            testCase.verifyLessThanOrEqual(max(heat(:)), 1.0);
        end

        function testReportGeneration(testCase)
            diag = predict_dr_grade(testCase.ScanImage, testCase.SegResults);
            [cam_over, ~] = explain_gradcam(testCase.ScanImage, testCase.SegResults, diag);
            iqaRep.QualityScore = 0.85;
            iqaRep.FinalStatus = "PASS";
            iqaRep.OriginalMetrics.Focus = 0.0020;
            iqaRep.OriginalMetrics.Illumination = 0.85;
            iqaRep.OriginalMetrics.FOV = 0.95;

            tempDir = fullfile(tempdir, 'drishya_test_reports');
            if ~exist(tempDir, 'dir'), mkdir(tempDir); end
            
            repPath = generate_clinical_report([], testCase.ScanImage, testCase.ScanImage, ...
                cam_over, testCase.ScanImage, iqaRep, diag, testCase.SegResults, tempDir);
            
            testCase.verifyTrue(exist(repPath, 'file') == 2);
            info = dir(repPath);
            testCase.verifyGreaterThan(info.bytes, 50000); % Non-empty report image
        end
    end
end
