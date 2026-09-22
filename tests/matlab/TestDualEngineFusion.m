classdef TestDualEngineFusion < matlab.unittest.TestCase
    % TESTDUALENGINEFUSION Tests clinical decision fusion, safety overrides, and uncertainty calibration.

    methods (Test)
        function testPDRMorphologySafetyOverride(testCase)
            % Scenario: Deep learning predicts Grade 0 (No DR), but segmented retina shows NVD
            dummy_img = zeros(384, 384, 3, 'uint8');
            seg = run_segmentation_pipeline(dummy_img);
            
            % Inject morphological NVD detection
            seg.HasNVD = true;
            seg.PDRRiskScore = 0.85;
            
            % DL model falsely predicts Grade 0 with modest confidence
            dl_logits = [2.5, -1.0, 0.5, 0.2, 0.8]; % Predicts Grade 0, but referral_prob > 0.25
            
            diag = predict_dr_grade(dummy_img, seg, dl_logits);
            
            testCase.verifyEqual(diag.Grade, 4);
            testCase.verifyTrue(diag.IsReferable);
            testCase.verifyEqual(diag.Urgency, "EMERGENCY_INTERVENTION");
            testCase.verifyTrue(contains(diag.Rationale, "Safety Escalation") || contains(diag.Rationale, "PDR"));
        end

        function testCSMEUrgentReferralOverride(testCase)
            % Scenario: Patient has Grade 1 Mild NPDR (a few MAs), but hard exudate is in fovea center (CSME)
            dummy_img = zeros(384, 384, 3, 'uint8');
            seg = run_segmentation_pipeline(dummy_img);
            
            seg.NumMA = 2;
            seg.ExudateStats.AreaPct = 0.05;
            
            % Inject CSME
            seg.IsCSME = true;
            seg.DMEStats.Stage = 2;
            seg.DMEStats.IsCSME = true;
            seg.DMEStats.MinFoveaDistDD = 0.18; % Well inside 1/3 DD
            seg.DMEStats.Description = "HIGH-RISK CSME: Exudates within 0.18 DD.";
            
            % DL predicts Grade 1
            dl_logits = [-1.0, 3.0, 0.0, -1.0, -2.0];
            
            diag = predict_dr_grade(dummy_img, seg, dl_logits);
            
            testCase.verifyEqual(diag.Grade, 1);
            testCase.verifyTrue(diag.IsReferable); % Must be referred due to CSME!
            testCase.verifyEqual(diag.Urgency, "REFER_URGENT_CSME");
            testCase.verifyTrue(contains(diag.ActionPlan, "CSME") || contains(diag.ActionPlan, "Macular Edema"));
        end

        function testFourQuadrantSevereNPDREscalation(testCase)
            % Scenario: DL predicts Grade 1, but 4-2-1 rule is met with severe hemorrhages in all 4 quadrants
            dummy_img = zeros(384, 384, 3, 'uint8');
            seg = run_segmentation_pipeline(dummy_img);
            
            seg.HemorrhageStats.TotalCount = 28;
            seg.HemorrhageStats.QuadrantsInvolved = 4;
            seg.HemorrhageStats.SevereQuadrants = 4;
            
            dl_logits = [-0.5, 2.8, 0.5, -0.5, -2.0]; % DL predicts Grade 1
            
            diag = predict_dr_grade(dummy_img, seg, dl_logits);
            
            testCase.verifyEqual(diag.Grade, 3); % Escalated to Severe NPDR (Grade 3)
            testCase.verifyTrue(diag.IsReferable);
            testCase.verifyEqual(diag.Urgency, "REFER_URGENT");
        end

        function testConcordantNormalRetina(testCase)
            dummy_img = zeros(384, 384, 3, 'uint8');
            seg = run_segmentation_pipeline(dummy_img);
            
            % DL strongly predicts Grade 0
            dl_logits = [5.0, -2.0, -3.0, -4.0, -5.0];
            
            diag = predict_dr_grade(dummy_img, seg, dl_logits);
            
            testCase.verifyEqual(diag.Grade, 0);
            testCase.verifyFalse(diag.IsReferable);
            testCase.verifyEqual(diag.Urgency, "ROUTINE");
            testCase.verifyGreaterThan(diag.Confidence, 85.0);
        end

        function testCalibratedUncertaintyBoundary(testCase)
            % Completely flat logits (maximum uncertainty)
            flat_logits = zeros(1, 5);
            [probs, conf, ci] = calibrate_confidence(flat_logits, 1.0);
            
            testCase.verifyEqual(length(probs), 5);
            testCase.verifyTrue(all(abs(probs - 0.20) < 1e-4));
            testCase.verifyLessThan(conf, 45.0); % Low confidence (<45%)
            testCase.verifyLessThan(ci(1), 35.0);
            testCase.verifyGreaterThan(ci(2), 5.0);
        end
    end
end
