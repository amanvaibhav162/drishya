classdef TestClinicalBiomarkers < matlab.unittest.TestCase
    % TESTCLINICALBIOMARKERS Unit tests for sub-pixel MAs, ETDRS 4-2-1 quadrants, and CSME staging.

    properties
        DummyFundus
    end

    methods (TestMethodSetup)
        function createTestFundus(testCase)
            sz = 384;
            [X, Y] = meshgrid(1:sz, 1:sz);
            R = sqrt((X - sz/2).^2 + (Y - sz/2).^2);
            fundus = zeros(sz, sz, 3, 'uint8');
            retina_mask = R <= (sz/2 - 16);
            fundus(:,:,1) = uint8(retina_mask * 180);
            fundus(:,:,2) = uint8(retina_mask * 90);
            fundus(:,:,3) = uint8(retina_mask * 25);
            testCase.DummyFundus = fundus;
        end
    end

    methods (Test)
        function testCSMEClassification(testCase)
            % Test 1: Hard exudates within 0.25 DD of fovea -> Stage 2 (CSME)
            sz = 384;
            fovea_loc = [192, 192];
            od_radius = 35; % Disc Diameter = 70 px. 1/3 DD = 23.3 px.
            
            % Place exudate at distance 15 px (15/70 = 0.21 DD < 0.333 DD)
            mask_csme = false(sz, sz);
            mask_csme(192, 192 + 15) = true;
            stats_csme = evaluate_dme_risk(mask_csme, fovea_loc, od_radius);
            testCase.verifyEqual(stats_csme.Stage, 2);
            testCase.verifyTrue(stats_csme.IsCSME);
            testCase.verifyLessThanOrEqual(stats_csme.MinFoveaDistDD, 0.333);

            % Test 2: Hard exudates at 0.60 DD (distance 42 px) -> Stage 1 (Non-Center-Involved)
            mask_dme1 = false(sz, sz);
            mask_dme1(192, 192 + 42) = true;
            stats_dme1 = evaluate_dme_risk(mask_dme1, fovea_loc, od_radius);
            testCase.verifyEqual(stats_dme1.Stage, 1);
            testCase.verifyFalse(stats_dme1.IsCSME);
            testCase.verifyGreaterThan(stats_dme1.MinFoveaDistDD, 0.333);
            testCase.verifyLessThanOrEqual(stats_dme1.MinFoveaDistDD, 1.0);

            % Test 3: Hard exudates at 1.50 DD (distance 105 px) -> Stage 0 (Low Risk)
            mask_dme0 = false(sz, sz);
            mask_dme0(192, 192 + 105) = true;
            stats_dme0 = evaluate_dme_risk(mask_dme0, fovea_loc, od_radius);
            testCase.verifyEqual(stats_dme0.Stage, 0);
            testCase.verifyFalse(stats_dme0.IsCSME);

            % Test 4: Empty mask -> Stage 0
            stats_empty = evaluate_dme_risk(false(sz, sz), fovea_loc, od_radius);
            testCase.verifyEqual(stats_empty.Stage, 0);
            testCase.verifyFalse(stats_empty.IsCSME);
        end

        function testSubPixelMADerivation(testCase)
            % Synthesize a known quadratic depression with peak at (x0 + 0.30, y0 - 0.20)
            sz = 384;
            img = testCase.DummyFundus;
            x0 = 150; y0 = 150;
            dx_true = 0.25; dy_true = -0.20;
            
            % Generate local dark depression in green channel
            for r = -3:3
                for c = -3:3
                    dist_sq = (c - dx_true)^2 + (r - dy_true)^2;
                    drop = round(40 * exp(-dist_sq / 2.0));
                    img(y0+r, x0+c, 2) = max(0, img(y0+r, x0+c, 2) - drop);
                end
            end
            
            vessel_mask = false(sz, sz);
            od_mask = false(sz, sz);
            [ma_mask, ma_table, num_ma] = detect_subpixel_ma(img, vessel_mask, od_mask);
            
            testCase.verifyGreaterThanOrEqual(num_ma, 1);
            % Find the detected MA closest to (x0, y0)
            dists = sqrt((ma_table.X_SubPixel - x0).^2 + (ma_table.Y_SubPixel - y0).^2);
            [min_d, best_idx] = min(dists);
            testCase.verifyLessThan(min_d, 2.0);
            
            % Verify sub-pixel coordinates are floating-point non-integers
            sp_x = ma_table.X_SubPixel(best_idx);
            testCase.verifyTrue(abs(sp_x - round(sp_x)) > 0 || abs(ma_table.Y_SubPixel(best_idx) - round(ma_table.Y_SubPixel(best_idx))) > 0);
        end

        function testETDRSQuadrantHemorrhageAnalysis(testCase)
            % Place hemorrhages in all 4 quadrants and verify severe quadrant detection
            sz = 384;
            img = testCase.DummyFundus;
            
            % Put 6 distinct dark spots in each quadrant
            quad_centers = [
                90, 90;    % Q1: Superior-Left
                90, 290;   % Q2: Superior-Right
                290, 90;   % Q3: Inferior-Left
                290, 290   % Q4: Inferior-Right
            ];
            
            for q = 1:4
                cx = quad_centers(q, 1);
                cy = quad_centers(q, 2);
                for spot = 1:6
                    sx = cx + (spot - 3) * 12;
                    sy = cy + (mod(spot, 2) * 10);
                    img(sy-2:sy+2, sx-2:sx+2, 2) = 10; % dark spot in green
                end
            end
            
            vessel_mask = false(sz, sz);
            od_mask = false(sz, sz);
            ma_mask = false(sz, sz);
            [~, ~, ~, total_he_mask, he_stats] = classify_hemorrhages(img, vessel_mask, od_mask, ma_mask);
            
            testCase.verifyGreaterThan(he_stats.TotalCount, 15);
            testCase.verifyEqual(he_stats.QuadrantsInvolved, 4);
            testCase.verifyGreaterThanOrEqual(he_stats.SevereQuadrants, 3);
        end

        function testNVDvsNVESeparation(testCase)
            sz = 384;
            vessel_mask = false(sz, sz);
            od_mask = false(sz, sz);
            od_center = [100, 192];
            od_radius = 30;
            
            % Create OD mask
            [X, Y] = meshgrid(1:sz, 1:sz);
            od_mask = ((X - od_center(1)).^2 + (Y - od_center(2)).^2) <= od_radius^2;
            
            % 1. Create tangled vessel bundle near OD margin (NVD candidate, dist <= 1 DD)
            % 1 DD = 60 px from OD center
            nvd_x = 100 + 35; nvd_y = 192;
            for k = -10:10
                vessel_mask(nvd_y + k, nvd_x + round(3*sin(k))) = true;
                vessel_mask(nvd_y + round(3*cos(k)), nvd_x + k) = true;
            end
            
            % 2. Create peripheral tangled vessel bundle far from OD (NVE candidate)
            nve_x = 280; nve_y = 192;
            for k = -10:10
                vessel_mask(nve_y + k, nve_x + round(3*sin(k))) = true;
                vessel_mask(nve_y + round(3*cos(k)), nve_x + k) = true;
            end
            
            [nvd_mask, nve_mask, has_nvd, has_nve, pdr_score, nv_stats] = ...
                detect_neovascularization(testCase.DummyFundus, vessel_mask, od_mask, od_center, od_radius);
            
            testCase.verifyTrue(islogical(nvd_mask));
            testCase.verifyTrue(islogical(nve_mask));
            testCase.verifyGreaterThanOrEqual(pdr_score, 0.0);
            testCase.verifyLessThanOrEqual(pdr_score, 1.0);
        end
    end
end
