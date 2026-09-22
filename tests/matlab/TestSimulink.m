classdef TestSimulink < matlab.unittest.TestCase
    % TESTSIMULINK Unit tests for Simulink Model and Discrete-Event Logistics Simulation.

    properties
        ModelPath
    end

    methods (TestMethodSetup)
        function checkModelFile(testCase)
            simDir = fileparts(which('simulate_telemedicine_district'));
            testCase.ModelPath = fullfile(simDir, 'telemed_screening_network.slx');
        end
    end

    methods (Test)
        function testModelExistsAndLoads(testCase)
            % Check file exists on disk (Simulink .slx returns 4, regular files return 2)
            testCase.verifyTrue(ismember(exist(testCase.ModelPath, 'file'), [2, 4]));
            [~, modelName, ~] = fileparts(testCase.ModelPath);
            load_system(testCase.ModelPath);
            testCase.verifyTrue(bdIsLoaded(modelName));
            close_system(modelName, 0);
        end

        function testSimulationRunsAndCompletes(testCase)
            p.annual_patients = 1000; % Short test volume
            p.operating_days = 5;
            p.clinic_hours_per_day = 7;
            p.bandwidth_kbps = 512;
            p.num_doctors = 2;
            p.mode = 'AI_ASSISTED';
            
            [summary, ts] = simulate_telemedicine_district(p);
            testCase.verifyEqual(summary.Mode, 'AI_ASSISTED');
            testCase.verifyGreaterThan(summary.TotalPatientsDaily, 0);
            testCase.verifyLessThan(summary.MeanTurnaroundMinutes, 30.0);
            testCase.verifyGreaterThan(length(ts.TimeMinutes), 0);
        end

        function testManualReviewQueueCollapse(testCase)
            % With only 1 doctor, manual review will accumulate heavy backlog
            p_manual.annual_patients = 5000;
            p_manual.operating_days = 10;
            p_manual.num_doctors = 1;
            p_manual.mode = 'MANUAL_UNASSISTED';
            
            [summary_m, ~] = simulate_telemedicine_district(p_manual);
            testCase.verifyGreaterThan(summary_m.EndOfDayBacklog, 100);
            testCase.verifyGreaterThan(summary_m.MeanTurnaroundMinutes, 100.0);
        end
    end
end
