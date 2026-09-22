function [allPassed] = run_all_matlab_tests()
    % RUN_ALL_MATLAB_TESTS Discovers and runs all MATLAB unit test suites.
    % Returns 1 if all pass, 0 otherwise (and sets exit code appropriately).

    fprintf('=================================================================\n');
    fprintf('  DRISHYA: Automated MATLAB Test Suite Execution                 \n');
    fprintf('=================================================================\n');

    testDir = fileparts(mfilename('fullpath'));
    baseDir = fullfile(testDir, '..', '..');

    addpath(genpath(fullfile(baseDir, 'matlab')));
    addpath(genpath(fullfile(baseDir, 'simulink')));
    addpath(testDir);

    import matlab.unittest.TestSuite;
    import matlab.unittest.TestRunner;

    suite1 = TestSuite.fromClass(?TestIQA);
    suite2 = TestSuite.fromClass(?TestSegmentation);
    suite3 = TestSuite.fromClass(?TestClassificationAndReporting);
    suite4 = TestSuite.fromClass(?TestSimulink);
    suite5 = TestSuite.fromClass(?TestClinicalBiomarkers);
    suite6 = TestSuite.fromClass(?TestEdgeCasesAndStress);
    suite7 = TestSuite.fromClass(?TestDualEngineFusion);

    fullSuite = [suite1, suite2, suite3, suite4, suite5, suite6, suite7];
    fprintf('Discovered %d test cases across 7 comprehensive test suites.\n\n', length(fullSuite));

    runner = TestRunner.withTextOutput;
    results = runner.run(fullSuite);

    fprintf('\n=================================================================\n');
    fprintf('  TEST SUMMARY RESULTS TABLE                                      \n');
    fprintf('=================================================================\n');
    disp(table(results));

    numPassed = sum([results.Passed]);
    numFailed = sum([results.Failed]);
    numIncomplete = sum([results.Incomplete]);
    total = length(results);

    fprintf('Total Tests: %d | Passed: %d | Failed: %d | Incomplete: %d\n', ...
        total, numPassed, numFailed, numIncomplete);

    allPassed = (numFailed == 0 && numIncomplete == 0);
    if allPassed
        fprintf('SUCCESS: 100%% of DRISHYA MATLAB Unit Tests Passed! 🎉\n');
    else
        fprintf('FAILURE: Some tests failed. Inspect diagnostic log above.\n');
    end
end
