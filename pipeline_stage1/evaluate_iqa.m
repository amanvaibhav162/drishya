function [status, feedback, weights, Q] = evaluate_iqa(metrics, thresholds)
    % Composite Quality Score: Q = w_f*F + w_i*I + w_v*V + w_c*C
    F_norm = min(1, metrics.Focus / thresholds.F_target);
    I_norm = metrics.Illumination;
    V_norm = metrics.FOV;
    C_norm = min(1, metrics.Contrast / thresholds.C_target);
    
    weights = [0.35, 0.25, 0.20, 0.20];
    Q = weights(1)*F_norm + weights(2)*I_norm + weights(3)*V_norm + weights(4)*C_norm;
    
    % Two-category policy: Reject or Accept & Enhance
    % Eliminates bimodal distribution shift so all training data shares identical feature representation
    if Q < thresholds.Q_reject || V_norm < 0.60
        status = "UNGRADABLE";
        feedback = sprintf("RECAPTURE: F=%.2f, I=%.2f, V=%.2f", F_norm, I_norm, V_norm);
    else
        status = "ACCEPT_AND_ENHANCE";
        feedback = "STANDARDIZE_AND_ENHANCE";
    end
end
