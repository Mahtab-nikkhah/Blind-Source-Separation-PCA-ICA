%% Blind Source Separation (BSS) using Manual PCA and FastICA (No Toolbox)
% Signals and Systems Course Project — Sharif University of Technology
%
% Implements Principal Component Analysis (PCA) via covariance
% eigendecomposition and Independent Component Analysis (FastICA with
% symmetric orthogonalization) completely from scratch in pure MATLAB.

clear; clc; close all;

%% Step 1: Generate Synthetic Independent Sources
fs = 1000;                          % Sampling frequency (Hz)
t  = 0:1/fs:2;                      % Time vector (2 seconds duration)
N  = length(t);

source1 = sin(2 * pi * 50 * t);             % 50 Hz sine wave
source2 = sign(sin(2 * pi * 70 * t));       % 70 Hz square wave
S = [source1; source2];

%% Step 2: Mix Sources Using a Linear Mixing Matrix
H = [1.0, 0.1;
     0.1, 1.0];                     % Mixing matrix
X = H * S;                          % Observed mixed signals

%% Step 3: Manual PCA (Centering + Covariance Eigendecomposition)
X_centered = X - mean(X, 2);
cov_matrix = (X_centered * X_centered') / N;

[eig_vec, eig_val] = eig(cov_matrix);
[eig_val_sorted, idx] = sort(diag(eig_val), 'descend');
eig_vec = eig_vec(:, idx);

% Project centered observations onto principal components
PCA_output = eig_vec' * X_centered;

%% Step 4: Manual FastICA (Whitening + Fixed-Point Symmetric Orthogonalization)
% 4a. Whitening transform: Z = D^(-1/2) * E' * X_centered so E[Z*Z'] = I
D_inv_sqrt = diag(1 ./ sqrt(eig_val_sorted));
V = D_inv_sqrt * eig_vec';
Z = V * X_centered;

% 4b. Initialize orthogonal unmixing matrix W
theta0 = 0.5;
W = [cos(theta0), -sin(theta0);
     sin(theta0),  cos(theta0)];

max_iter = 200;
tol = 1e-8;

for iter = 1:max_iter
    U = W * Z;                      % Current source estimates
    g = tanh(U);                    % Contrast nonlinearity
    g_prime = 1 - g.^2;             % Derivative of tanh(U)

    % Fixed-point FastICA update
    W_new = (g * Z') / N - diag(mean(g_prime, 2)) * W;

    % Symmetric orthogonalization: W_new = (W_new * W_new')^(-1/2) * W_new
    [E_w, D_w] = eig(W_new * W_new');
    W_new = E_w * diag(1 ./ sqrt(diag(D_w))) * E_w' * W_new;

    % Check convergence (invariant to sign flips)
    if max(abs(abs(diag(W_new * W')) - 1)) < tol
        W = W_new;
        break;
    end
    W = W_new;
end

% Recover independent components
ICA_output = W * Z;

%% Step 5: Resolve BSS Permutation/Scale Ambiguity & Evaluate SNR
[PCA_aligned, SNR_PCA, corr_PCA] = align_and_evaluate(S, PCA_output);
[ICA_aligned, SNR_ICA, corr_ICA] = align_and_evaluate(S, ICA_output);

fprintf('FastICA converged in %d iterations.\n\n', iter);
fprintf('--- Separation Performance ---\n');
fprintf('PCA     | Source 1: %6.2f dB (|r| = %.4f) | Source 2: %6.2f dB (|r| = %.4f) | Mean SNR: %6.2f dB\n', ...
    SNR_PCA(1), corr_PCA(1), SNR_PCA(2), corr_PCA(2), mean(SNR_PCA));
fprintf('FastICA | Source 1: %6.2f dB (|r| = %.4f) | Source 2: %6.2f dB (|r| = %.4f) | Mean SNR: %6.2f dB\n', ...
    SNR_ICA(1), corr_ICA(1), SNR_ICA(2), corr_ICA(2), mean(SNR_ICA));

%% Step 6: Visualization (Zoomed to First 100 ms for Waveform Clarity)
zoom_mask = t <= 0.10;
t_ms = t(zoom_mask) * 1000;

figure('Name', 'Blind Source Separation: PCA vs. FastICA', 'Position', [100, 100, 1050, 750]);

subplot(4, 2, 1); plot(t_ms, S(1, zoom_mask), 'LineWidth', 1.4);
title('Original Source 1 (50 Hz Sine)'); ylabel('Amplitude'); grid on; ylim([-1.35, 1.35]);

subplot(4, 2, 2); plot(t_ms, S(2, zoom_mask), 'LineWidth', 1.4);
title('Original Source 2 (70 Hz Square)'); ylabel('Amplitude'); grid on; ylim([-1.35, 1.35]);

subplot(4, 2, 3); plot(t_ms, X(1, zoom_mask), 'LineWidth', 1.4);
title('Mixed Signal 1'); ylabel('Amplitude'); grid on; ylim([-1.35, 1.35]);

subplot(4, 2, 4); plot(t_ms, X(2, zoom_mask), 'LineWidth', 1.4);
title('Mixed Signal 2'); ylabel('Amplitude'); grid on; ylim([-1.35, 1.35]);

subplot(4, 2, 5); plot(t_ms, PCA_aligned(1, zoom_mask), 'LineWidth', 1.4);
title(sprintf('PCA Output 1 (SNR: %.2f dB)', SNR_PCA(1))); ylabel('Amplitude'); grid on; ylim([-1.35, 1.35]);

subplot(4, 2, 6); plot(t_ms, PCA_aligned(2, zoom_mask), 'LineWidth', 1.4);
title(sprintf('PCA Output 2 (SNR: %.2f dB)', SNR_PCA(2))); ylabel('Amplitude'); grid on; ylim([-1.35, 1.35]);

subplot(4, 2, 7); plot(t_ms, ICA_aligned(1, zoom_mask), 'LineWidth', 1.4);
title(sprintf('FastICA Output 1 (SNR: %.2f dB)', SNR_ICA(1))); xlabel('Time (ms)'); ylabel('Amplitude'); grid on; ylim([-1.35, 1.35]);

subplot(4, 2, 8); plot(t_ms, ICA_aligned(2, zoom_mask), 'LineWidth', 1.4);
title(sprintf('FastICA Output 2 (SNR: %.2f dB)', SNR_ICA(2))); xlabel('Time (ms)'); ylabel('Amplitude'); grid on; ylim([-1.35, 1.35]);

%% Local Helper Function: Permutation/Scale Alignment & SNR Calculation
function [Y_aligned, snr_db, abs_corr] = align_and_evaluate(S_true, Y_est)
    % Compute pairwise absolute correlation without Statistics Toolbox
    C = zeros(2, 2);
    for r = 1:2
        s_zero = S_true(r, :) - mean(S_true(r, :));
        for c = 1:2
            y_zero = Y_est(c, :) - mean(Y_est(c, :));
            C(r, c) = abs((s_zero * y_zero') / (norm(s_zero) * norm(y_zero)));
        end
    end

    % Match recovered components to ground-truth sources
    if (C(1, 2) + C(2, 1)) > (C(1, 1) + C(2, 2))
        Y_matched = Y_est([2, 1], :);
        abs_corr  = [C(1, 2); C(2, 1)];
    else
        Y_matched = Y_est;
        abs_corr  = [C(1, 1); C(2, 2)];
    end

    % Align sign and amplitude via least-squares scaling
    Y_aligned = zeros(size(Y_matched));
    for k = 1:2
        scale = (S_true(k, :) * Y_matched(k, :)') / (Y_matched(k, :) * Y_matched(k, :)');
        Y_aligned(k, :) = scale * Y_matched(k, :);
    end

    % Compute per-source Signal-to-Noise Ratio (dB)
    signal_power = sum(S_true.^2, 2);
    noise_power  = sum((S_true - Y_aligned).^2, 2);
    snr_db = 10 * log10(signal_power ./ noise_power);
end
