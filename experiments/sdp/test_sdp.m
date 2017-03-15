clear;clc;close all
seed = 0;

%% generate problem
m = 30;
n = 10; % the effective n size is n^2
prob_seed = 0;
[c, A, b, X_opt] = generate_sdp_problem(m, n , prob_seed);

%% parameters
MAX_ITER = 1e4; % max # of iterations
TOL = 1e-6;     % Tolerance

% set beta parameters to test (for augmenting lagrangian)
% use a negative beta, which indicates to the solver that it must
% auto-select beta
beta_range_low = linspace(0.1, 0.9, 5);     
beta_range_hi = linspace(2, 10, 5);
beta_range = [beta_range_low, beta_range_hi 20, -1];
N = 1; % # number of problems to solve
corr_tol = 0.1; % Tolerance for correctness

legend_obj = {};
% Variables for plotting
for b_idx=1:length(beta_range)
    legend_obj{b_idx} = strcat('\beta=',num2str(beta_range(b_idx)));
end


%% Run Primal SDP Experiments for Various Beta Values

figure(1)
cmap = colormap(hsv);
num_colors = size(cmap, 1);
color_spacing = floor(num_colors / size(beta_range,2));

for i=1:N
    prob_seed = i-1;
    disp(' ')
    disp(['Problem ',num2str(i)])
    
    for beta_idx = 1:length(beta_range)
        beta = beta_range(beta_idx);
        disp(['beta=', num2str(beta)])
        [ov,~,~,~,eh, beta_guess] = sdp_primal( c, A, b, MAX_ITER, TOL, beta, seed);
        disp(num2str(ov))
        
        % Plot the error history
        if beta > 0
             semilogy(1:length(eh),eh, 'Color', cmap(1+(beta_idx-1)*color_spacing,:))
        else
            semilogy(1:length(eh),eh, '--', 'Color', cmap(1+(beta_idx-1)*color_spacing,:))
            legend_obj{beta_idx} = strcat('\beta=',num2str(beta_guess, 3));
        end
        hold on
    end
    legend(legend_obj)
    xlabel('Iteration')
    ylabel('Abs Error')
    title('SDP Primal')

end

%% Run Dual SDP Experiments for Various Beta Values

figure(2)
cmap = colormap(hsv);
num_colors = size(cmap, 1);
color_spacing = floor(num_colors / size(beta_range,2));

for i=1:N
    prob_seed = i-1;
    disp(' ')
    disp(['Problem ',num2str(i)])
    
    for beta_idx = 1:length(beta_range)
        beta = beta_range(beta_idx);
        disp(['beta=', num2str(beta)])
        [ov,~,~,~,eh, beta_guess] = sdp_dual( c, A, b, MAX_ITER, TOL, beta, seed);
        disp(num2str(ov))
        
        % Plot the error history
        if beta > 0
             semilogy(1:length(eh),eh, 'Color', cmap(1+(beta_idx-1)*color_spacing,:))
        else
            semilogy(1:length(eh),eh, '--', 'Color', cmap(1+(beta_idx-1)*color_spacing,:))
            legend_obj{beta_idx} = strcat('\beta=',num2str(beta_guess, 3));
        end
        hold on
    end
    legend(legend_obj)
    xlabel('Iteration')
    ylabel('Abs Error')
    title('SDP Dual')

end