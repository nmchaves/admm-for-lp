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
beta_range_low = linspace(0.1, 0.9, 5);     % parameter (for augmenting lagrangian)
beta_range_hi = linspace(2, 10, 5);
beta_range = [beta_range_low beta_range_hi];
N = 1; % # number of problems to solve
corr_tol = 0.1; % Tolerance for correctness

legend_obj = {};
% Variables for plotting
for b_idx=1:length(beta_range)
    legend_obj{b_idx} = strcat('b=',num2str(beta_range(b_idx)));
end


%% Run Experiments for Various Beta Values

figure(1)
cmap = colormap(hsv);
num_colors = size(cmap, 1);
color_spacing = floor(num_colors / size(beta_range,2));

for i=1:N
    prob_seed = i-1;
    disp(' ')
    disp(['Problem ',num2str(i)])
    
    beta_idx = 0;
    for beta=beta_range
        beta_idx = beta_idx + 1;
        disp(['beta=', num2str(beta)])
        [ov,~,~,~,eh] = sdp_primal( c, A, b, MAX_ITER, TOL, beta, seed);
        disp(num2str(ov))
        
        % Plot the error history
        semilogy(1:length(eh),eh, 'Color', cmap(1+(beta_idx-1)*color_spacing,:))
        hold on
    end
    legend(legend_obj)
    xlabel('Iteration')
    ylabel('Abs Error')
    title('SDP Primal')

end
