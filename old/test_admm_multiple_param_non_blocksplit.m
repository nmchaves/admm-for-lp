clear;clc;close all
%% Problem Paramters
n = 300;     % # of variables
m = 50;      % # of equality constraints
N = 10;      % # number of problems to solve

%% Solver Paramters
% methods = {'primal','dual','primal ip','dual ip'}; % method names
methods = {'primal','dual'}; % method names
s = 0;        % solver seed
MIT = 1e4;    % max # of iterations
TOL = 1e-3;   % Tolerance for algorithm
corr_tol = 0.1; % Tolerance for correctness
g = 0.1;     % rate for change in mu (for interior point methods)

all_beta = 0.2:0.1:0.9;

%% Initialization
single_result = zeros(length(all_beta),N);
result = repmat({single_result}, length(methods),2); % (lp solver type, precondition or not)

%% Experiment
for i_prob = 1:N
    prob_seed = i_prob-1;
    disp(' ')
    disp(['Problem ',num2str(i_prob)])
    [c, A, b, opt_val] = generate_linprog_problem(m,n,prob_seed);
    for i_beta = 1:length(all_beta)
        be = all_beta(i_beta);
        for i_precond = 1:2
            switch i_precond
                case 1
                    p = false;
                case 2
                    p = true;
            end
            for i_method = 1:length(methods)
                switch methods{i_method}
                    case 'primal'
                        [ov,~,~,~,eh] = lp_primal_admm(c, A, b, MIT, TOL, be, p, s);
                    case 'dual'
                        [ov,~,~,~,eh] = lp_dual_admm(c, A, b, MIT, TOL, be, p, s);
                    case 'primal ip'
                        [ov,~,~,~,eh] = lp_primal_ip_admm(c, A, b, MIT, TOL, be, g, p, s);
                    case 'dual ip'
                        [ov,~,~,~,eh] = lp_dual_ip_admm(c, A, b, MIT, TOL, be, g, p, s);
                end
                
                if abs(ov - opt_val) > corr_tol
                    disp(['Method: ', [methods{i_method}]])
                    disp(['Problem: ',num2str(i_prob)])
                    disp(['Beta: ',num2str(all_beta(i_beta))])
                    if p
                        disp('Using Preconditioning')
                    end
                    disp(['Converged at:', num2str(length(eh))])
                    warning('Incorrect Solution!')
                    % store the number of steps used for convergence
                    result{i_method,i_precond}(i_beta,i_prob) = -1;
                else 
                    % store the number of steps used for convergence
                    result{i_method,i_precond}(i_beta,i_prob) = length(eh);
                end
            end
        end
    end
end

save('test_large_admm_precond.mat','result','methods','all_beta')

%% Plot Results
load('test_large_admm_precond.mat')

figure('Position', [100, 100, 500, 200]);
subplot(1,2,1)
plot_errorbar_param_conv(result(:,1),all_beta,methods, [0,10000],'\beta')
title('without preconditioning')
subplot(1,2,2)
title('with preconditioning')
plot_errorbar_param_conv(result(:,2),all_beta,methods, [0,10000],'\beta')
fname = 'primal_dual_preconditioning';
save_current_figure(['figures/',fname],'high','-png');