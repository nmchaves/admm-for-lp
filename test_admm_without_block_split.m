clear;clc;close all
seed = 0;

%% generate problem
prob_seed = 0;
[c, A, b, opt_val] = generate_linprog_problem(100,200,prob_seed);

%% parameters
MAX_ITER = 1e4; % max # of iterations
TOL = 1e-4;     % Tolerance
beta = 0.9;     % parameter (for augmenting lagrangian)
gamm = 0.1;     % rate for change in mu (for interior point methods)
precondition = false;

%% Primal ADMM
[~,~,~,~,~] = lp_primal_admm(c, A, b, MAX_ITER, TOL, beta, false, seed);

%% Dual ADMM
[~,~,~,~,~] = lp_dual_admm(c, A, b, MAX_ITER, TOL, beta, false, seed);

%% Primal ADMM Pre-conditioning
[~,~,~,~,~] = lp_primal_admm(c, A, b, MAX_ITER, TOL, beta, true, seed);

%% Dual ADMM Pre-conditioning
[~,~,~,~,~] = lp_dual_admm(c, A, b, MAX_ITER, TOL, beta, true, seed);

%% Primal Interior Point ADMM
[~,~,~,~,~] = lp_primal_ip_admm(c, A, b, MAX_ITER, TOL, beta, gamm, false, seed);

%% Dual Interior Point ADMM
[~,~,~,~,~] = lp_dual_ip_admm(c, A, b, MAX_ITER, TOL, beta, gamm, false, seed);

%% Primal Interior Point ADMM Pre-conditioning
[~,~,~,~,~] = lp_primal_ip_admm(c, A, b, MAX_ITER, TOL, beta, gamm, true, seed);

%% Dual Interior Point ADMM Pre-conditioning
[~,~,~,~,~] = lp_dual_ip_admm(c, A, b, MAX_ITER, TOL, beta, gamm, true, seed);
