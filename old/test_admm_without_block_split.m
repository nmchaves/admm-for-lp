clear;clc;close all

%% generate problem
prob_seed = 0;
n_blocks = 10;
sparsity = 0.2;
% [c, A, b, opt_val] = generate_linprog_problem(100,200,prob_seed);

% [c, A, b, opt_val,blocks] = generate_large_sparse_problem(100,200,...
%    'block',sparsity,n_blocks,prob_seed,true);

[c, A, b, opt_val,blocks] = generate_large_sparse_problem(100,200,...
    'random',sparsity,n_blocks,prob_seed,true);

A = full(A);


%% parameters
seed = 0;
MAX_ITER = 1e4; % max # of iterations
TOL = 1e-4;     % Tolerance
beta = 0.9;     % parameter (for augmenting lagrangian)
gamm = 0.1;     % rate for change in mu (for interior point methods)
precondition = false;

%% Primal ADMM
[~,~,~,~,~] = lp_primal_admm(c, A, b, MAX_ITER, TOL, beta, false, seed, true);

%% Dual ADMM
[~,~,~,~,~] = lp_dual_admm(c, A, b, MAX_ITER, TOL, beta, false, seed, true);

%% Primal ADMM Pre-conditioning
[~,~,~,~,~] = lp_primal_admm(c, A, b, MAX_ITER, TOL, beta, true, seed, true);

%% Dual ADMM Pre-conditioning
[~,~,~,~,~] = lp_dual_admm(c, A, b, MAX_ITER, TOL, beta, true, seed, true);

%% Primal Interior Point ADMM
[~,~,~,~,~] = lp_primal_ip_admm(c, A, b, MAX_ITER, TOL, beta, gamm, false, seed, true);

%% Dual Interior Point ADMM
[~,~,~,~,~] = lp_dual_ip_admm(c, A, b, MAX_ITER, TOL, beta, gamm, false, seed, true);

%% Primal Interior Point ADMM Pre-conditioning
[~,~,~,~,~] = lp_primal_ip_admm(c, A, b, MAX_ITER, TOL, beta, gamm, true, seed, true);

%% Dual Interior Point ADMM Pre-conditioning
[~,~,~,~,~] = lp_dual_ip_admm(c, A, b, MAX_ITER, TOL, beta, gamm, true, seed, true);
