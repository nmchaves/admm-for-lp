clear;clc;close all
addpath(genpath('figure_util'))

%% generate problem
m = 3;
n = 3;

A = [1,1,1;1,1,2;1,2,2];
b = zeros(m,1);
c = zeros(n,1);

% Compute the Optimal Solution
disp('Running linprog solver...')
[opt_x, opt_val] = linprog(c,[],[],A,b,zeros(n,1));
disp(['linprog optval : ', num2str(opt_val)])

%% parameters
methods = {'primal','dual'}; % method names
s = 0;           % solver seed
MAX_ITER = 1e3;       % max # of iterations
TOL = 1e-4;      % Tolerance for algorithm
corr_tol = 0.01; % Tolerance for correctness
beta = 0.9;
rnd_perm = false;
verb = true;
precond = false;

%% Experiment with Primal
num_blocks = 3;
[ov,x_opt_primal,~,~,eh] = lp_primal_admm_with_splitting(c, A, b, MAX_ITER, TOL, beta, precond, num_blocks, rnd_perm, s, verb);
disp(['Error in x_opt: ', num2str(norm(x_opt_primal-opt_x))])

%% Experiment with Dual
num_blocks = 3;
[ov,x_opt_dual,~,~,eh] = lp_dual_admm_with_splitting(c, A, b, MAX_ITER, TOL, beta, precond, num_blocks, rnd_perm, s, verb);
disp(['Error in x_opt: ', num2str(norm(x_opt_dual-opt_x))])



                   