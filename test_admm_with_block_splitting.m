clear;clc;close all
seed = 0;

%% generate problem
m = 100;
n = 200;
prob_seed = 0;
[c, A, b, opt_val] = generate_linprog_problem(m, n , prob_seed);

%% parameters
MAX_ITER = 1e4; % max # of iterations
TOL = 1e-4;     % Tolerance
beta = 0.9;     % parameter (for augmenting lagrangian)
precondition = false;

%% Primal ADMM with 1 block (no splitting)
NUM_BLOCKS = 1;
rnd_permute = true; % This would have no effect anyways
[ov1,~,~,~,eh1] = lp_primal_admm_with_splitting(c, A, b, MAX_ITER, TOL, beta, ...
                                    precondition, NUM_BLOCKS, rnd_permute, seed);

%% Primal ADMM with 2 blocks
NUM_BLOCKS = 2;
rnd_permute = true;
[ov2,~,~,~,eh2] = lp_primal_admm_with_splitting(c, A, b, MAX_ITER, TOL, beta, ...
                                    precondition, NUM_BLOCKS, rnd_permute, seed);
                                
%% Primal ADMM with 3 blocks
NUM_BLOCKS = 3;
rnd_permute = true;
[ov3,~,~,~,eh3] = lp_primal_admm_with_splitting(c, A, b, MAX_ITER, TOL, beta, ...
                                    precondition, NUM_BLOCKS, rnd_permute, seed);
%% Plot 
                                
figure(1)
hold on
plot(eh1, 'r')
plot(eh2, 'b')
plot(eh3, 'g')