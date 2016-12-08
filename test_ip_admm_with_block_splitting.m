clear;clc;close all
seed = 0;

%% generate problem
m = 20;
n = 100;
prob_seed = 0;
[c, A, b, opt_val] = generate_linprog_problem(m, n , prob_seed);

%% parameters
MAX_ITER = 1e4; % max # of iterations
TOL = 1e-3;     % Tolerance
beta = 0.9;     % parameter (for augmenting lagrangian)
gamma = 0.1;
precondition = false;
verb = true; 

%% Primal IP ADMM with 1 block (no splitting)
tic
NUM_BLOCKS = 1;
rnd_permute = true; % This would have no effect anyways
[ov1,~,~,~,eh1] = lp_primal_ip_admm_with_splitting(c, A, b, MAX_ITER, TOL, beta, gamma, ...
                                    precondition, NUM_BLOCKS, rnd_permute, seed, verb);
toc
%% Primal IP ADMM with 5 blocks
tic
NUM_BLOCKS = 5;
rnd_permute = true;
[ov2,~,~,~,eh2] = lp_primal_ip_admm_with_splitting(c, A, b, MAX_ITER, TOL, beta, gamma,...
                                    precondition, NUM_BLOCKS, rnd_permute, seed, verb);
toc                             
%% Primal IP ADMM with 10 blocks
tic
NUM_BLOCKS = 10;
rnd_permute = true;
[ov3,~,~,~,eh3] = lp_primal_ip_admm_with_splitting(c, A, b, MAX_ITER, TOL, beta,gamma, ...
                                    precondition, NUM_BLOCKS, rnd_permute, seed, verb);
toc

%% Plot                        
figure(1)
semilogy(1:length(eh1),eh1, 'r')
hold on
semilogy(1:length(eh2),eh2, 'g')
semilogy(1:length(eh3),eh3, 'b')
xlabel('Iteration')
ylabel('Abs Error: ||A*x1-b||')
title('Primal IP ADMM')

%% Dual IP ADMM with 1 block (no splitting)
tic
NUM_BLOCKS = 1;
rnd_permute = true; % This would have no effect anyways
[ov1,~,~,~,eh1] = lp_dual_ip_admm_with_splitting(c, A, b, MAX_ITER, TOL, beta, gamma, ...
                                    precondition, NUM_BLOCKS, rnd_permute, seed, verb);
toc
%% Dual IP ADMM with 5 blocks
tic
NUM_BLOCKS = 5;
rnd_permute = true;
[ov2,~,~,~,eh2] = lp_dual_ip_admm_with_splitting(c, A, b, MAX_ITER, TOL, beta, gamma,...
                                    precondition, NUM_BLOCKS, rnd_permute, seed, verb);
toc                             
%% Dual IP ADMM with 10 blocks
tic
NUM_BLOCKS = 10;
rnd_permute = true;
[ov3,~,~,~,eh3] = lp_dual_ip_admm_with_splitting(c, A, b, MAX_ITER, TOL, beta,gamma, ...
                                    precondition, NUM_BLOCKS, rnd_permute, seed, verb);
toc

%% Plot                        
figure(2)
semilogy(1:length(eh1),eh1, 'r')
hold on
semilogy(1:length(eh2),eh2, 'g')
semilogy(1:length(eh3),eh3, 'b')
xlabel('Iteration')
ylabel('Abs Error: ||A*x1-b||')
title('Dual IP ADMM')
