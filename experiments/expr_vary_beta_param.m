clear;clc;close all
addpath(genpath('figure_util'))
seed = 0;

%% 

beta_range = [1, 2, 3];


%% generate problem
m = 50;
n = 300;
prob_seed = 1;
[c, A, b, opt_val] = generate_linprog_problem(m, n , prob_seed);

%% parameters
MAX_ITER = 5e3; % max # of iterations
TOL = 1e-3;     % tolerance
beta = 0.9;     % parameter (for augmenting lagrangian)
gamma = 0.99;
precondition = false;
rnd_permute = true; % This would have no effect anyways
verb = true;
