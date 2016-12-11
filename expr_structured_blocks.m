clear;clc;close all;

m = 100;
n = 200;
sp = 0.8; % sparsity
n_blocks = 10;
prob_seed = 0;

[c, A, b, opt_val,blocks] = generate_large_sparse_problem(m,n,'block',sp,n_blocks,prob_seed,true);
