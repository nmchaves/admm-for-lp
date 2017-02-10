clear;clc;close all
addpath(genpath('figure_util'))
seed = 0;

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
rnd_permute = false;
verb = true;

%% 
NUM_BLOCKS = [2, 3, 10];
figure('Position', [100, 100, 600, 450]);
oveh = cell(length(NUM_BLOCKS),2);
x_hist = cell(length(NUM_BLOCKS), 1);
for i = 1:length(NUM_BLOCKS)
     [oveh{i,1},x_hist{i},~,~,oveh{i,2}] = lp_primal_admm_with_splitting(c, A, b, MAX_ITER, TOL, beta, ...
    precondition, NUM_BLOCKS(i), rnd_permute, seed, verb);
end


subplot(1,1,1)
colors = {'k','g','m','b','c','r'};
for i = 1:length(NUM_BLOCKS)
    eh = oveh{i,2};
    semilogy(1:length(eh), eh, colors{i})
    axis([1,MAX_ITER,TOL,1e2])
    grid on
    box on
    hold on
end
xlabel('iteration')
ylabel('absolute error')
legend(cellfun(@(x) ['B = ',num2str(x)], num2cell(NUM_BLOCKS), 'UniformOutput',false))
title('primal')

%% Look for evidence of cycling

x2block = x_hist{1}; % converging case. 1block
x3block = x_hist{2}; % non-converging case for 3 blocks
x10block = x_hist{3}; % non-converging case for 10 blocks

% Last 1000 x1 values for the 3 block case
x3blockLast1k = x2block(:, MAX_ITER-999:end);

% Variances of x1 values for non-conv case over last 1k iters
x3blockVar = var(x3blockLast1k')';

% Component of x1 that had the highest variance for 3 block case
[maxVar, idxMaxVar] = max(x3blockVar);

% Optimal solution for this max variance component
maxVarOptSoln = x2block(idxMaxVar, end);

% Plot the movement of this variable
figure(2)
title('yo')
subplot(1,2,1)
hold on
plot(x10block(idxMaxVar,:), char(colors(6)))
plot(x3block(idxMaxVar,:), char(colors(3)))
plot(x2block(idxMaxVar,:), char(colors(2)))
plot([1, MAX_ITER], [maxVarOptSoln, maxVarOptSoln], 'black')
ylim([-1, 1])
xlabel('iteration')
ylabel('absolute error')
%legend({'B=10','B=3', 'B=2'})
hold off

lastIterShow = 200;
subplot(1,2,2)
hold on
plot(x10block(idxMaxVar,1:lastIterShow), char(colors(6)))
plot(x3block(idxMaxVar,1:lastIterShow), char(colors(3)))
plot(x2block(idxMaxVar,:), char(colors(2)))
plot([1, lastIterShow], [maxVarOptSoln, maxVarOptSoln], 'black')
ylim([-0.2, 0.2])
xlim([1, lastIterShow])
xlabel('iteration')
ylabel('absolute error')
legend({'B=10','B=3', 'B=2'})
hold off
    
%% Save figure 
fname = 'variable_cycling_analysis';
save_current_figure(['figures/',fname],'high','-png');
    
    

