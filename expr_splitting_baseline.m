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
rnd_permute = false; % This would have no effect anyways
verb = true;

%%
methods = {'primal','dual','primal ip','dual ip'};
NUM_BLOCKS =  [1,2,3,4,5,10];
figure('Position', [100, 100, 600, 450]);

for j = 1:length(methods)
    method = methods{j};
    oveh = cell(length(NUM_BLOCKS),2);
    for i = 1:length(NUM_BLOCKS)
        tic
        switch method
            case 'primal'
                [oveh{i,1},~,~,~,oveh{i,2}] = lp_primal_admm_with_splitting(c, A, b, MAX_ITER, TOL, beta, ...
                    precondition, NUM_BLOCKS(i), rnd_permute, seed, verb);
            case 'dual'
                [oveh{i,1},~,~,~,oveh{i,2}] = lp_dual_admm_with_splitting(c, A, b, MAX_ITER, TOL, beta, ...
                    precondition, NUM_BLOCKS(i), rnd_permute, seed, verb);
            case 'primal ip'
                [oveh{i,1},~,~,~,oveh{i,2}] = lp_primal_ip_admm_with_splitting(c, A, b, MAX_ITER, TOL, beta, gamma,...
                    precondition, NUM_BLOCKS(i), rnd_permute, seed, verb);
            case 'dual ip'
                [oveh{i,1},~,~,~,oveh{i,2}] = lp_dual_ip_admm_with_splitting(c, A, b, MAX_ITER, TOL, beta, gamma,...
                    precondition, NUM_BLOCKS(i), rnd_permute, seed, verb);
            otherwise
                error('method not recognized')
        end
        
        toc
        if (abs(opt_val-oveh{i,1}) > 1e-3)
            warning('The objective value is not close enough')
        end
    end
    if (length(oveh{i,2})== MAX_ITER)
        warning('The result did not converge')
    end
    
    %% Plot
    subplot(2,2,j)
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
    title(method)
end
if precondition && rnd_permute
    fname = 'precond_rndperm';
elseif precondition && ~rnd_permute
    fname = 'precond_norndperm';
elseif ~precondition && rnd_permute
    fname = 'noprecond_rndperm';
else
    fname = 'noprecond_norndperm';
end
save_current_figure(['figures/',fname],'high','-png');
