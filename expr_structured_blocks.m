clear;clc;close all;
addpath(genpath('figure_util'))

m = 100;
n = 200;
sp = 0.8; % sparsity
n_blocks = 10;
prob_seed = 0;

figure('Position', [100, 100, 600, 300]);
[c, A, b, opt_val,blocks] = generate_large_sparse_problem(m,n,'block',sp,n_blocks,prob_seed,true);
save_current_figure('figures/struct_prob','high','-png');


%% parameters
MAX_ITER = 5e3; % max # of iterations
TOL = 1e-3;     % tolerance
beta = 0.9;     % parameter (for augmenting lagrangian)
gamma = 0.99;
precondition = true;
rnd_permute = true; % This would have no effect anyways
verb = true;
seed = 0;

%%
methods = {'primal','primal ip'};
NUM_BLOCKS =  [1,2,3,4,5,10];
figure('Position', [100, 100, 600, 200]);

A = full(A);

for j = 1:length(methods)
    method = methods{j};
    oveh = cell(length(NUM_BLOCKS),2);
    for i = 1:2
        if i == 1
            block_in = 1;
        else
            block_in = blocks;
        end
        tic
        switch method
            case 'primal'
                [oveh{i,1},~,~,~,oveh{i,2}] = lp_primal_admm_with_splitting(c, A, b, MAX_ITER, TOL, beta, ...
                    precondition, block_in, rnd_permute, seed, verb);
            case 'dual'
                [oveh{i,1},~,~,~,oveh{i,2}] = lp_dual_admm_with_splitting(c, A, b, MAX_ITER, TOL, beta, ...
                    precondition, block_in, rnd_permute, seed, verb);
            case 'primal ip'
                [oveh{i,1},~,~,~,oveh{i,2}] = lp_primal_ip_admm_with_splitting(c, A, b, MAX_ITER, TOL, beta, gamma,...
                    precondition, block_in, rnd_permute, seed, verb);
            case 'dual ip'
                [oveh{i,1},~,~,~,oveh{i,2}] = lp_dual_ip_admm_with_splitting(c, A, b, MAX_ITER, TOL, beta, gamma,...
                    precondition, block_in, rnd_permute, seed, verb);
            otherwise
                error('method not recognized')
        end
        
        toc
        if (abs(opt_val-oveh{i,1}) > 1e-1)
            warning('The objective value is not close enough')
        end
    end
    if (length(oveh{i,2})== MAX_ITER)
        warning('The result did not converge')
    end
    
    %% Plot
    subplot(1,2,j)
    colors = {'k','b'};
    for i = 1:2
        eh = oveh{i,2};
        semilogy(1:length(eh), eh, colors{i})
        axis([1,1500,TOL,1e7])
        grid on
        box on
        hold on
    end
    xlabel('iteration')
    ylabel('absolute error')
    legend({'single block','multi-block'})
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
save_current_figure(['figures/struct_',fname],'high','-png');
