clear;clc;close all
addpath(genpath('figure_util'))
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
gamma = 0.1;
precondition = false;
rnd_permute = true; % This would have no effect anyways
verb = true;

%% Dual ADMM with 1 block (no splitting)

NUM_BLOCKS =  [1,5,10];
oveh = cell(length(NUM_BLOCKS),2);
for i = 1:length(NUM_BLOCKS)
    tic
    [oveh{i,1},~,~,~,oveh{i,2}] = lp_dual_admm_with_splitting(c, A, b, MAX_ITER, TOL, beta, ...
        precondition, NUM_BLOCKS(i), rnd_permute, seed, verb);
    toc
    if i == 1
        [ov,~,~,~,eh] = lp_dual_admm(c, A, b, MAX_ITER, TOL, beta, precondition, seed);
        if (norm(eh - oveh{i,2}) < 1e-5)
            disp('* PASS: one-block implementation is correct')
        else
            error('one-block implementation is incorrect')
        end
    else
        if (abs(opt_val-oveh{i,1}) > 1e-3)
            warning('The objective value is not close enough')
        end
    end
    if (length(oveh{i,2})== MAX_ITER)
        warning('The result did not converge')
    end
end

%% Plot
colors = {'r','g','b'};
figure('Position', [100, 100, 400, 200]);
for i = 1:length(NUM_BLOCKS)
    eh = oveh{i,2};
    semilogy(1:length(eh), eh, colors{i})
    hold on
end
xlabel('Iteration')
ylabel('Abs Error: ||A*x1-b||')
legend(cellfun(@(x) ['B = ',num2str(x)], num2cell(NUM_BLOCKS), 'UniformOutput',false))
if precondition && rnd_permute
    title('preconditioning, random perumutation')
    fname = 'dualadmm_precond_rndperm';
elseif precondition && ~rnd_permute
    fname = 'dualadmm_precond_norndperm';
    title('preconditioning, no random perumutation')
elseif ~precondition && rnd_permute
    fname = 'dualadmm_noprecond_rndperm';
    title('no preconditioning, random perumutation')
else
    fname = 'dualadmm_noprecond_norndperm';
    title('no preconditioning, no random perumutation')
end
save_current_figure(['figures/',fname],'low','-png');
