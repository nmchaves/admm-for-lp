clear;clc;close all

%% parameters
m = 20;
n = 100;

MAX_ITER = 1e4; % max # of iterations
TOL = 1e-4;     % Tolerance
beta = 0.9;     % parameter (for augmenting lagrangian)
seed = 0; % solver seed

N = 1; % # number of problems to solve
corr_tol = 0.01; % Tolerance for correctness
num_blocks_range = [1, 5, 10, 15, 20]; % # of blocks to use for each splitting experiment
verb = true;

%% Experiment with various block sizes

for i_prob = 1:N
    prob_seed = i_prob-1;
    disp(' ')
    disp(['Problem ',num2str(i_prob)])
    [c, A, b, opt_val] = generate_linprog_problem(m,n,prob_seed);
    for i_num_blocks = 1:length(num_blocks_range)
        num_blocks = num_blocks_range(i_num_blocks);
        for rnd_perm = [true, false]  
            for precond = [true, false]
                if rnd_perm
                    disp('Rand perm')
                else
                    disp('Sequential')
                end
                if precond
                    disp('Preconditioning')
                else
                    disp('Not conditioned')
                end
                [ov,~,~,~,eh] = lp_primal_admm_with_splitting(c, A, b, MAX_ITER, TOL, beta, ...
                                        precond, num_blocks, rnd_perm, seed, verb);

                if abs(ov - opt_val) > corr_tol
                    disp(['Block size: ',num2str(num_blocks)])
                    if precond
                        disp('Using Preconditioning')
                    end
                    disp(['Converged at:', num2str(length(eh))])
                    warning('Incorrect Solution!')
                    % store the number of steps used for convergence
                    result{rnd_perm+1, precond+1}(i_num_blocks, i_prob) = -1;
                else 
                    % store the number of steps used for convergence
                    result{rnd_perm+1, precond+1}(i_num_blocks, i_prob) = length(eh);
                end
            end    
        end
    end
end

save('test_admm_primal_block_split.mat','result','num_blocks_range')

%% Plot the results

figure
subplot(1,2,1)
title('without pre-conditioning')
plot_errorbar_param_conv(result(:,1),num_blocks_range, ...
                {'Sequential', 'Rand Permute'}, [0,10000], '# of Blocks')

subplot(1,2,2)
title('with preconditioning')
plot_errorbar_param_conv(result(:,2),num_blocks_range, ...
                {'Sequential', 'Rand Permute'}, [0,10000], '# of Blocks')