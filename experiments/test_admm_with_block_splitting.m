clear;clc;close all

%% parameters
m = 20;
n = 100;

MAX_ITER = 1e4; % max # of iterations
TOL = 1e-3;     % Convergence tolerance
beta = 0.9;     % parameter (for augmenting lagrangian)
seed = 1; % solver seed

N = 1; % # number of problems to solve
corr_tol = 0.01; % Tolerance for correctness
num_blocks_range = [1, 5, 10]; %, 15, 20 % # of blocks to use for each splitting experiment
verb = true;

gamma = -1; % don't use interior point

%% Experiment with various block sizes
for i_prob = 1:N
    prob_seed = i_prob-1;
    disp(' ')
    disp(['Problem ',num2str(i_prob)])
    [c, A, b, opt_val] = generate_linprog_problem(m,n,prob_seed);
    for num_blocks = num_blocks_range
        for rnd_perm = [true, false]  
            p = 1; % index of preconditiong type
            for precond = {'none', 'standard', 'ichol'} 
                [ov,~,~,~,eh] = lp_primal(c, A, b, MAX_ITER, TOL, beta, ...
                                 gamma, precond{1}, num_blocks, rnd_perm, seed, verb);

                if abs(ov - opt_val) > corr_tol
                    warning('Incorrect Solution!')
                    disp(['Block size was: ',num2str(num_blocks)])
                    % store the number of steps used for convergence
                    result{rnd_perm+1, p}(num_blocks, i_prob) = -1;
                else 
                    % store the number of steps used for convergence
                    result{rnd_perm+1, p}(num_blocks, i_prob) = length(eh);
                end
                p = p+1;
            end    
        end
    end
end

%save('test_admm_primal_block_split.mat','result','num_blocks_range')

%% Plot the results

figure
subplot(1,3,1)
title('no pre-conditioning')
plot_errorbar_param_conv(result(:,1),num_blocks_range, ...
                {'Sequential', 'Rand Permute'}, [0,MAX_ITER], '# of Blocks')

subplot(1,3,2)
title('standard preconditioning')
plot_errorbar_param_conv(result(:,2),num_blocks_range, ...
                {'Sequential', 'Rand Permute'}, [0,MAX_ITER], '# of Blocks')
            
subplot(1,3,3)
title('incomplete cholesky preconditioning')
plot_errorbar_param_conv(result(:,3),num_blocks_range, ...
                {'Sequential', 'Rand Permute'}, [0,MAX_ITER], '# of Blocks')