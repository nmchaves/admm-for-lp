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
preconditioners = {'none', 'standard', 'ichol'};

N = 1; % # number of problems to solve
corr_tol = 0.01; % Tolerance for correctness
num_blocks_range = [1, 5, 10]; %, 15, 20 % # of blocks to use for each splitting experiment
rnd_perm = true;
verb = true;

gamma = -1; % don't use interior point

% Variables for plotting
for n_idx=1:length(num_blocks_range)
    legend_obj{n_idx} = strcat('B=',num2str(num_blocks_range(n_idx)));
end

colors = {'k','g','m','b','c','r'};

assert(length(colors) >= length(num_blocks_range), ...
    'Add more colors. There needs to be a color to use when plotting each block size');

%% Experiment with various configurations (primal)
figure(1)

for i_prob = 1:N
    prob_seed = i_prob-1;
    disp(' ')
    disp(['Problem ',num2str(i_prob)])
    [c, A, b, opt_val] = generate_linprog_problem(m,n,prob_seed);
    p_idx = 0; % index of current preconditioner type
    for precond = preconditioners
        p_idx = p_idx + 1;
        subplot(1, length(preconditioners), p_idx)
       
        b_idx = 0; % index of current block number
        for num_blocks = num_blocks_range     
            b_idx = b_idx+1;
            [ov,~,~,~,eh] = lp_primal(c, A, b, MAX_ITER, TOL, beta, ...
                gamma, precond{1}, num_blocks, rnd_perm, seed, verb);
            
            if abs(ov - opt_val) > corr_tol
                warning('Incorrect Solution!')
                disp(['Block size was: ',num2str(num_blocks)])
                % store the number of steps used for convergence
                result{rnd_perm+1, p_idx}(num_blocks, i_prob) = -1;
            else
                % store the number of steps used for convergence
                result{rnd_perm+1, p_idx}(num_blocks, i_prob) = length(eh);
            end
     
            % Plot the error history
            semilogy(1:length(eh),eh, colors{b_idx})
            hold on
            
            
        end
        legend(legend_obj)
        title(preconditioners{p_idx})
    end
end


%% Experiment with various configurations (dual)
figure(2)

for i_prob = 1:N
    prob_seed = i_prob-1;
    disp(' ')
    disp(['Problem ',num2str(i_prob)])
    [c, A, b, opt_val] = generate_linprog_problem(m,n,prob_seed);
    p_idx = 0; % index of current preconditioner type
    for precond = preconditioners
        p_idx = p_idx + 1;
        subplot(1, length(preconditioners), p_idx)
       
        b_idx = 0; % index of current block number
        for num_blocks = num_blocks_range     
            b_idx = b_idx+1;
            [ov,~,~,~,eh] = lp_dual(c, A, b, MAX_ITER, TOL, beta, ...
                gamma, precond{1}, num_blocks, rnd_perm, seed, verb);
            
            if abs(ov - opt_val) > corr_tol
                warning('Incorrect Solution!')
                disp(['Block size was: ',num2str(num_blocks)])
                % store the number of steps used for convergence
                result{rnd_perm+1, p_idx}(num_blocks, i_prob) = -1;
            else
                % store the number of steps used for convergence
                result{rnd_perm+1, p_idx}(num_blocks, i_prob) = length(eh);
            end
     
            % Plot the error history
            semilogy(1:length(eh),eh, colors{b_idx})
            hold on
            
            
        end
        title(preconditioners{p_idx})
    end
end

%% Primal IP ADMM with 1 block (no splitting)
tic
NUM_BLOCKS = 1;
rnd_permute = true; % This would have no effect anyways
[ov1,~,~,~,eh1] = lp_primal(c, A, b, MAX_ITER, TOL, beta, gamma, ...
                                    precondition, NUM_BLOCKS, rnd_permute, seed, verb);
toc
%% Primal IP ADMM with 5 blocks
tic
NUM_BLOCKS = 5;
rnd_permute = true;
[ov2,~,~,~,eh2] = lp_primal(c, A, b, MAX_ITER, TOL, beta, gamma,...
                                    precondition, NUM_BLOCKS, rnd_permute, seed, verb);
toc                             
%% Primal IP ADMM with 10 blocks
tic
NUM_BLOCKS = 10;
rnd_permute = true;
[ov3,~,~,~,eh3] = lp_primal(c, A, b, MAX_ITER, TOL, beta,gamma, ...
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
[ov1,~,~,~,eh1] = lp_dual(c, A, b, MAX_ITER, TOL, beta, gamma, ...
                                    precondition, NUM_BLOCKS, rnd_permute, seed, verb);
toc
%% Dual IP ADMM with 5 blocks
tic
NUM_BLOCKS = 5;
rnd_permute = true;
[ov2,~,~,~,eh2] = lp_dual(c, A, b, MAX_ITER, TOL, beta, gamma,...
                                    precondition, NUM_BLOCKS, rnd_permute, seed, verb);
toc                             
%% Dual IP ADMM with 10 blocks
tic
NUM_BLOCKS = 10;
rnd_permute = true;
[ov3,~,~,~,eh3] = lp_dual(c, A, b, MAX_ITER, TOL, beta,gamma, ...
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


