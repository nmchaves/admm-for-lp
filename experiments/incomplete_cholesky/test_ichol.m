clear;clc;close all
seed = 0;

%% parameters
m = 100;
n = 500;
MAX_ITER = 2e3; % max # of iterations
TOL = 1e-3;     % Tolerance
beta = 0.9;     % parameter (for augmenting lagrangian)
gamma = -1;     % don't use interior point
preconditioners = [
    Preconditioner('none', struct()), ...
    Preconditioner('standard', struct()), ...
    Preconditioner('ichol', struct('type', 'nofill')), ...
    Preconditioner('ichol', struct('type', 'ict', 'droptol', 1e-3)), ...
    Preconditioner('ichol', struct('type', 'ict', 'droptol', 0.1))
];

N = 1; % # number of problems to solve
corr_tol = 0.01; % Tolerance for correctness
num_blocks_range = [1, 5, 10]; %, 15, 20 % # of blocks to use for each splitting experiment
rnd_perm = true;
verb = true;

% Variables for plotting
for n_idx=1:length(num_blocks_range)
    legend_obj{n_idx} = strcat('B=',num2str(num_blocks_range(n_idx)));
end

colors = {'k','g','m','b','c','r'};

assert(length(colors) >= length(num_blocks_range), ...
    'Add more colors. There needs to be a color to use when plotting each block size');

%% Experiment with various configurations (primal)
figure

for i_prob = 1:N
    prob_seed = i_prob-1;
    disp(['Problem ',num2str(i_prob)])
    [c, A, b, opt_val] = generate_linprog_problem(m,n,prob_seed, true);
    
    for p_idx = 1:length(preconditioners)
        precond = preconditioners(p_idx);
        subplot(1, length(preconditioners), p_idx)

        [A_pre, b_pre] = precond.apply(A, b);

        b_idx = 0; % index of current block number
        for num_blocks = num_blocks_range     
            b_idx = b_idx+1;
            [ov,~,~,~,eh] = lp_primal(c, A_pre, b_pre, MAX_ITER, TOL, ...
                    beta, gamma, num_blocks, rnd_perm, seed, verb);
            
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
            xlim([0, MAX_ITER])
            ylim([1e-3, 1e2])
            hold on
            
        end
        legend(legend_obj)
        title(precond.toTitle())
    end
end


%% Experiment with various configurations (dual)
figure

for i_prob = 1:N
    prob_seed = i_prob-1;
    disp(['Problem ',num2str(i_prob)])
    [c, A, b, opt_val] = generate_linprog_problem(m,n,prob_seed,true);
    
    for p_idx = 1:length(preconditioners)
        precond = preconditioners(p_idx);
        subplot(1, length(preconditioners), p_idx)

        [A_pre, b_pre] = precond.apply(A, b);

        b_idx = 0; % index of current block number
        for num_blocks = num_blocks_range     
            b_idx = b_idx+1;
            [ov,~,~,~,eh] = lp_dual(c, A_pre, b_pre, MAX_ITER, TOL, ...
                    beta, gamma, num_blocks, rnd_perm, seed, verb);
            
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
            xlim([0, MAX_ITER])
            ylim([1e-3, 1e2])
            hold on
            
        end
        legend(legend_obj)
        title(precond.toTitle())
    end
end
