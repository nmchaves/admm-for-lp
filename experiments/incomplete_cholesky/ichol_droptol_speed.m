clear;clc;close all
prob_seed = 0;
solver_seed = 0;

%% Specify which problem sizes and preconditioners to use

problem_sizes = [
    [10, 100]; ...
    [50, 500]; ...
    [100, 1000]; ...
];

n_problems = length(problem_sizes);
matlab_solver_times = zeros(n_problems, 1);

preconditioners = [
    Preconditioner('none', struct()), ...
    Preconditioner('standard', struct()), ...
    Preconditioner('ichol', struct('type', 'nofill')), ...
    Preconditioner('ichol', struct('type', 'ict', 'droptol', 1e-3)), ...
    Preconditioner('ichol', struct('type', 'ict', 'droptol', 0.01))
];
n_precond = length(preconditioners);
precond_times = zeros(n_problems, n_precond);
solver_times = zeros(n_problems, n_precond);

% Specify which preconditioner to actually use when solving each problem
% It would be too expensive to run all of the solver, and they will all
% run in a very similar amount of iterations after preconditioning anyways.
precond_solv_idx = 1;
precond_solver_times = zeros(n_problems, 1);

%% ADMM Solver Parameters
MAX_ITER = 1e4; % max # of iterations
TOL = 1e-3;     % Tolerance
beta = 0.9;     % parameter (for augmenting lagrangian)
gamma = -1; % don't use interior point

corr_tol = 0.01; % Tolerance for correctness
num_blocks_range = [1]; %, 15, 20 % # of blocks to use for each splitting experiment
rnd_perm = true;
verb = true;

%% Solve Problems and Measure Time
for prob_idx = 1:n_problems
    prob_size = problem_sizes(prob_idx,:);
    m = prob_size(1);
    n = prob_size(2);
    disp('----------------')
    disp(['Problem size: m=', num2str(m), ...
        ', n=', num2str(n)])
    
    %[c, A, b, ~, ~] = generate_linprog_problem(m, n , prob_seed, false);

    [c, A, b, ov, solv_time] = generate_linprog_problem(m, n , prob_seed, true);
    matlab_solver_times(prob_idx) = solv_time;
    
    for prec_idx = 1:n_precond
        precond = preconditioners(prec_idx);
        
        num_blocks = 1;
        
        tic;
        [A_pre, b_pre] = precond.apply(A, b);
        precond_times(prob_idx, prec_idx) = toc;
        
        if prec_idx == precond_solv_idx
        else
            tic;
            lp_dual(c, A_pre, b_pre, MAX_ITER, TOL, ...
                    beta, gamma, num_blocks, rnd_perm, solver_seed, verb);
        
            precond_solver_times(prob_idx) = toc;
        end
        
    end
    
end


%% Plot results
xtick_labels = {};
for prob_idx = 1:n_problems
    prob_size = problem_sizes(prob_idx,:);
    xtick_labels{prob_idx} = strcat(num2str(prob_size(1)), '\times', num2str(prob_size(2)));
end

legend_obj = {};
legend_idx = 1;
for prec_idx = 1:n_precond
    precond = preconditioners(prec_idx);
    if strcmp(precond.type, 'none')
        continue
    else
        legend_obj{legend_idx} = precond.toTitle();
        legend_idx = legend_idx + 1;
    end
end


figure
subplot(1, 2, 1)
for prec_idx = 1:n_precond
    precond = preconditioners(prec_idx);
    if strcmp(precond.type, 'none')
        continue
    end
      
    semilogy(precond_times(:, prec_idx))
    hold on
end
legend(legend_obj)
xlabel('Size of A')
ylabel('Preconditioning Time (sec)')

% Note: newer versions of MATLAB have a better syntax, but
% this should work with 2015b an up.
set(gca,'xtick',1:n_problems); 
set(gca,'xticklabel', xtick_labels);
set(gca, 'xticklabelrotation', 30);

% Same figure, but as percent of the approximate total solver time
subplot(1, 2, 2)
for prec_idx = 1:n_precond
    precond = preconditioners(prec_idx);
    if strcmp(precond.type, 'none')
        continue
    end
    
    p_times = precond_times(:, prec_idx);
    plot(100 * p_times ./ (precond_solver_times + p_times))
    hold on
end
legend(legend_obj)
xlabel('Size of A')
ylabel('Approximate % of Solver Time Spent on Preconditioning')
set(gca,'xtick',1:n_problems); 
set(gca,'xticklabel', xtick_labels);
set(gca, 'xticklabelrotation', 30);

%% todo: show 2 figures
%% fig a: precond time as pct of total exec time for
% each method for various problem sizes

%% fig b: 
% show abs exec tuime for each method for various
% problme sizes



%% generate problem
m = 400;
n = 800;
prob_seed = 0;
[c, A, b, opt_val] = generate_linprog_problem(m, n , prob_seed);


%% Test how long standard preconditioning takes
tic
sqrtm(inv(A * A'));
t_std_ms = toc;

%% Test how long ichol takes when not using drop.
tic
ichol(sparse(A * A'));
t_no_drop_ms = toc;

%% Test how long ichol takes for various drop tolerances
precond_opts = struct('type', 'ict');
droptols = logspace(-6, 0, 10);
times_ms = [];
for dt = droptols
    precond_opts.droptol = dt;
    tic
    ichol(sparse(A * A'), precond_opts);
    t = toc;
    times_ms = [times_ms t];
end

figure
semilogx(droptols, 1000*times_ms, 'b')
hold on

std_line = refline(0, 1000*t_std_ms);
std_line.Color = 'g';

no_drop_line = refline(0, 1000*t_no_drop_ms);
no_drop_line.Color = 'r';
xlabel('Drop Tolerance')
ylabel('Execution Time (ms)')