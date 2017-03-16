clear;clc;close all
seed = 0;

%% generate problem
m = 500;
n = 1000;
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