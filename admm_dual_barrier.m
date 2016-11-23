clear;clc;close all
%% generate problem
rng(0)
generate_feasible_problem

%% initializtion
x = - rand(n, 1);
y = zeros(m,1);
s = ones(n, 1);

% problem parameters
MAX_ITER = 1e3; % max # of iterations
TOL = 1e-4;     % Tolerance
beta = 0.9;     % parameter (for augmenting lagrangian)
mu = 1;
gamma = 0.1;
% beta = rand(); 
precondition = false;

%% solve problem with ADMM
error_history = [];

if precondition
    disp('Using pre-conditioning')
    AAT_inv_sqrt = sqrtm(inv(A * A')) * A;
    b = sqrtm(inv(A * A')) * b;
    A = AAT_inv_sqrt;
end

AAt_inv = inv(A * A');


for i=1:MAX_ITER
    % update equations
    y = AAt_inv * (-A * (s - c) + 1/beta * (A * x + b));
    % s = c - A' * y + 1/beta * x;
    % s = s .* (s > 0); 
    cAy = c - A' * y;
    s = 1/(2*beta) * (beta*cAy + x + sqrt(beta^2*cAy.^2 + 2*beta*cAy.*x + 4*beta*mu + x.^2));
    
    x = x - beta * (A' * y + s - c);
    
    mu = gamma * mu;
    
    abs_err = norm(A * x + b);
    error_history = [error_history abs_err];
    % early stopping condition
    if abs_err < TOL
        fprintf('Converged at step %d \n', i)
        break
    end
end

x_opt = - x;
y_opt = y;
s_opt = s;

figure(1)
plot(error_history)
xlabel('Iteration')
ylabel('Abs Error: Norm(A*x1-b)')

% Optimal Objective value
opt_obj = c' * x_opt;
fprintf('Optimal Objective Value: %f \n', opt_obj)
