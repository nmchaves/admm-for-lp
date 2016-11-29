clear;clc;close all
%% generate problem
rng(2)
generate_feasible_problem

%% initializtion
x1 = randn(n, 1);
x2 = rand(n, 1);
y = zeros(m,1);
s = ones(n, 1);

% problem parameters
MAX_ITER = 1e3; % max # of iterations
TOL = 1e-5;     % Tolerance
beta = 0.9;     % parameter (for augmenting lagrangian)
mu = 1;
gamma = 0.1;
% beta = rand(); 
precondition = true;

%% solve problem with ADMM
error_history = [];

if precondition
    disp('Using pre-conditioning')
    AAT_inv_sqrt = sqrtm(inv(A * A')) * A;
    b = sqrtm(inv(A * A')) * b;
    A = AAT_inv_sqrt;
end

ATA_plus_I_inv = inv(A'*A + eye(size(A,2)));

 % history of errors at each iteration
error_history = [];

for i=1:MAX_ITER
    x1 = ATA_plus_I_inv * ((1/beta)*A'*y + (1/beta)*s - (1/beta)*c + A'*b + x2);
    % x2 = max(x1 - (1/beta)*s, 0);
    x2 = 1/(2*beta)* (beta * x1 - s + sqrt(beta^2*x1.^2 - 2*beta*s.*x1 + 4*beta*mu + s.^2));
    
    y = y - beta * (A * x1 - b);
    s = s - beta * (x1 - x2);
    
    
    mu = mu*gamma; % update mu
    
    abs_err = norm(A*x1 - b);
    error_history = [error_history abs_err]; 
    % Early stopping condition
    if abs_err < TOL
        fprintf('Converged at step %d \n', i)
        break
    end
end


x_opt = x1;
y_opt = y;
s_opt = s;

figure(1)
plot(error_history)
xlabel('Iteration')
ylabel('Abs Error: Norm(A*x1-b)')

% Optimal Objective value
opt_obj = c' * x_opt;
fprintf('Optimal Objective Value: %f \n', opt_obj)
