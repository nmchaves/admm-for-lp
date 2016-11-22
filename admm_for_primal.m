% # of constraints
m = 20; 

% # of equations
n = 100; 

% Maximum # of iterations to run
MAX_ITER = 1e3;

% Whether or not to apply preconditioning to A and b
precondition = false;

% Tolerance (stop early if the error is less than this)
TOL = 1e-3;

% Beta parameter (for augmenting lagrangian). Set randomly between 0 and 1
beta = rand();  

% Cost (must be nonnegative)
c = rand(n,1);

A = randn(m, n);
b = randn(m,1);

if precondition
    AAT_inv_sqrt = sqrt(inv(A * A')) * A;
end

% Initialize y
y = zeros(m,1);

% Initialize s
s = ones(n, 1);

% Initialize x1 randomly (doesn't need to be positive).
x1 = randn(n, 1);

% Initialize x2 randomly (must be nonnegative). 
x2 = rand(n, 1);

ATA_plus_I_inv = inv(A'*A + eye(size(A,2)));

 % history of errors at each iteration
error_history = [];

for i=1:MAX_ITER
    % x1 update
    x1 = ATA_plus_I_inv * ((1/beta)*A'*y + (1/beta)*s - (1/beta)*c + A'*b + x2);

    % x2 update
    x2 = max(x1 - (1/beta)*s, 0);
    
    % y update
    y = y - beta * (A * x1 - b);
    
    % s update
    s = s - beta * (x1 - x2);
    
    abs_err = norm(A*x1 - b);
    error_history = [error_history abs_err];
    
    % Early stopping condition
    if abs_err < TOL
        fprintf('Converged at step %d \n', i)
        break
    end
    
end

figure(1)
plot(error_history)
xlabel('Iteration')
ylabel('Abs Error: Norm(A*x1-b)')

% Optimal Objective value
opt_obj = c' * x1;
fprintf('Optimal Objective Value: %f \n', opt_obj)