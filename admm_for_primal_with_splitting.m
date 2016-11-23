% random seed for re-producability
rng('default')

% # of equations
m = 20; 

% # of constraints
n = 100; 

% Beta parameter (for augmenting lagrangian). Set randomly between 0 and 1
beta = rand();  

% Cost (must be nonnegative)
c = rand(n,1);

A = randn(m, n);
b = randn(m,1);

% Maximum # of iterations to run
MAX_ITER = 1e4;

% Whether or not to apply preconditioning to A and b
precondition = false;

% Tolerance (stop early if the error is less than this)
TOL = 1e-5;

if precondition
    AAT_inv_sqrt = sqrt(inv(A * A')) * A;
    b = sqrt(inv(A * A')) * b;
    A = AAT_inv_sqrt;
end

% Initialize y
y = zeros(m,1);

% Initialize s
s = ones(n, 1);

% Initialize x1 randomly (doesn't need to be positive).
x1 = randn(n, 1);

% Initialize x2 randomly (must be nonnegative). 
x2 = rand(n, 1);

% Split data into 2 blocks
midpoint = floor(n/2);
x1_1 = x1(1:midpoint);
x1_2 = x1(midpoint+1:end);
A_1 = A(:,1:midpoint);
A_2 = A(:,midpoint+1:end);

x2_1 = x2(1:midpoint);
x2_2 = x2(midpoint+1:end);
c_1 = c(1:midpoint);
c_2 = c(midpoint+1:end);
s_1 = s(1:midpoint);
s_2 = s(midpoint+1:end);

% Compute inverses on smaller matrices
ATA_plus_I_inv_1 = inv(A_1'*A_1 + eye(size(A_1,2)));
ATA_plus_I_inv_2 = inv(A_2'*A_2 + eye(size(A_2,2)));

% Precompute matrix products
A1TA2 = A_1' * A_2;
A2TA1 = A_2' * A_1;

% history of errors at each iteration
error_history = [];

for i=1:MAX_ITER
    
    % x1_1 update.
    x1_1 = ATA_plus_I_inv_1 * ((1/beta)*A_1'*y + (1/beta)*s_1 - (1/beta)*c_1 + A_1'*b + x2_1 - A1TA2 * x1_2);
    
    % x1_2 update. 
    x1_2 = ATA_plus_I_inv_2 * ((1/beta)*A_2'*y + (1/beta)*s_2 - (1/beta)*c_2 + A_2'*b + x2_2 - A2TA1 * x1_1);
    
    x1_full = [x1_1; x1_2];
    
    % x2 update
    x2 = max(x1_full - (1/beta)*s, 0);
    x2_1 = x2(1:midpoint);
    x2_2 = x2(midpoint+1:end);
    
    % y update
    y = y - beta * (A * x1_full - b);
    
    % s update
    s = s - beta * (x1_full - x2);
    s_1 = s(1:midpoint);
    s_2 = s(midpoint+1:end);
    
    abs_err = norm(A*x1_full - b);
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
opt_obj = c' * x1_full;
fprintf('Optimal Objective Value: %f \n', opt_obj)
