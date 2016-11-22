m = 20; % # of constraints
n = 100; % # of equations

% Tolerance (for stopping condition)
tol = 1e-3;

% Beta parameter (for augmenting lagrangian)
beta = rand();  % random between 0 and 1

A = randn(m, n);
b = randn(m,1);

% Initialize y
y = zeros(m,1);

% Initialize s
s = ones(n, 1);

% Initialize x1. TODO: init correctly
x1 = zeros(n, 1);

% Initialize x2. TODO: init correctly
x2 = zeros(n, 1);

ATA_plus_I_inv = inv(A'*A + eye(size(A,2)));

iter = 0;
err = inf; % TODO: compute error

while err > tol
    
    % x1 update
    x1 = ATA_plus_I_inv * ((1/b)*A'*y + (1/beta)*s - (1/beta)*c + A'*b + x2);

    % x2 update
    x2 = max(x1 - (1/beta)*s, 0);
    
    % y update
    y = y - beta * (A * x1 - b);
    
    % s update
    s = s - beta * (x1 - x2);
    
    % update error. TODO
    err = 0;
    
    iter = iter + 1;
end



