function [opt_val, x_opt, y_opt, s_opt, err_hist] = lp_dual_admm(c, A, b, MAX_ITER, TOL, beta, precondition, seed, verbose)
% admm_lp_primal
%
%   See also SUM, PLUS.

switch nargin
    case 8
        verbose = false;
    case 9
        verbose = true;
    otherwise
        error('Wrong number of inputs');
end
if verbose
    fprintf('------------------------------------\n')
    fprintf('Solving LP with Dual ADMM\n')
end

% preconditioning
if precondition
    if verbose
        fprintf('NOTE: using pre-conditioning\n')
    end
    AAT_inv_sqrt = sqrtm(inv(A * A')) * A;
    b = sqrtm(inv(A * A')) * b;
    A = AAT_inv_sqrt;
end

[m,n] = size(A);

% random initilization
rng(seed)
y = zeros(m,1);
s = ones(n, 1);
x = -rand(n, 1);

AAt_inv = inv(A * A');
error_history = []; % history of errors at each iteration

for i=1:MAX_ITER
    % update equations
    y = AAt_inv * (-A * (s - c) + 1/beta * (A * x + b));
    s = c - A' * y + 1/beta * x;
    s = s .* (s > 0);
    x = x - beta * (A' * y + s - c);
    % compute error for feasibility
    % abs_err = norm(A' * y + s - c);
    abs_err = norm(A * x + b);
    error_history = [error_history abs_err];
    % early stopping condition
    if abs_err < TOL
        if verbose
            
            fprintf('Converged at step %d \n', i)
        end
        break
    end
end

x_opt = - x;
y_opt = y;
s_opt = s;

opt_val = c' * x_opt;
err_hist = error_history;
if verbose
    fprintf('Optimal Objective Value: %f \n', opt_val)
end
end