function [opt_val, x_opt, y_opt, s_opt, err_hist] = lp_primal_admm(c, A, b, MAX_ITER, TOL, beta, precondition, seed, verbose)
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
    fprintf('Solving LP with Primal ADMM\n')
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
x1 = randn(n, 1);
x2 = rand(n, 1);

ATA_plus_I_inv = inv(A'*A + eye(size(A,2)));
error_history = []; % history of errors at each iteration

for i=1:MAX_ITER    
    % variable updates
    x1 = ATA_plus_I_inv * ((1/beta)*A'*y + (1/beta)*s - (1/beta)*c + A'*b + x2);
    x2 = max(x1 - (1/beta)*s, 0);
    y = y - beta * (A * x1 - b);
    s = s - beta * (x1 - x2);
    % store error and check stopping criteria
    abs_err = norm(A*x1 - b);
    error_history = [error_history abs_err];
    if abs_err < TOL
        if verbose
            fprintf('Converged at step %d \n', i)
        end
        break
    end
end


opt_val = c' * x1;
x_opt = x1;
y_opt = y;
s_opt = s;
err_hist = error_history;

if verbose
    fprintf('Optimal Objective Value: %f \n', opt_val)
end
end