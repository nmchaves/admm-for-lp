function [opt_val, x_opt, y_opt, s_opt, err_hist] = lp_dual_ip_admm(c, A, b, MAX_ITER, TOL, beta, gamma, precondition, seed, verbose)
% admm_lp_primal  
%
%   See also SUM, PLUS.

switch nargin 
    case 9
        verbose = false;
    case 10
        verbose = true;
    otherwise
        error('Wrong number of inputs');
end

if verbose
    fprintf('------------------------------------------\n')
    fprintf('Solving LP with Dual Interior Point ADMM\n')
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
mu = 1;

AAt_inv = inv(A * A');
error_history = []; % history of errors at each iteration

for i=1:MAX_ITER
    % update equations
    y = AAt_inv * (-A * (s - c) + 1/beta * (A * x + b));
    cAy = c - A' * y;
    s = 1/(2*beta) * (beta*cAy + x + sqrt(beta^2*cAy.^2 + 2*beta*cAy.*x + 4*beta*mu + x.^2));   
    x = x - beta * (A' * y + s - c);  
    mu = gamma * mu;
    
    % early stopping condition
    abs_err = norm(A * x + b);
    error_history = [error_history abs_err];
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