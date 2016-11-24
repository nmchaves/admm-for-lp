function [opt_val, x_opt, y_opt, s_opt, err_hist] = lp_primal_ip_admm(c, A, b, MAX_ITER, TOL, beta, gamma, precondition, seed)
% admm_lp_primal  
%
%   See also SUM, PLUS.

if (nargin ~= 9)
    error('Wrong number of inputs');
else
    fprintf('------------------------------------------\n')
    fprintf('Solving LP with Primal Interior Point ADMM\n')
end

% preconditioning 
if precondition
    fprintf('NOTE: using pre-conditioning\n')
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
mu = 1;

ATA_plus_I_inv = inv(A'*A + eye(size(A,2)));
error_history = []; % history of errors at each iteration

for i=1:MAX_ITER
    % variable updates
    x1 = ATA_plus_I_inv * ((1/beta)*A'*y + (1/beta)*s - (1/beta)*c + A'*b + x2);
    x2 = 1/(2*beta)* (beta * x1 - s + sqrt(beta^2*x1.^2 - 2*beta*s.*x1 + 4*beta*mu + s.^2)); 
    y = y - beta * (A * x1 - b);
    s = s - beta * (x1 - x2);
    mu = mu*gamma; % update mu
    
    abs_err = norm(A*x1 - b);
    error_history = [error_history abs_err]; 
    if abs_err < TOL
        fprintf('Converged at step %d \n', i)
        break
    end
end

opt_val = c' * x1;
x_opt = x1;
y_opt = y;
s_opt = s;
err_hist = error_history;

fprintf('Optimal Objective Value: %f \n', opt_val)
end