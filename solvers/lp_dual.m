function [ opt_val, x_hist, y_opt, s_opt, err_hist ] = lp_dual( c, A, b, MAX_ITER, TOL, beta, ...
    gamma, precondition, BLOCKS, rnd_permute_y_update, seed, verbose)
% lp_dual  A dual ADMM solver for linear programs. Supports 
%   block splitting, random and non-random variable updates, 
%   preconditioning, and interior point approach.
%
%   MAX_ITER (integer) maximum number of iterations before terminating
%   TOL (number) problem tolerance, i.e. convergence occurs when 
%       ||A*x-b|| < TOL
%   beta (number) The beta parameter used in the augmented Lagrangian
%   gamma (number) The gamma parameter for updating the interior point 
%       barrier parameter. If not using interior point, then set gamma<=0
%   precondition (string) The type of preconditioning to apply if any. 
%       'none' means not to apply preconditiong, 'standard' means to apply 
%       the standard inv(A*A') preconditioning, 'ichol' means to apply
%       incomplete Cholesky preconditioning.
%   BLOCKS (integer) The number of blocks to use for block splitting
%   rnd_permute_x1_update (boolean) If true, use random update order for
%       x1. Otherwise, update x1 sequentially
%   seed (integer) The random seed (for reproducibility)
%   verbose (optional boolean) Whether to run in verbose mode


switch nargin 
    case 11
        verbose = false;
    case 12
        % verbose argument was passed as a parameter
    otherwise
        error('Wrong number of inputs');
end

if gamma > 0
    % Interior point algorithm should be used
    int_pt = true;
    
    % Set the initial mu value
    mu = 1;
else
    int_pt = false;
end

if verbose
    fprintf('---------------------------------------------------\n')
    fprintf('Solving LP with Primal ADMM with block splitting\n')
    if rnd_permute_y_update
        fprintf('  Using random permutation on updates\n')
    else
        fprintf('  Using sequential updates\n')
    end
    
    if int_pt
        fprintf('  Using interior point method.\n')
    else
        fprintf('  NOT using interior point method.\n')
    end
end


if length(BLOCKS) == 1 % only the number of blocks specified
    NUM_BLOCKS = BLOCKS;
    if verbose
        disp(['  Only specified ',num2str(NUM_BLOCKS),' blocks to be splitted evenly']);
    end
else  % the block assignment specified
    NUM_BLOCKS = max(BLOCKS);
    if verbose
        disp(['  Splitting into ',num2str(NUM_BLOCKS),' blocks according to block assignment']);
    end
end

% Apply preconditioning if it was specified
switch precondition
    case 'standard'
        if verbose
            fprintf('  Using standard pre-conditioning\n')
        end
        AAT_inv_sqrt = sqrtm(inv(A * A'));
        b = AAT_inv_sqrt * b;
        A = AAT_inv_sqrt * A;
    case 'ichol'
        if verbose
            fprintf('  Using Cholesky pre-conditioning\n')
        end
        A_ichol = full(ichol(sparse(A * A'))); % ichol expects a sparse matrix
        b = A_ichol * b;
        A = A_ichol * A;
    otherwise
        if verbose
            fprintf('  NOT using pre-conditioning\n')
        end
end

% random initilization
rng(seed)

[m, n] = size(A);

% Initialize decision variables
y = zeros(m,1);
s = ones(n, 1);
x = -rand(n, 1);

% Split data into blocks
y_blocks = split_blocks(y, BLOCKS, 'vertical');
A_blocks = split_blocks(A, BLOCKS, 'vertical');
b_blocks = split_blocks(b, BLOCKS, 'vertical');

% Compute inverses on smaller matrices
AAT_inv_blocks = cell(NUM_BLOCKS, 1);
for i=1:NUM_BLOCKS
    A_cur = A_blocks{i};
    AAT_inv_blocks{i} = inv(A_cur*A_cur');
end

% Precompute matrix products for each Ai, Aj block of A
AiAjT_blocks = cell(NUM_BLOCKS^2, 1);
for i = 1: NUM_BLOCKS
    A_i = A_blocks{i};
    for j = 1:NUM_BLOCKS
        if i ~= j
            A_j = A_blocks{j};
            AiAjT_blocks{i,j} = A_i * A_j';
        end
    end
end

% history of errors at each iteration
err_hist = [];
x_hist = [];

for t=1:MAX_ITER
    
    % Determine the update order for the y blocks
    if rnd_permute_y_update 
        % Update in random order
        y_update_order = randperm(NUM_BLOCKS);
    else
        % Update in lexicographical order
        y_update_order = 1:NUM_BLOCKS;
    end
    
    % Update each y block
    for i = y_update_order
        A_cur = A_blocks{i};
        
        % Compute sum of cross terms
        cross_terms_sum = zeros(size(y_blocks{i}));
        for j=1:NUM_BLOCKS
            if i ~= j
                cross_terms_sum = cross_terms_sum + AiAjT_blocks{i,j} * y_blocks{j};
            end
        end
        
        y_blocks{i} = AAT_inv_blocks{i} * ((1/beta) * (A_cur * x + b_blocks{i}) ...
            - cross_terms_sum - A_cur * (s-c));
    end
    
    ATy = zeros(size(x)); % A * x1 (needed for updating y)
    for i=1:NUM_BLOCKS
        ATy = ATy + A_blocks{i}'* y_blocks{i};
    end
    
    if int_pt
        % update s
        cAy = c -  ATy;
        s = 1/(2*beta) * (beta*cAy + x + sqrt(beta^2*cAy.^2 + 2*beta*cAy.*x + 4*beta*mu + x.^2));   

        % update x
        x = x - beta * (ATy + s - c);

        % update mu
        mu = mu*gamma;
    else
        s = c - ATy + 1/beta * x;
        s = s .* (s > 0);
        x = x - beta * (ATy + s - c);
    end
        
    % Compute error and update history
    abs_err = norm(A*x + b);
    err_hist = [err_hist abs_err];
    x_hist = [x_hist x];
    % Early stopping condition
    if abs_err < TOL
        if verbose
            fprintf('Converged at step %d \n', t)
        end
        break
    end
end

if abs_err >= TOL 
     fprintf('Failed to converge after %d steps.\n', MAX_ITER)                 
end

% Return the optimal solution and objective value
x_opt = - x;
y_opt = cell2mat(y_blocks);
s_opt = s;
opt_val = c' * x_opt;

if verbose
    fprintf('Optimal Objective Value: %f \n', opt_val)
end


end


