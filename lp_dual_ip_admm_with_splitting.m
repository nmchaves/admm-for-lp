function [ opt_val, x_opt, y_opt, s_opt, err_hist ] = lp_dual_ip_admm_with_splitting(...
    c, A, b, MAX_ITER, TOL, beta, gamma, precondition, BLOCKS, rnd_permute_x1_update, seed, verbose)

switch nargin 
    case 11
        verbose = false;
    case 12
        fprintf('\n');
        verbose = true;
    otherwise
        error('Wrong number of inputs');
end


if verbose
    fprintf('---------------------------------------------------\n')
    fprintf('Solving LP with Dual IP ADMM with block splitting\n')
    if (rnd_permute_x1_update)
        fprintf('  Using random permutation on updates\n')
    else
        fprintf('  Using sequential updates\n')
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
        disp(['  Splitting into ',num2str(NUM_BLOCKS),'blocks according to block assignment']);
    end
end

% preconditioning
if verbose
    if precondition
        fprintf('  Using pre-conditioning\n')
    else 
        fprintf('  NOT using pre-conditioning\n')
    end
end 

if precondition
    
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

% Split data into blocks
y_blocks = split_blocks(y, NUM_BLOCKS, 'vertical');
A_blocks = split_blocks(A, NUM_BLOCKS, 'vertical');
b_blocks = split_blocks(b, NUM_BLOCKS, 'vertical');

% Precompute inverses on smaller matrices
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

for t = 1:MAX_ITER
    % Determine the update order for the x1 blocks
    if rnd_permute_x1_update % Update in random order
        y_update_order = randperm(NUM_BLOCKS);
    else % Update in lexicographical order
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
    
    % update s
    cAy = c -  ATy;
    s = 1/(2*beta) * (beta*cAy + x + sqrt(beta^2*cAy.^2 + 2*beta*cAy.*x + 4*beta*mu + x.^2));   

    % update x
    x = x - beta * (ATy + s - c);
    
    % update mu
    mu = mu*gamma;
    
    % Compute error and update history
    abs_err = norm(A * x + b);
    err_hist = [err_hist abs_err];
    
    % Early stopping condition
    if abs_err < TOL
        if verbose
            fprintf('Converged at step %d \n', t)
        end
        break
    end
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

