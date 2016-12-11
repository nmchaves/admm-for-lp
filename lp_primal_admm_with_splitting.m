function [ opt_val, x_opt, y_opt, s_opt, err_hist ] = lp_primal_admm_with_splitting( c, A, b, MAX_ITER, TOL, beta, ...
    precondition, BLOCKS, rnd_permute_x1_update, seed, verbose)


switch nargin 
    case 10
        verbose = false;
    case 11
        fprintf('\n');
        verbose = true;
    otherwise
        error('Wrong number of inputs');
end

if verbose
    fprintf('---------------------------------------------------\n')
    fprintf('Solving LP with Primal ADMM with block splitting\n')
    if (rnd_permute_x1_update)
        fprintf('  Using random permutation on updates\n')
    else
        fprintf('  Using sequential updates\n')
    end
end


if length(BLOCKS) == 1 % only the number of blocks specified
    NUM_BLOCKS = BLOCKS;
    if verbose
        disp(['Only specified ',num2str(NUM_BLOCKS),' blocks to be splitted evenly']);
    end
else  % the block assignment specified
    NUM_BLOCKS = max(BLOCKS);
    if verbose
        disp(['Splitting into ',num2str(NUM_BLOCKS),' blocks according to block assignment']);
    end
end

[m, n] = size(A);

if verbose
    if precondition
        fprintf('  Using pre-conditioning\n')
    else 
        fprintf('  NOT using pre-conditioning\n')
    end
end 

if precondition
    AAT_inv_sqrt = sqrtm(inv(A * A'));
    b = AAT_inv_sqrt * b;
    A = AAT_inv_sqrt * A;
end

% random initilization
rng(seed)

% Initialize y
y = zeros(m, 1);

% Initialize s
s = ones(n, 1);

% Initialize x1 randomly (doesn't need to be positive).
x1 = randn(n, 1);

% Initialize x2 randomly (must be nonnegative).
x2 = rand(n, 1);

% Split data into blocks
x1_blocks = split_blocks(x1, BLOCKS, 'vertical');
A_blocks = split_blocks(A, BLOCKS, 'horizontal');
x2_blocks = split_blocks(x2, BLOCKS, 'vertical');
c_blocks = split_blocks(c, BLOCKS, 'vertical');
s_blocks = split_blocks(s, BLOCKS, 'vertical');

% Compute inverses on smaller matrices
ATA_plus_I_inv_blocks = cell(NUM_BLOCKS, 1);
for i=1:NUM_BLOCKS
    A_cur = A_blocks{i};
    ATA_plus_I_inv_blocks{i} = inv(A_cur'*A_cur + eye(size(A_cur,2)));
end

% Precompute matrix products for each Ai, Aj block of A
AiTAj_blocks = cell(NUM_BLOCKS^2, 1);
for i=1:NUM_BLOCKS
    A_i = A_blocks{i};
    for j=1:NUM_BLOCKS
        if i ~= j
            A_j = A_blocks{j};
            AiTAj_blocks{i,j} = A_i' * A_j;
        end
    end
end

% history of errors at each iteration
err_hist = [];

for t=1:MAX_ITER
    
    % Determine the update order for the x1 blocks
    if rnd_permute_x1_update
        % Update in random order
        x1_update_order = randperm(NUM_BLOCKS);
    else
        % Update in lexicographical order
        x1_update_order = 1:NUM_BLOCKS;
    end
    
    % Update each x1 block
    for i=x1_update_order
        A_cur = A_blocks{i};
        
        % Compute sum of cross terms
        cross_terms_sum = zeros(size(x1_blocks{i}));
        for j=1:NUM_BLOCKS
            if i ~= j
                cross_terms_sum = cross_terms_sum + AiTAj_blocks{i,j} * x1_blocks{j};
            end
        end
        
        x1_blocks{i} = ATA_plus_I_inv_blocks{i} * ((1/beta)*A_cur'*y + ...
            (1/beta)*s_blocks{i} - (1/beta)*c_blocks{i} + A_cur'*b + ...
            x2_blocks{i} - cross_terms_sum);
    end
    
    % update x2 1 block at a time
    for i=1:NUM_BLOCKS
        x2_blocks{i} = max(x1_blocks{i} - (1/beta)*s_blocks{i}, 0);
    end
    
    % Update y using the new x1 blocks
    Ax1 = zeros(size(y)); % A * x1 (needed for updating y)
    for i=1:NUM_BLOCKS
        Ax1 = Ax1 + A_blocks{i} * x1_blocks{i};
    end
    y = y - beta * (Ax1 - b);
    
    % update s 1 block at a time
    for i=1:NUM_BLOCKS
        s_blocks{i} = s_blocks{i} - beta * (x1_blocks{i} - x2_blocks{i});
    end
        
    % Compute error and update history
    abs_err = norm(Ax1 - b);
    err_hist = [err_hist abs_err];
    % Early stopping condition
    if abs_err < TOL
        %fprintf('Converged at step %d \n', t)
        if verbose
            fprintf('Converged at step %d \n', t)
        end
        break
    end
end

% Return the optimal solution and objective value
x_opt = cell2mat(x1_blocks);
opt_val = c' * x_opt;
y_opt = y;
s_opt = cell2mat(s_blocks);

if verbose
    fprintf('Optimal Objective Value: %f \n', opt_val)
end


end

