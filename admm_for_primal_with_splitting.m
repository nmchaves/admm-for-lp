% random seed for re-producability
rng('default')

% # of equations
m = 20; 

% # of constraints
n = 100; 

% Number of blocks to split the problem into
NUM_BLOCKS = 6;

% Whether or not to randomly permute the x1 update order
rnd_permute_x1_update = false;

% Beta parameter (for augmenting lagrangian). Set randomly between 0 and 1
beta = rand();  

% Cost (must be nonnegative)
c = rand(n,1);

A = randn(m, n);
b = randn(m,1);

% Maximum # of iterations to run
MAX_ITER = 1e3;

% Whether or not to apply preconditioning to A and b
precondition = false;

% Tolerance (stop early if the error is less than this)
TOL = 1e-4;

if precondition
    AAT_inv_sqrt = sqrt(inv(A * A')) * A;
    b = sqrt(inv(A * A')) * b;
    A = AAT_inv_sqrt;
end

% Initialize y
y = zeros(m, 1);

% Initialize s
s = ones(n, 1);

% Initialize x1 randomly (doesn't need to be positive).
x1 = randn(n, 1);

% Initialize x2 randomly (must be nonnegative). 
x2 = rand(n, 1);

% Split data into blocks
x1_blocks = splitMatIntoBlocks(x1, NUM_BLOCKS, 'vertical');
A_blocks = splitMatIntoBlocks(A, NUM_BLOCKS, 'horizontal');
x2_blocks = splitMatIntoBlocks(x2, NUM_BLOCKS, 'vertical');
c_blocks = splitMatIntoBlocks(c, NUM_BLOCKS, 'vertical');
s_blocks = splitMatIntoBlocks(s, NUM_BLOCKS, 'vertical');

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
error_history = [];

for i=1:MAX_ITER
    
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
opt_obj = 0;
for i=1:NUM_BLOCKS
    opt_obj = opt_obj + c_blocks{i}' * x1_blocks{i};
end
fprintf('Optimal Objective Value: %f \n', opt_obj)
