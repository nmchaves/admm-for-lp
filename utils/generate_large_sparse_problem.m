function [c, A, b, opt_val, blocks] = generate_large_sparse_problem(m,n,type,density,n_blocks,seed,verbose)
%
% minimize t(c) * x
% subject to:
%   A * x = b
%   x >= 0
%
% m : # of equations
% n : # of variables

switch nargin
    case 4
        seed = 0;
        n_blocks = 5;
        verbose = false;
    case 5
        seed = 0;
        verbose = false;
    case 6
        verbose = false;
    case 7
        verbose = true;
    otherwise
        error('Wrong number of inputs');
end

rng(seed) % random seed for re-producability
%% create A based on type specification
if strcmp(type,'random')
    n_blocks = 1;
    A = sprand(m,n,density); % uniformly random sparse matrix
    blocks = ones(n,1);
elseif strcmp(type,'block')
    A = sparse(m,n);
    m_i = round(m/(n_blocks+1));
    n_i = round(n/n_blocks);
    blocks = zeros(n,1);
    for i = 1:n_blocks
        row_idx = ((i-1)*m_i+1):(i*m_i);
        if i == n_blocks
            col_idx = ((i-1)*n_i+1):n;
        else
            col_idx = ((i-1)*n_i+1):(i*n_i);
        end
        blocks(col_idx) = i;
        A(row_idx,col_idx) = sprand(length(row_idx),length(col_idx),density);
    end
    row_idx = (n_blocks*m_i+1):m;
    A(row_idx,:) = sprand(length(row_idx),n,density);
else
    error('Problem type not recognized!')
end

%%
if verbose
    disp(['Generating feasible and bounded problem with m = ', ...
        num2str(m),', n = ',num2str(n),'.'])
    imagesc(A)
end


%%
c = rand(n,1);  % so the problem is bounded
x0 = rand(n,1); % non-negative feasible solution
b = A * x0;

%% Compute the Optimal Solution
disp('Running linprog solver...')
[opt_x, opt_val] = linprog(c,[],[],A,b,zeros(n,1));
disp(['linprog optval : ', num2str(opt_val)])

%% (Optional) Compute the Optimal Solution with CVX
% disp('Running cvx solver...')
% cvx_begin quiet
%     variable x(n)
%     minimize(c'*x)
%     subject to
%         A * x == b
%         x >= 0
% cvx_end
% cvx_x = x;
% disp(['cvx optval : ', num2str(cvx_optval)])
end