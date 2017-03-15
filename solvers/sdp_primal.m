function [ opt_val, x_opt, y_opt, s_opt, err_hist, beta ] = sdp_primal( c, A, b, MAX_ITER, TOL, beta, seed)
% lp_primal  A primal ADMM solver for SDP problems. 
%
%   MAX_ITER (integer) maximum number of iterations before terminating
%   TOL (number) problem tolerance, i.e. convergence occurs when 
%       ||A*x-b|| < TOL
%   beta (number) The beta parameter used in the augmented Lagrangian
%   seed (integer) The random seed (for reproducibility)
%

% random initilization
rng(seed)

[~, m] = size(A); 
[n,~] = size(A{1}); 

% Initialize decision variables
y = zeros(m, 1); % Initialize legrange multiplier. 
s = eye(n, n); % Initialize lagrange multiplier. Needs to be PSD.
x1 = eye(n, n); % Initialize x1 (just needs to be symmetric, not necessarily PSD)

x2 = eye(n, n); % Initialize x2 (just needs to be PSD, symmetric)
iter_term_no = MAX_ITER;

bigeye = eye(n^2);
ATA = zeros(n^2);

for j=1:m
    for k=1:n
        for l=1:n
            ATA((k-1)*n+l,:) = ATA((k-1)*n+l,:) + A{j}(:)'*(A{j}(k,l));
        end
    end
end

% If necessary (i.e. the user passed a negative beta value), compute the beta guess.
if beta < 0
    beta = (1.0*trace(ATA)) / n^2;
end

ATA_plus_I_inv = inv(ATA + bigeye);

bb = frobBlockMultTranspose(A,b,m,n);

err_hist = [];
x_hist = [];

for i=1:MAX_ITER
    yy = frobBlockMultTranspose(A,y,m,n);

    x1(:) = ATA_plus_I_inv*(yy(:)/beta + -c(:)/beta + s(:)/beta + x2(:) + bb(:));

    x2rough = x1-s*1/beta;    
    [V,D] = eig(x2rough);
    D = diag(bsxfun(@max,zeros(n,1),diag(D)));   
    x2 = V*D*inv(V);

    currFrobProd = frobBlockMult(A,x1,m);
    
    y = y - beta*(currFrobProd-b);

    s = s - beta*(x1-x2);
    
    abs_err = norm(currFrobProd-b);

    err_hist = [err_hist abs_err];
    x_hist = [x_hist x1(:)];

    if abs_err < TOL
        iter_term_no = i;
        break;
    end
end
x_opt = x1;
s_opt = s;
y_opt = y;
opt_val = c(:)' * x_hist(:, end);


function y = frobBlockMult(A,x,m)
y = zeros(m,1);
for j=1:m
    y(j) = (A{j}(:))'*x(:);
end
end

function y = frobBlockMultTranspose(A,x,m,n)
y = zeros(n,n);
for j=1:m
    y = y + A{j}*x(j);
end
end

end
