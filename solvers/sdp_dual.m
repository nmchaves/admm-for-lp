function [ opt_val, x_opt, y_opt, s_opt, err_hist, beta ] = sdp_primal( c, A, b, MAX_ITER, TOL, beta, seed)
% lp_primal  A primal ADMM solver for SDP problems. Supports 
%   block splitting, random and non-random variable updates, 
%   preconditioning, and interior point approach.
%   
%   The original primal problem is
%       min c'*x
%       s.t. A*x=b
%
%   MAX_ITER (integer) maximum number of iterations before terminating
%   TOL (number) problem tolerance, i.e. convergence occurs when 
%       ||A*x-b|| < TOL
%   beta (number) The beta parameter used in the augmented Lagrangian

% random initilization
rng(seed)

[~, m] = size(A); %'r' will not be used
[n,~] = size(A{1}); %'r' will not be used

% Initialize decision variables
y = zeros(m, 1); %Initialize legrange multiplier. Doesn't need to be negative.
s = eye(n, n); %initialize lagrange multiplier. Needs to be positive definite.
x = eye(n, n); % Initialize x randomly (doesn't need to be positive).
iter_term_no = MAX_ITER;

ATA = zeros(m,m);

for k=1:m
    for l=1:m
        ATA(k,l) = A{k}(:)'*A{l}(:);
    end
end

% If necessary (i.e. the user passed a negative beta value), compute the beta guess.
if beta < 0
    beta = (1.0*trace(ATA)) / n;
end

ATAInv = inv(ATA);

bb = frobBlockMultTranspose(A,b,m,n);
cc = frobBlockMult(A,c,m);

err_hist = [];
y_hist = [];

for i=1:MAX_ITER
    
    yy = frobBlockMultTranspose(A,y,m,n);
    xx = frobBlockMult(A,x,m);
    ss = frobBlockMult(A,s,m);
    y = ATAInv*(1/beta*(xx+b)-ss+cc);

    yy2 = frobBlockMultTranspose(A,y,m,n);
    srough = 1/beta*x-yy2+c;
    
    [V,D] = eig(srough);
    D = diag(bsxfun(@max,zeros(n,1),diag(D)));
    
    s = V*D*inv(V);
    
    x = x-beta*(yy2+s-c);

    abs_err = norm(yy2+s-c);

    err_hist = [err_hist abs_err];
    y_hist = [y_hist y];

    if abs_err < TOL
        iter_term_no = i;
        break;
    end
end

x_opt = -x; %for some reason x is coming out negative of what it should be. Flipping it back.
s_opt = s;
y_opt = y;
opt_val = b' * y;


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
