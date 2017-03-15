function [c, A, b, X] = generate_sdp_problem(m,n,seed)
% Generates params for an SDP problem.
%
% m : # of equations
% n : # of variables


rng(seed) % random seed for re-producability
%% Paramter Setting
c = rand(n,n);  % so the problem is bounded
c = .5*(c + c'); %WLOG we can make c symmetric, because x is symmetric.
b = zeros(m,1);
x0 = rand(n,n);
x1 = .5*(x0+x0');  %make x symetric
x2 = x1+abs(min(eig(x1)))+1; %make sure all eigenvalues are positive.
for k=1:m
    A{k} = randn(n, n);
    A{k} = .5*(A{k} + A{k}'); %WLOG we can make the "A"s symmetric, because x is symmetric.
    b(k) = sum(sum(A{k} .* x2));
end

cvx_begin
variable X(n,n) symmetric;
minimize trace(c*X);
subject to
for i=1:m
    trace(A{i}*X) == b(i);
end
X == semidefinite(n);
cvx_end
%disp(['Generated feasible and bounded problem with m = ', ...
%      num2str(m),', n = ',num2str(n),'.'])
  
%% Compute the Optimal Solution
%disp('Running linprog solver...')
%[opt_x, opt_val] = linprog(c,[],[],A,b,zeros(n,1));
%disp(['linprog optval : ', num2str(opt_val)])

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
