function [c, A, b, opt_val] = generate_linprog_problem(m,n,seed)
%
% minimize t(c) * x
% subject to: 
%   A * x = b
%   x >= 0
%
% m : # of equations
% n : # of variables


rng(seed) % random seed for re-producability
%% Paramter Setting
c = rand(n,1);  % so the problem is bounded
A = randn(m, n);
x0 = rand(n,1); % non-negative
b = A * x0;

disp(['Generated feasible and bounded problem with m = ', ...
      num2str(m),', n = ',num2str(n),'.'])
  
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