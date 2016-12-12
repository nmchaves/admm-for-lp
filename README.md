# Alternating Direction Method of Multipliers (ADMM) for Linear Programming

This project was developed by Junjie (Jason) Zhu and Nico Chaves for Stanford MS&E 310 (Linear Programming). We implemented several novel configurations of the ADMM optimizaton method and ran several experiments. For a complete discussion on background, experiments, and results, please see our report in report/derivations.pdf.

## Problem Generation

- To generate a single feasible and bounded problem for testing, run:
```
m = 50;
n = 300;
prob_seed = 1;
[c, A, b, opt_val] = generate_linprog_problem(m, n , prob_seed);
```

Here the problem will have 50 constraints and 300 variables. The problem seed is merely for reproducibility. Note that generate_linprog_problem returns the optimal value of the LP (as computed by MATLAB's linprog function).

## Solver Functions

We developed 4 ADMM solvers: primal, interior point primal, dual, and interior point dual. You can specify arguments to each solver to use preconditioning and/or block splitting. If you choose to specify block splitting with > 2 blocks, then we strongly recommend setting the random permutation argument to true. As we showed in our report, the algorithms may not converge when there are > 2 blocks and you do not use random permutation. We implemented the 4 solvers in the following 4 MATLAB functions:

1) Primal
```
lp_primal_admm_with_splitting.m 
```

2) Dual
```
lp_dual_admm_with_splitting.m
```

3) Interior Point Primal 
```
lp_primal_ip_admm_with_splitting.m
```

4) Interior Point Dual
```
lp_dual_ip_admm_with_splitting.m
```

For example, the following code will run the primal ADMM solver with 3 blocks, randomly permuted updates, and no preconditioning. The function returns the LP optimal value, the value of x at each iteration, the optimal y and s values, and the absolute error ||Ax-b|| at each iteration.

```
beta = 0.9;
precondition = false;
rnd_permute = true;
num_blocks = 3;
verbose = true;
seed = 0; // for reproducibility
[opt_val, x_hist, y_opt, s_opt, err_hist] = lp_primal_admm_with_splitting(c, A, b, MAX_ITER, TOL, beta, ...
                    precondition, 3, rnd_permute, seed, verbose);
```

Files named expr\_\* contain the code for the experiments we ran. These may also be helpful if you want to see more examples of running the different solvers. 

## References
- ADMM Paper: http://web.stanford.edu/~boyd/papers/pdf/admm_distr_stats.pdf
