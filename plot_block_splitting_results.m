%% Load Experimental Results for Block Splitting in Primal and IP Primal
load('test_admm_primal_ip_block_split.mat')
load('test_admm_primal_block_split.mat')

%% Plot The Results (# of iterations vs. # of blocks)

% Plot the generic primal with block splitting
figure
subplot(2,2,1)
title('Primal')
plot_errorbar_param_conv(result(:,1),num_blocks_range, ...
                {'Sequential', 'Rand Permute'}, [0,10000], '# of Blocks')

subplot(2,2,2)
title('Primal With Preconditioning')
plot_errorbar_param_conv(result(:,2),num_blocks_range, ...
                {'Sequential', 'Rand Permute'}, [0,10000], '# of Blocks')

% Plot the interior point primal with block splitting
subplot(2,2,3)
title('Interior Point Primal')
plot_errorbar_param_conv(result_ip(:,1),num_blocks_range, ...
                {'Sequential', 'Rand Permute'}, [0,10000], '# of Blocks')

subplot(2,2,4)
title('Interior Point Primal With Preconditioning')
plot_errorbar_param_conv(result_ip(:,2),num_blocks_range, ...
                {'Sequential', 'Rand Permute'}, [0,10000], '# of Blocks')


