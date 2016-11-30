function [blocks_as_cell_arr] = split_blocks(mat, blocks, split_direction, verbose)

% check inputs
switch nargin 
    case 3
        verbose = false;
    case 4
        verbose = verbose;
    otherwise
        error('Wrong number of inputs');
end

if length(blocks) == 1 % only the number of blocks specified
    n_blocks = blocks;
    is_block_given = false;
    if verbose
        disp(['Only specified ',num2str(n_blocks),' blocks to be splitted evenly']);
    end
else  % the block assignment specified
    is_block_given = true;
    n_blocks = max(blocks);
    if verbose
        disp(['Splitting into ',num2str(n_blocks),'blocks according to block assignment']);
    end
end


is_vertical_split = strcmp(split_direction, 'vertical');

if is_block_given % check if the dimensions match
    block_assignment = blocks;
    tot_len = length(block_assignment);
    if is_vertical_split
        if size(mat,1) ~= tot_len
            error('Mismatch matrix-block dimension.')
        end
    else
        if size(mat,2) ~= tot_len
            error('Mismatch matrix-block dimension.')
        end
    end
else % create a block vector based on even consequetive splits
    if is_vertical_split
        tot_len = size(mat,1);
    else
        tot_len = size(mat,2);
    end
    block_size = floor(tot_len/n_blocks);
    block_assignment = zeros(tot_len,1);
    for i = 1:n_blocks
        first = (i-1) * block_size + 1;
        % Set size of last block to be remainder of the matrix
        if i==n_blocks
            last = tot_len;
        else
            last = first + block_size - 1;
        end
        block_assignment(first:last) = i;
    end
end

blocks_as_cell_arr = cell(n_blocks, 1);

for i = 1:n_blocks
    sel_idx = (block_assignment == i);
    if is_vertical_split
        blocks_as_cell_arr{i} = mat(sel_idx,:);
    else
        blocks_as_cell_arr{i} = mat(:,sel_idx);
    end
end

end

