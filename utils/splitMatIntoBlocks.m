function [ blocks_as_cell_arr ] = splitMatIntoBlocks( mat, num_blocks, split_direction )
    
    blocks_as_cell_arr = cell(num_blocks, 1);

    is_vertical_split = strcmp(split_direction, 'vertical');
    if is_vertical_split
        block_size = floor(size(mat, 1) / num_blocks);
    else
        block_size = floor(size(mat, 2) / num_blocks);
    end

    for i=1:num_blocks
        first = (i-1) * block_size + 1;
        
        % Set size of last block to be remainder of the matrix
        if i==num_blocks
            if is_vertical_split
                last = size(mat,1);
            else
                last = size(mat,2);
            end
        else
            last = first + block_size - 1;
        end
        
        if is_vertical_split
            blocks_as_cell_arr{i} = mat(first:last,:);
        else
            blocks_as_cell_arr{i} = mat(:,first:last); 
        end           
    end
  
end

