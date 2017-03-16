function [ A_pre, b_pre ] = precondition(A, b, type, opts)
%PRECONDITION Apply preconditioning to a linear system.
%   Apply preconditioning to a linear system of the form:
%           Ax=b
%   type (string) The type of preconditioner. Must be one of:
%       'standard', 'ichol'
%   opts (struct) An optional argument to pass to the preconditioner.
%   returns The preconditioned A matrix and b vector.

    switch type
        case 'standard'
            AAT_inv_sqrt = sqrtm(inv(A * A'));
            b_pre = AAT_inv_sqrt * b;
            A_pre = AAT_inv_sqrt * A;
        case 'ichol'
            %opts.type = 'ict';
            %opts.droptol = 1e-1;
            %opts.type = 'nofill';
            %opts.michol = 'off';
            % also try (L*L')^(-1/2)
            A_ichol_left = ichol(sparse(A * A'), opts);
            A_ichol = inv(A_ichol_left);
            b_pre = A_ichol * b;
            A_pre = A_ichol * A;
        otherwise
            fprintf('NOT applying any pre-conditioning\n');
            b_pre = b;
            A_pre = A;
    end


end

