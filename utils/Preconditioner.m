classdef Preconditioner
    %Preconditioner A class to represent the ADMM preconditioning type
    properties
        type
        opts
    end
    
    methods
        function obj = Preconditioner(type, opts)
            obj.type = type;
            obj.opts = opts;
        end 
        
        function [A_pre, b_pre] = apply(obj, A, b)
            [A_pre, b_pre] = precondition(A, b, obj.type, obj.opts);
        end
        
        function title = toTitle(obj)
            switch obj.type
                case 'none'
                    title = 'Not Preconditioned';
                case 'standard'
                    title = '(A*A^{T})^{-1/2}';
                case 'ichol'
                    switch obj.opts.type
                        case 'nofill'
                            title = 'ichol';
                        case 'ict'
                            title = strcat('ichol, droptol=', num2str(obj.opts.droptol));
                    end
                    
            end
        end
    end 
end

