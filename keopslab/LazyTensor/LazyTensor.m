


classdef LazyTensor
   
    properties (Access = private)
        formula_needs_parenthesis = 0;
    end
    
    properties
        vars
        shape
        formula
        nbatchdims = 0;
    end
      
    methods
        
        function obj = LazyTensor(x,nbatchdims)
            if isa(x,'LazyTensor')
                obj = x;
            else
                obj.vars = {ArrayHandle(x)};
                obj.shape = size(x);
                obj.formula = 'v1';
                if nargin == 2
                    obj.nbatchdims = nbatchdims;
                end
            end
        end   
        
        function obj = binary_op(x,y,str_id,shape_policy,is_operator)
            if nargin < 5
                is_operator = 0;
            end
            x_ = LazyTensor(x);
            obj = x_;
            y = LazyTensor(y);
            obj.shape = shape_policy(x_.shape,y.shape);
            obj.vars = [ x_.vars, y.vars ];
            if is_operator          
                if obj.formula_needs_parenthesis
                    prefix = '(';
                    mid_str = [')',str_id,'('];     
                    suffix = ')';
                else
                    prefix = '';
                    mid_str = str_id;     
                    suffix = '';  
                    obj.formula_needs_parenthesis = 1;
                end
            else
                prefix = [str_id,'('];  
                mid_str = ',';
                suffix = ')';  
            end                
            obj.formula = [prefix, x_.formula, mid_str, ...
                offsetvarnames(y.formula,length(y.vars), ...
                length(x_.vars)), suffix];
        end
        
        function obj = unary_op(x,str_id,shape_policy,is_operator,opt_arg)
            if nargin < 5
                opt_arg = [];
            end
            if nargin < 4
                is_operator = 0;
            end
            obj = LazyTensor(x);
            obj.shape = shape_policy(x.shape);
            obj.vars = x.vars;
            if is_operator   
                if obj.formula_needs_parenthesis
                    prefix = [str_id,'('];  
                    suffix = ')';  
                else
                    prefix = str_id;
                    suffix = '';
                    obj.formula_needs_parenthesis = 1;
                end
            else
                prefix = [str_id,'(']; 
                suffix = ')';   
                obj.formula_needs_parenthesis = 0;
            end
            if ~isempty(opt_arg)
                suffix = [',',opt_arg,suffix];
            end
            obj.formula = [prefix, x.formula,suffix];
        end
           
        function obj = plus(x,y)
            obj = binary_op(x,y,'+',@broadcast_shape,1);
        end
        
        function obj = minus(x,y)
            obj = binary_op(x,y,'-',@broadcast_shape,1);
        end
        
        function obj = rdivide(x,y)
            obj = binary_op(x,y,'/',@broadcast_shape,1);
        end
        
        function obj = mrdivide(x,y)
            if isnumeric(y) && numel(y)==1
                obj = rdivide(x,y);
            else
                error('not implemented');
            end
        end
        
        function obj = times(x,y)
            obj = binary_op(x,y,'*',@broadcast_shape,1);
        end
        
        function obj = mtimes(x,y)
            if (isnumeric(x) && numel(x)==1) || (isnumeric(y) && numel(y)==1)
                obj = times(x,y);
            else
                error('not implemented');
            end
        end
        
        function obj = dot(x,y)
            obj = binary_op(x,y,'|',@same_to_one,1);
        end
        
        function obj = uminus(x)
            obj = unary_op(x,'-',@same_shape,1);
        end
        
        function obj = exp(x)
            obj = unary_op(x,'Exp',@same_shape);
        end
        
        function obj = power(x,y)
            if isnumeric(y) && numel(y)==1
                if y==2
                    obj = unary_op(x,'Square',@same_shape);
                elseif y==.5
                    obj = unary_op(x,'Sqrt',@same_shape);
                elseif y==-.5
                    obj = unary_op(x,'Sqrt',@same_shape);
                elseif y==-1
                    obj = unary_op(x,'Inv',@same_shape);
                elseif y==1
                    obj = x;
                elseif mod(y,1)==0
                    obj = unary_op(x,'Pow',@same_shape,0,num2str(y));
                else
                    y = LazyTensor(y);
                end
            end
            if isa(y,'LazyTensor')
                obj = binary_op(x,y,'Powf',@broadcast_shape);
            end
        end
        
        function out = reduction(x,str_id,dim)
            ndims = length(x.shape);
            switch dim
                case ndims-x.nbatchdims
                    aliases = get_aliases(x.vars,dim,dim-1);
                case ndims-x.nbatchdims-1
                    aliases = get_aliases(x.vars,dim,dim+1);
                otherwise
                        error('incorrect dimension for reduction')
            end
            F = keops_kernel(aliases{:},...
                 [str_id,'_Reduction(',x.formula,',0)']);
            vars_input_str = ['x.vars{1}.array'];
            for k=2:length(x.vars)
                vars_input_str = [vars_input_str,',x.vars{',num2str(k),'}.array']; 
            end
            out = eval(['F(',vars_input_str,')']);
        end
        
        function out = sum_reduction(x,dim)
            out = reduction(x,'Sum',dim);
        end
        
    end
    
end

