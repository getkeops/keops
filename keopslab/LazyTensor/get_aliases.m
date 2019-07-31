        function aliases = get_aliases(vars,dim_i,dim_j)
            nvars = length(vars);
            aliases = cell(1,nvars);
            for k=1:nvars
                sz = size(vars{k}.array);
                maxdim = max(dim_i,dim_j);
                if length(sz) < maxdim
                    sz = [sz,ones(1,maxdim-length(sz))];
                end
                keops_dim = prod(sz(1:(min(dim_i,dim_j)-1)));
                if sz(dim_i)==1
                    if sz(dim_j)==1
                        vartype = 'Pm';
                    else
                        vartype = 'Vj';
                    end
                elseif sz(dim_j)==1
                    vartype = 'Vi';
                else
                    error('incorrect shape for LazyTensor')
                end
                aliases{k} = ['v',num2str(k),' = ',vartype,'(',num2str(keops_dim),')'];
            end
        end
                
        
