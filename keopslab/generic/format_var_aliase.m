function [var_aliases,indxyp] =format_var_aliase(var_options)
% format var_aliases to pass option to cmake
var_aliases = '';
indxyp = [-1,-1,-1]; % indxyp will be used to calculate nx, ny and np from input variables

for k=1:length(var_options)

    [varname,vartype] = sepeqstr(var_options{k});
    if ~isempty(varname)
        [vartype, pos] = vartype_cpp(vartype,k);
        var_aliases = [var_aliases,'decltype(',vartype,') ',varname,';'];
    end
    
    % analysing vartype : ex 'Vi(4)' means variable of type x
    %  with dimension 4. The position is given by the rank in
    % the function argument. Here we are interested in the type
    % and position, so 2nd and the variable 'pos'
    type = vartype(2);
    if type=='x' && indxyp(1)==-1
        indxyp(1) = pos;
    elseif type=='y' && indxyp(2)==-1
        indxyp(2) = pos;
    elseif type=='m' && indxyp(3)==-1
        indxyp(3) = pos;
    end
end
end


function [vartype, pos] = vartype_cpp(str,k)
% format the var aliases so that it has the
% form 'VarType(pos,dim)' where Vartype is 
% 'Pm', 'Vi', 'Vj', pos and dim are integer.

comma = find(str==',');
if isempty(comma) % assume 1 arg, ie we have Vi(3)
    vartype = [str(1:3),num2str(k-1),',',str(4:end)];
    pos = k;

else % assume 2 arg, ie we have Vi(0,3)
    vartype = str;
    pos  = str2num(str(4:(comma-1))) +1; %overwrite the position number
end
end
