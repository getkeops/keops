function [F,fname] = Kernel(varargin)
% Defines a kernel convolution function based on a formula
% arguments are strings defining variables and formula
%
% Examples:
%
% - Define and test a function that computes for each i the sum over j
% of the square of the scalar products of xi and yj (both 3d vectors)
% F = Kernel('x=Vx(0,3)','y=Vy(1,3)','Square((x,y))');
% x = rand(3,2000);
% y = rand(3,5000);
% res = F(x,y);
%
% - Define and test the convolution with a Gauss kernel i.e. the sum
% over j of e^(lambda*||xi-yj||^2)beta_j (xi,yj, beta_j 3d vectors):
% F = Kernel('x=Vx(0,3)','y=Vy(1,3)','beta=Vy(2,3)','lambda=Pm(0)','Exp(Cst(lambda)*SqNorm2(x-y))*beta');
% x = rand(3,2000);
% y = rand(3,5000);
% beta = rand(3,5000);
% lambda = .25;
% res = F(x,y,beta,lambda);
%
% - Define and test the gradient of the previous function with respect
% to the xi :
% F = Kernel('x=Vx(0,3)','y=Vy(1,3)','beta=Vy(2,3)','eta=Vx(3,3)','lambda=Pm(0)',...
%           'Grad(Exp(Cst(lambda)*SqNorm2(x-y))*beta,x,eta)');
% x = rand(3,2000);
% y = rand(3,5000);
% beta = rand(3,5000);
% eta = rand(3,2000);
% lambda = .25;
% res = F(x,y,beta,eta,lambda);

build_dir = '../build/';
cur_dir = pwd;


Nargin = nargin;

% last arg is optional structure to override default tags
if isstruct(varargin{end})
    options = varargin{end};
    Nargin = Nargin-1;
    varargin = varargin(1:Nargin);
else
    options=struct;
end
% tagIJ=0 means sum over j, tagIj=1 means sum over j
options = setoptions(options,'tagIJ',0);
% tagCpuGpu=0 means convolution on Cpu, tagCpuGpu=1 means convolution on Gpu, tagCpuGpu=2 means convolution on Gpu from device data
options = setoptions(options,'tagCpuGpu',0);
% tag1D2D=0 means 1D Gpu scheme, tag1D2D=1 means 2D Gpu scheme
options = setoptions(options,'tag1D2D',1);



% from the string inputs we form the code which will be added to the source cpp/cu file, and the string used to encode the file name
formula = varargin{Nargin};
[CodeVars,indxy ] = format_var_aliase(varargin(1:(Nargin-1)));

% we use a hash to shorten string and avoid special characters in the filename
Fname = string2hash(lower([CodeVars,formula]));
mex_name = [Fname,'.',mexext];

if ~(exist(mex_name,'file')==3)
    buildFormula(CodeVars,formula,Fname,mex_name,build_dir,cur_dir)
end

% return function handler
F = @Eval;

% the evaluation function
function out = Eval(varargin)
    nx = size(varargin{indxy(1)},2);
    ny = size(varargin{indxy(2)},2);
    out = feval(Fname,nx,ny,options.tagIJ,options.tagCpuGpu,...
        options.tag1D2D,varargin{:});
end

end



function [left,right] = sepeqstr(str)
% get string before and after equal sign
pos = find(str=='=');
if isempty(pos)
    pos = 0;
end
left = str(1:pos-1);
right = str(pos+1:end);
end




function [var_aliases,indxy] =format_var_aliase(var_options)
% format var_aliases to pass option to cmake
var_aliases = '';
indxy = [-1,-1]; % indxy will be used to calculate nx and ny from input variables

for k=1:length(var_options)

    [varname,vartype] = sepeqstr(var_options{k});
    if ~isempty(varname)
        var_aliases = [var_aliases,'decltype(',vartype,') ',varname,';'];
    end
    
    % analysing vartype : ex 'Vx(2,4)' means variable of type x
    % at position 2 and with dimension 4. Here we are
    % interested in the type and position, so 2nd and 4th
    % characters
    type = vartype(2);
    pos = vartype(4);
    if type=='x' && indxy(1)==-1
        indxy(1) = str2num(pos)+1;
    elseif type=='y' && indxy(2)==-1
        indxy(2) = str2num(pos)+1;
    end
end
end


function testbuild = buildFormula(code1, code2, filename, mex_name, build_dir, cur_dir)

    disp('Formula is not compiled yet ; compiling...')

    % I do not have a better option to set working dir...
    cd(build_dir)
    try
        [~,out0] =mysystemcall(['/usr/bin/cmake ../.. -DVAR_ALIASES="',code1,'" -DFORMULA_OBJ="',code2,'" -DUSENEWSYNTAX=TRUE -D__TYPE__=float -Dmex_name="../',filename,'"' ])
        [~,out1] = mysystemcall(['make mex_cpp'])
    catch
        warning('comp pb')
    end
    % ...comming back to curent directory
    cd(cur_dir)


    if (exist(mex_name,'file')==3)
        disp('Compilation succeeded')
    else
        error('Compilation failed')
    end
end

function [status,cmdout] = mysystemcall(command)
    [status,cmdout] = system([command])
    if cmdout(end)==char(10) || cmdout(end)==char(13)
        cmdout = cmdout(1:end-1);
    end
end
