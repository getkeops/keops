function [F,Fname] = keops_kernel(varargin)
% Defines a kernel convolution function based on a formula
% arguments are strings defining variables and formula
%
% Examples:
%
% - Define and test a function that computes for each i the sum over j
% of the square of the scalar products of xi and yj (both 3d vectors)
% F = keops_kernel('x=Vi(3)','y=Vj(3)','Sum_Reduction(Square((x,y)),0)');
% x = rand(3,2000);
% y = rand(3,5000);
% res = F(x,y);
%
% - Define and test the convolution with a Gauss kernel i.e. the sum
% over j of e^(lambda*||xi-yj||^2)beta_j (xi,yj, beta_j 3d vectors):
% F = keops_kernel('x=Vi(3)','y=Vj(3)','beta=Vj(3)','lambda=Pm(1)','Sum_Reduction(Exp(lambda*SqNorm2(x-y))*beta,0)');
% x = rand(3,2000);
% y = rand(3,5000);
% beta = rand(3,5000);
% lambda = .25;
% res = F(x,y,beta,lambda);

Nargin = nargin;

% get verbosity level to force compilation
[~,~,~,verbosity,use_cuda_if_possible] = default_options();

try
    gpuArray(1);
    canUseGPU = true;
catch
    canUseGPU = false;
end

% last arg is optional structure to override default tags
if isstruct(varargin{end})
    options = varargin{end};
    Nargin = Nargin-1;
    varargin = varargin(1:Nargin);
else
    options=struct;
end
% tagCpuGpu=0 means convolution on Cpu, tagCpuGpu=1 means convolution on Gpu, tagCpuGpu=2 means convolution on Gpu from device data
if use_cuda_if_possible && canUseGPU
    options = set_default_option(options, 'tagCpuGpu', 1);
else
    options = set_default_option(options, 'tagCpuGpu', 0);
end
% tag1D2D=0 means 1D Gpu scheme, tag1D2D=1 means 2D Gpu scheme
options = set_default_option(options,'tag1D2D',0);
% device_id is id of GPU device in case several GPUs can be used
options = set_default_option(options,'device_id',0);

% detect formula and aliases from inputs. Formula should be the only string
% without '=' character.
if isempty(strfind(varargin{Nargin},'='))
    % formula is in last position, after the list of aliases
    formula = varargin{Nargin};
    aliases = varargin(1:(Nargin-1));
elseif isempty(strfind(varargin{1},'='))
    % formula is in first position, before the list of aliases
    formula = varargin{1};
    aliases = varargin(2:Nargin);
else
    error('Incorrect inputs')
end

% for backward compability : if formula does not specify the type of
% reduction, we assume the user uses the old syntax where only summation was possible
% and summation over i or j was specified via an optional flag
if isempty(strfind(formula,'Reduction('))
    if isfield(options,'tagIJ')
        tagIJ = options.tagIJ;
    else
        tagIJ = 0;
    end
    formula = ['Sum_Reduction(',formula,',',num2str(tagIJ),')'];
end

% accuracy options
options = set_accuracy_options(options,formula);

% sumoutput is an optional tag (0 or 1) to tell wether we must further sum the
% output in the end. This is used when taking derivatives with respect to
% parameter variables (see grad function)
options = set_default_option(options,'sumoutput',0);

% from the string inputs we form the code which will be added to the source cpp/cu file, and the string used to encode the file name
[CodeVars, indijp] = format_var_aliase(aliases);

% we use a hash to shorten string and avoid special characters in the filename
hash = string2hash(lower(compile_formula(CodeVars, formula, 'gros_bidon', options, 'no_compile')));
Fname = ['keops', hash];
mex_name = [Fname, '.', mexext];

if ~(exist(mex_name,'file') == 3) || (verbosity == 1)
    compile_formula(CodeVars, formula, hash, options);
end

% return function handler
F = @Eval;

% the evaluation function
    function out = Eval(varargin)
        if nargin==0
            out = feval(Fname);
        else
			nx = size(varargin{indijp(1)},2);
			ny = size(varargin{indijp(2)},2);
            out = feval(Fname, options.tagCpuGpu, options.tag1D2D, options.device_id, nx, ny, varargin{:});
            if options.sumoutput
                out = sum(out,2); % '2' because we sum with respect to index, not dimension !
            end
        end
    end
	
% numvars is the number of input arguments of the formula.
% numvars is used by function Grad because taking gradients introduces new
% variables whose dimensions are unknown at the matlab level.
% Here we guess the value of numvars from the number of aliases and the
% number of variables actually present in the formula (which is obtained by
% calling the formula with no argument). In some special
% cases this is not correct and the value must be manually set.
options = set_default_option(options,'numvars',max(length(aliases),F()));

end
