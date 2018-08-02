function [F,Fname] = Kernel(varargin)
% Defines a kernel convolution function based on a formula
% arguments are strings defining variables and formula
%
% Examples:
%
% - Define and test a function that computes for each i the sum over j
% of the square of the scalar products of xi and yj (both 3d vectors)
% F = Kernel('x=Vx(3)','y=Vy(3)','SumReduction(Square((x,y)),0)');
% x = rand(3,2000);
% y = rand(3,5000);
% res = F(x,y);
%
% - Define and test the convolution with a Gauss kernel i.e. the sum
% over j of e^(lambda*||xi-yj||^2)beta_j (xi,yj, beta_j 3d vectors):
% F = Kernel('x=Vx(3)','y=Vy(3)','beta=Vy(3)','lambda=Pm(1)','SumReduction(Exp(lambda*SqNorm2(x-y))*beta,0)');
% x = rand(3,2000);
% y = rand(3,5000);
% beta = rand(3,5000);
% lambda = .25;
% res = F(x,y,beta,lambda);
%
% - Define and test the gradient of the previous function with respect
% to the xi :
% F = Kernel('x=Vx(3)','y=Vy(3)','beta=Vy(3)','eta=Vx(3)','lambda=Pm(1)',...
%           'SumReduction(Grad(Exp(lambda*SqNorm2(x-y))*beta,x,eta),0)');
% x = rand(3,2000);
% y = rand(3,5000);
% beta = rand(3,5000);
% eta = rand(3,2000);
% lambda = .25;
% res = F(x,y,beta,eta,lambda);

cur_dir = pwd;

Nargin = nargin;

% get verbosity level to force compilation
[~,~,~,verbosity,use_cuda_if_possible] = default_options();

% last arg is optional structure to override default tags
if isstruct(varargin{end})
    options = varargin{end};
    Nargin = Nargin-1;
    varargin = varargin(1:Nargin);
else
    options=struct;
end
% tagCpuGpu=0 means convolution on Cpu, tagCpuGpu=1 means convolution on Gpu, tagCpuGpu=2 means convolution on Gpu from device data
options = setoptions(options,'tagCpuGpu',1);
% tag1D2D=0 means 1D Gpu scheme, tag1D2D=1 means 2D Gpu scheme
options = setoptions(options,'tag1D2D',0);

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

% from the string inputs we form the code which will be added to the source cpp/cu file, and the string used to encode the file name
[CodeVars,indxy ] = format_var_aliase(aliases);

% we use a hash to shorten string and avoid special characters in the filename
Fname = string2hash(lower([CodeVars,formula]));
mex_name = [Fname,'.',mexext];

if ~(exist(mex_name,'file')==3) || (verbosity == 1)
     compile_formula(CodeVars,formula,Fname);
end

% return function handler
F = @Eval;

% the evaluation function
function out = Eval(varargin)
    nx = size(varargin{indxy(1)},2);
    ny = size(varargin{indxy(2)},2);
    options.tagCpuGpu = 0
    out = feval(Fname,nx,ny,options.tagCpuGpu,...
        options.tag1D2D,varargin{:});
end

end
