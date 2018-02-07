function F = Kernel(varargin)
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

% the follwing adds path settings from the shell before calling external
% programs; this is used to get nvcc and mex commands in the path
if exist('~/.bash_profile')
    bashpathcommand = 'source ~/.bash_profile;';
elseif exist('~/.profile')
    bashpathcommand = 'source ~/.profile;';
end


options = struct;
% tagIJ=0 means sum over j, tagIj=1 means sum over j
options.tagIJ = 0;
% tagCpuGpu=0 means convolution on Cpu, tagCpuGpu=1 means convolution on Gpu,
% tagCpuGpu=2 means convolution on Gpu from device data
options.tagCpuGpu = 0;
% tagUseNvcc=1 means compilation is done using nvcc compiler if possible, 0
% means use c++ compiler
options.tagUseNvcc = 0;
% tag1D2D=0 means 1D Gpu scheme, tag1D2D=1 means 2D Gpu scheme
options.tag1D2D = 1;

Nargin = nargin;

if isstruct(varargin{end})
    % last arg is optional structure to override default tags
    s = varargin{end};
    option = fieldnames(s);
    for k=1:length(option)
        eval(['options.',option{k},'=s.',option{k},';'])
    end  
    Nargin = Nargin-1;
    varargin = varargin(1:Nargin);
end

[~,tmp] = mysystemcall([bashpathcommand,'which nvcc']);
if ~isempty(tmp) && options.tagUseNvcc==1
    compilescript = 'compile_mex';
    tagcompile = 'cuda';
else
    compilescript = 'compile_mex_cpu'
    tagcompile = 'cpp';
end

indxy = [-1,-1]; % indxy will be used to calculate nx and ny from input variables

% from the string inputs we form the code which will be added to the source
% cpp/cu file, and the string used to encode the file name
formula = varargin{Nargin};
fname = formula;
CodeFormula = ['#define FORMULA_OBJ ',formula];
CodeVars = '';
for k=1:Nargin-1
    str = varargin{k};
    fname = [str,';',fname];
    [varname,vartype] = sepeqstr(str);
    if ~isempty(varname)
        CodeVars = [CodeVars,'decltype(',vartype,') ',varname,';'];
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
% we use md5 hash to shorten string and avoid special characters in the
% filename
[~,sysid] = mysystemcall(['uname']);
if strcmp(sysid,'Darwin')
    md5cmd = 'md5';
else
    md5cmd = 'md5sum';
end
[~,fname] = mysystemcall(['echo "',fname,'"|',md5cmd]);
fname = cleanstring(fname);
Fname = ['F',fname,'_',tagcompile];
filename = [Fname,'.',mexext];
if ~(exist(filename,'file')==3)
    disp('Formula is not compiled yet ; compiling...')
    TestBuild = buildFormula(CodeVars,CodeFormula,filename)
    if TestBuild
        disp('Compilation succeeded')
    else
        error('Compilation failed')
    end
else
    TestBuild = 1;
end

% the evaluation function
    function out = Eval(varargin)
        nx = size(varargin{indxy(1)},2);
        ny = size(varargin{indxy(2)},2);
        out = feval(Fname,nx,ny,options.tagIJ,options.tagCpuGpu,...
            options.tag1D2D,varargin{:});
    end
F = @Eval;

    function testbuild = buildFormula(code1,code2,filename)
        cd ..
        ['./',compilescript,' "',code1,'" "',code2,'"']
        mysystemcall(['./',compilescript,' "',code1,'" "',code2,'"'])
        cd matlab_bindings
        testbuild = exist(['build/tmp.',mexext],'file');
        if testbuild
            eval(['!mv build/tmp.',mexext,' "build/',filename,'"'])
        end
    end

    function [status,cmdout] = mysystemcall(command)
        [status,cmdout] = system([command])
        if cmdout(end)==char(10) || cmdout(end)==char(13)
            cmdout = cmdout(1:end-1);
        end
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

function str = cleanstring(str)
str = str(((str>='0')&(str<='9'))|((str>='a')&(str<='z')));
end

