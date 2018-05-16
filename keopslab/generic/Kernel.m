function [F,Fname] = Kernel(varargin)
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
% F = Kernel('x=Vx(0,3)','y=Vy(1,3)','beta=Vy(2,3)','lambda=Pm(3,1)','Exp(lambda*SqNorm2(x-y))*beta');
% x = rand(3,2000);
% y = rand(3,5000);
% beta = rand(3,5000);
% lambda = .25;
% res = F(x,y,beta,lambda);
%
% - Define and test the gradient of the previous function with respect
% to the xi :
% F = Kernel('x=Vx(0,3)','y=Vy(1,3)','beta=Vy(2,3)','eta=Vx(3,3)','lambda=Pm(4,1)',...
%           'Grad(Exp(lambda*SqNorm2(x-y))*beta,x,eta)');
% x = rand(3,2000);
% y = rand(3,5000);
% beta = rand(3,5000);
% eta = rand(3,2000);
% lambda = .25;
% res = F(x,y,beta,eta,lambda);

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
    buildFormula(CodeVars,formula,Fname);
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




function [var_aliases,indxyp] =format_var_aliase(var_options)
% format var_aliases to pass option to cmake
var_aliases = '';
indxyp = [-1,-1,-1]; % indxyp will be used to calculate nx, ny and np from input variables

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
    if type=='x' && indxyp(1)==-1
        indxyp(1) = str2num(pos)+1;
    elseif type=='y' && indxyp(2)==-1
        indxyp(2) = str2num(pos)+1;
    elseif type=='m' && indxyp(3)==-1
        indxyp(3) = str2num(pos)+1;
    end
end
end


function testbuild = buildFormula(code1, code2, filename)

    disp(['Compiling formula ',code2,' with ',code1,' ...'])
    
    [src_dir,build_dir,precision] = default_options();
    
    % it seems to be a workaround to flush Matlab's default LD_LIBRARY_PATH
    setenv('LD_LIBRARY_PATH','') 
    % I do not have a better option to set working dir...
    cur_dir= pwd; cd(build_dir) ;
    % find cmake :
    cmake = getcmake();
    % cmake command:
    cmdline = [cmake,' ', src_dir , ' -DVAR_ALIASES="',code1,'" -DFORMULA_OBJ="',code2,'" -D__TYPE__=',precision,' -Dmex_name="',filename,'" -Dshared_obj_name="',filename,'" -DMatlab_ROOT_DIR="',matlabroot,'"' ];
    %fprintf([cmdline,'\n'])
    try
        [~,prebuild_output] = system(cmdline);
        [~,build_output]  = system(['make mex_cpp']);
    catch
        cd(cur_dir)
        error('Compilation  Failed')
    end
    % ...comming back to curent directory
    cd(cur_dir)

    testbuild = (exist([filename,'.',mexext],'file')==3);
    if  testbuild
        disp('Compilation succeeded')
    else
        error(['File "',filename,'.',mexext, '" not found!'])
    end
end

function cmake = getcmake()
    % check wether cmake is available from Matlab. Since matlab overrides
    % default search path, we may need to ask the user to enter its 
    % location manually the first time and then save the path in a local file
    S = dbstack('-completenames');
    pathtocmakefile = [fileparts(S(1).file),'/pathtocmake'];
    if exist(pathtocmakefile,'file')
        fid = fopen(pathtocmakefile);
        pathtocmake = fgetl(fid);
        fclose(fid);
    else
        pathtocmake = '';
    end
    [testcmake,~]=system([pathtocmake,'cmake']);
    if testcmake~=0
        system(['rm -f ',pathtocmakefile]);
        [testcmake,~]=system('cmake');
        if testcmake==0
            pathtocmake = '';
        else
            pathtocmake = input('cmake command is required but was not found. Enter path to cmake command here : ','s');
            if length(pathtocmake)>4 && strcmp(pathtocmake(end-4:end),'cmake')
                pathtocmake = pathtocmake(1:end-5);
            end
            if ~isempty(pathtocmake) && pathtocmake(end)~='/'
                pathtocmake = [pathtocmake,'/'];
            end
            [testcmake,~]=system([pathtocmake,'cmake']);
            if testcmake~=0
                error('cmake command not found.')
            end
            if ~isempty(pathtocmake)
                fid = fopen(pathtocmakefile,'w');
                fprintf(fid,pathtocmake);
                fclose(fid);
            end
        end
    end
    cmake = [pathtocmake,'cmake'];
end
