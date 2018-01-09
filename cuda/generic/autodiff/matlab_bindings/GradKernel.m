function G = GradKernel(F,var,newvar)

% defines the gradient of a kernel convolution
% F is the output of a call to Kernel function
% var is a string identifying the variable with respect
% to which the gradient must be taken
% newvar is a string identifying the new variable which will be the input
% to the graident in the convolution
% Example : define a Gaussian kernel convolution, then take its gradient
% with respect to the first variable and test
% F = Kernel('x=Vx(0,3)','y=Vy(1,3)','GaussKernel_(3,3)');
% G = GradKernel(F,'x','c=Vx(3,3)');
% Nx = 5000;
% Ny = 2000;
% x = randn(3,Nx);
% y = randn(3,Ny);
% b = randn(3,Ny);
% c = randn(3,Nx);
% p = .25;
% res = G(x,y,b,c,p);

% we get the arguments (variables and formula) from the original call to 
% the Kernel function in the nested function F
s = functions(F);
vars = s.workspace{1}.varargin(1:end-1);
formula = s.workspace{1}.varargin{end};
options = s.workspace{1}.options;

% we analyse the "newvar" string argument : the string can be either of the form
% 'name=type' or simply 'type'
[newvarname,newvarid] = sepeqstr(newvar);
if isempty(newvarname) % simple 'type' form
    newvarname = newvar; % because newvarname will be the string inserted in the gradient formula
    newvar = {};
end

% we analyse the "var" string argument : if variable is of type Vy, we must
% perform convolution with respect to j instead of i
% first if var is entered as a name, we must retrieve the corresponding
% type
if length(var)<2 || (~strcmp(var(1:2),'Vx') && ~strcmp(var(1:2),'Vx'))
    for k=1:length(vars)
        [vkname,vkid] = sepeqstr(vars{k});
        if strcmp(vkname,var)
            vartype = vkid;
        end
    end
else
    vartype = var;
end

% vartype is of the form 'Vx(...' or 'Vy(...' so we look at the 2nd character :
if vartype(2)=='y' 
    options.tagIJ = 1;
end

% finally we call the Kernel function with the new formula
args = [vars,newvar,['Grad(',formula,',',var,',',newvarname,')']];
G = Kernel(args{:},options);

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


