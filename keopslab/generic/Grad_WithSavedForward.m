function G = Grad_WithSavedForward(F,var)

% defines the gradient of a kernel convolution
% F is the output of a call to Kernel function
% var is a string identifying the variable with respect
% to which the gradient must be taken
% Example : define a Gaussian kernel convolution, then take its gradient
% with respect to the first variable and test
% F = Kernel('GaussKernel(p,x,y,b)','p=Pm(0,1)','x=Vx(1,3)','y=Vy(2,3)','b=Vy(3,3)');
% G = Grad(F,'x');
% Nx = 5000;
% Ny = 2000;
% x = randn(3,Nx);
% y = randn(3,Ny);
% b = randn(3,Ny);
% c = randn(3,Nx); % we need to input a new array with correct size
% p = .25;
% res = G(p,x,y,b,c);

% we get the arguments (variables and formula) from the original call to 
% the Kernel function in the nested function F
s = functions(F);
vars = s.workspace{1}.aliases;
formula = s.workspace{1}.formula;
options = s.workspace{1}.options;

% get index position of new variable to feed in the gradient
% since indices in C++ code start at 0, the new index equals the number of
% variables
posnewvar = options.numvars;

% number of variables for gradient function
options.numvars = options.numvars+1;

% we analyse the "var" string argument. If var is entered as a name, 
% we must retrieve the corresponding type
if length(var)<3 || (~strcmp(var(1:3),'Vx(') && ~strcmp(var(1:3),'Vy(') && ~strcmp(var(1:3),'Pm('))
    for k=1:length(vars)
        [vkname,vkid] = sepeqstr(vars{k});
        if strcmp(vkname,var)
            vartype = vkid;
        end
    end
else
    vartype = var;
end

% if we take the gradient with respect to a parameter Pm(...),
% then we must further sum the output with respect to i index
if vartype(2)=='m' 
    options.sumoutput = 1;
end

% finally we call the Kernel function with the new formula
args = [vars,['Grad_WithSavedForward_FromPos(',formula,',',var,',',num2str(posnewvar),')']];
G = Kernel(args{:},options);

end
