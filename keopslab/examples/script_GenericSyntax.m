% Example script to understand the generic syntax of KeOps and its use
% with Matlab bindings. Please read file generic_syntax.md at the root of
% the library for explanations.

path_to_lib = [fileparts(mfilename('fullpath')), filesep, '..'];
addpath(genpath(path_to_lib))

% defining the kernel operation
f = keops_kernel('Square(p-a)*Exp(x+y)','p=Pm(1)','a=Vj(1)','x=Vi(3)','y=Vj(3)');

% defining input variables
n = 30;
m = 20;
p = .25;
a = randn(1,n);
x = randn(3,m);
y = randn(3,n);

% computing
c = f(p,a,x,y)

% defining the gradient of the kernel operation
Gfy = keops_grad(f,'y');

% defining new input variable
e = randn(3,m);

% computing gradient
Gfy(p,a,x,y,e)
