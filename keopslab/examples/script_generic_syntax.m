% Example script to understand the generic syntax of KeOps and its use
% with Matlab bindings. Please read file generic_syntax.md at the root of
% the library for explanations.

path_to_lib = '..';
addpath(genpath(path_to_lib))

% defining the kernel operation
f = Kernel('Square(p-a)*Exp(x+y)','p=Pm(1)','a=Vy(1)','x=Vx(3)','y=Vy(3)');

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
Gfy = GradKernel(f,'y','e=Vx(3)');

% defining new input variable
e = randn(3,m);

% computing gradient
Gfy(p,a,x,y,e)

