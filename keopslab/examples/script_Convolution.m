% Example script to understand the generic syntax of KeOps and its use
% with Matlab bindings. Please read file generic_syntax.md at the root of
% the library for explanations.

path_to_lib = [fileparts(mfilename('fullpath')), filesep, '..'];
addpath(genpath(path_to_lib))

% defining the kernel operation
F = keops_kernel('Exp(-SqDist(x,y)*g)*b','x=Vi(3)','y=Vj(3)','b=Vj(3)','g=Pm(1)');

% defining input variables
n = 30;
m = 10;
x = randn(3,m);
y = randn(3,n);
b = randn(3,n);
s = .5;

% computing with generic
res_gen = F(x,y,b,1/(s*s))

% computing with specific
res_spe = radial_kernel_conv(x,y,b,s,'gaussian')

% start benchmark
fprintf('Start benchmarking specific vs generic ... \n')


tic
for i=1:100
n = 3000;
m = 2000;
x = randn(3,m);
y = randn(3,n);
b = randn(3,n);
res_spe = radial_kernel_conv(x,y,b,s,'gaussian');
end
fprintf('Elapsed time for specific codes : %g s\n',toc)


tic
for i=1:100
n = 3000;
m = 2000;
x = randn(3,m);
y = randn(3,n);
b = randn(3,n);
res_gen = F(x,y,b,1/(s*s));
end
fprintf('Elapsed time for generic codes : %g s\n',toc)
