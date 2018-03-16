path_to_lib = '..';
addpath(path_to_lib)

Nx = 5000;
Ny = 2000;
x = randn(3,Nx);
y = randn(3,Ny);
b = randn(3,Ny);
a = randn(3,Nx);
p = .25;

options.tagIJ = 0;
F = Kernel('x=Vx(0,3)','y=Vy(1,3)','b=Vy(2,3)','a=Vx(3,3)', 'p=Pm(4,1)', 'Grad(Exp(-p*SqNorm2(x-y))*b,x,a)');

tic
g = F(x,y,b,a,p);
toc
g(:,1:10)

