path_to_lib = '..';
addpath(genpath(path_to_lib))

Nx = 5000;
Ny = 2000;
x0 = ones(3,Nx);
x1 = ones(3,Nx);
x2 = ones(3,Nx);
x3 = ones(3,Nx);
y4 = ones(3,Ny);
y5 = ones(3,Ny);
y6 = ones(3,Ny);
y7 = ones(3,Ny);
x8 = ones(3,Nx);
x9 = ones(3,Nx);
F = Kernel('x0=Vx(0,3)','x1=Vx(1,3)','x2=Vx(2,3)','x3=Vx(3,3)',...
    'y4=Vy(4,3)','y5=Vy(5,3)','y6=Vy(6,3)','y7=Vy(7,3)','x8=Vx(8,3)','x9=Vx(9,3)',...
    'Grad(x0+x1+x2+x3+y4+y5+y6+y7,x0,x8)');
tic
g = F(x0,x1,x2,x3,y4,y5,y6,y7,x8,x9);
toc

g(:,1:10)


