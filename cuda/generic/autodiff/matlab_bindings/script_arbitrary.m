addpath build

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
F = Kernel('x0=x0(3)','x1=x1(3)','x2=x2(3)','x3=x3(3)',...
    'y4=y4(3)','y5=y5(3)','y6=y6(3)','y7=y7(3)','x8=x8(3)','x9=x9(3)',...
    'Grad(x0+x1+x2+x3+y4+y5+y6+y7,x0,x8)');
%F = Kernel('x0(3)+x1(3)+x2(3)+x3(3)+y4(3)+y5(3)+y6(3)+y7(3)');
g = F(x0,x1,x2,x3,y4,y5,y6,y7,x8,x9);
g(:,1:10)


