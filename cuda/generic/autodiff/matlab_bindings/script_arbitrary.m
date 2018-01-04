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
g = EvalFormula('Grad(x0(3)+x1(3)+x2(3)+x3(3)+y4(3)+y5(3)+y6(3)+y7(3),x0(3),x8(3))',...
    x0,x1,x2,x3,y4,y5,y6,y7,x8);
g(:,1:10)

