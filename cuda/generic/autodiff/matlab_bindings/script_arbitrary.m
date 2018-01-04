addpath build

Nx = 5000;
Ny = 2000;
x0 = randn(3,Nx);
x1 = randn(3,Nx);
x2 = randn(3,Nx);
x3 = randn(3,Nx);
y0 = randn(3,Ny);
y1 = randn(3,Ny);
y2 = randn(3,Ny);
y3 = randn(3,Ny);
g = EvalFormula(['Add<Add<Add<_X<0,3>,_X<1,3>>,Add<_X<2,3>,_X<3,3>>>,',...
    'Add<Add<_Y<0,3>,_Y<1,3>>,Add<_Y<2,3>,_Y<3,3>>>>'],...
    x0,x1,x2,x3,y0,y1,y2,y3);
