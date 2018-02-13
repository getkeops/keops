addpath('..')
addpath('../build')

Nx = 5000;
Ny = 2000;
x = randn(3,Nx);
y = randn(3,Ny);
b = randn(3,Ny);
p = .25;

F = Kernel('Vx(0,3)','Vy(1,3)','GaussKernel_(3,3)');
tic
g = F(x,y,b,p);
toc

tic
ox = ones(Nx,1);
oy = ones(Ny,1);
r2=0;
for k=1:3
    xmy = ox*y(k,:)-(oy*x(k,:))';
    r2 = r2 + xmy.^2;
end
g0 = (exp(-p*r2)*b')';
toc

norm(g-g0)
