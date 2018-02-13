addpath('..')
addpath('../../../build')

Nx = 50;
Ny = 20;
x = randn(3,Nx);
y = randn(3,Ny);
b = randn(3,Ny);
u = randn(4,Nx);
v = randn(4,Ny);
p = .25;

%F = Kernel('Vx(0,3)','Vy(1,3)','GaussKernel_(x,y)');
F = Kernel('x=Vx(0,3)','y=Vy(1,3)','u=Vx(2,4)','v=Vy(3,4)','b=Vy(4,3)', 'p=Pm(0)', 'Square((u,v))*Exp(-Cst(p)*SqNorm2(x-y))*b');

tic
g = F(x,y,u,v,b,p);
toc

tic
ox = ones(Nx,1);
oy = ones(Ny,1);
r2=0;
for k=1:3
    xmy = ox*y(k,:)-(oy*x(k,:))';
    r2 = r2 + xmy.^2;
end


uv = u' * v;
g0 = ((uv .^2 .* exp(-p*r2)) *b')';
toc

norm(g-g0)
