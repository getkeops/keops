addpath build

Nx = 5000;
Ny = 2000;
x = randn(3,Nx);
y = randn(3,Ny);
b = randn(3,Ny);
c = randn(3,Nx);
p = .25;
F = Kernel('Grad(Grad(GaussKernel_(3,3),x0(3),x3(3)),x0(3),x4(3))');
g = F(x,y,b,c,c,p);

g(:,1:10)

% ox = ones(Nx,1);
% oy = ones(Ny,1);
% 
% r2=0;
% for k=1:3
%     xmy = ox*y(k,:)-(oy*x(k,:))';
%     r2 = r2 + xmy.^2;
% end
% g0 = (exp(-p*r2)*b')';
% 
% g(:,1:10)
% g0(:,1:10)
% norm(g-g0)
