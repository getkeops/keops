addpath build

Nx = 5000;
Ny = 2000;
x = randn(3,Nx);
y = randn(3,Ny);
b = randn(3,Ny);
c = randn(3,Nx);
p = .25;
options.tagIJ = 1;
F = Kernel('x=Vx(0,3)','y=Vy(1,3)','c=Vx(3,3)','d=Vx(4,3)','Grad(Grad(GaussKernel_(3,3),x,c),y,d)',options);

%F0 = Kernel('x=Vx(0,3)','y=Vy(1,3)','GaussKernel_(3,3)');
%F1 = GradKernel(F0,'x','c=Vx(3,3)');
%F = GradKernel(F1,'y','d=Vx(4,3)');

tic
g = F(x,y,b,c,c,p);
toc
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
