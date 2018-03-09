
addpath('..')
addpath('../../../build')
setenv('LD_LIBRARY_PATH','')

Nx = 5000;
Ny = 2000;
c = 1./[.25;.5;1;2].^2;
w = [-.5;2;-1;1.7];
x = randn(3,Nx);
y = randn(3,Ny);
b = randn(2,Ny);

%F = Kernel('c=Pm(0,4)','w=Pm(1,4)','x=Vx(2,3)','y=Vy(3,3)','b=Vy(4,3)','(w,Exp(-SqDist(x,y)*c))*b');
F = Kernel('c=Pm(0,4)','w=Pm(1,4)','x=Vx(2,3)','y=Vy(3,3)','b=Vy(4,2)','SumGaussKernel(c,w,x,y,b)');
f = F(c,w,x,y,b);
f(:,1:10)

%G = Kernel('c=Pm(0,1)','x=Vx(1,3)','y=Vy(2,3)','b=Vy(3,3)','Exp(-SqDist(x,y)*c)*b');
G = Kernel('c=Pm(0,1)','x=Vx(1,3)','y=Vy(2,3)','b=Vy(3,2)','GaussKernel(c,x,y,b)');
f1 = G(c(1),x,y,b);
f2 = G(c(2),x,y,b);
f3 = G(c(3),x,y,b);
f4 = G(c(4),x,y,b);
g = w(1)*f1+w(2)*f2+w(3)*f3+w(4)*f4;
g(:,1:10)

mean(abs(f(:)-g(:)))


