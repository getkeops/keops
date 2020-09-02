path_to_lib = [fileparts(mfilename('fullpath')), filesep, '..'];
addpath(genpath(path_to_lib))

%--------------------------------------%
%         create dataset               %
%--------------------------------------%

Nx = 5000;
Ny = 2000;
c = 1./[.25;.5;1;2].^2;
w = [-.5;2;-1;1.7];
x = randn(3,Nx);
y = randn(3,Ny);
b = randn(2,Ny);


%--------------------------------------%
%         automatic Kernel sums        %
%--------------------------------------%

% note here that the parameter w is a vector

% the two following formulas are equivalent
%F = keops_kernel('(w,Exp(-SqDist(x,y)*c))*b','c=Pm(0,4)','w=Pm(1,4)','x=Vi(2,3)','y=Vj(3,3)','b=Vj(4,2)');
F = keops_kernel('SumGaussKernel(c,w,x,y,b)','c=Pm(4)','w=Pm(4)','x=Vi(3)','y=Vj(3)','b=Vj(2)');
f = F(c,w,x,y,b);

% display part of the result
f(:,1:10)


%--------------------------------------%
%          manual Kernel sums          %
%--------------------------------------%

% here, the parameter is a real number ans the cpp/cuda routine is called
% several times

% the two following formulas are equivalent
%G = keops_kernel('c=Pm(0,1)','x=Vi(1,3)','y=Vj(2,3)','b=Vj(3,2)','Exp(-SqDist(x,y)*c)*b');
G = keops_kernel('c=Pm(1)','x=Vi(3)','y=Vj(3)','b=Vj(2)','GaussKernel(c,x,y,b)');
f1 = G(c(1),x,y,b);
f2 = G(c(2),x,y,b);
f3 = G(c(3),x,y,b);
f4 = G(c(4),x,y,b);
g = w(1)*f1+w(2)*f2+w(3)*f3+w(4)*f4;

% display part of the result
g(:,1:10)

% compare results
mean(abs(f(:)-g(:)))


