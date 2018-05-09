%% this example shows how to use the autodiff with keops. 

path_to_lib = '..';
addpath(genpath(path_to_lib))

%% gradient of a gaussian kernel:
%--------------------------------------%
%         create dataset               %
%--------------------------------------%

Nx = 5000;
Ny = 2000;
x = randn(3,Nx);
y = randn(3,Ny);
b = randn(3,Ny);
a = randn(3,Nx);
p = .25;

%-----------------------------------------%
%           Kernel with KeOps             %
%-----------------------------------------%

% compile the cpp/cuda routine
F = Kernel('x=Vx(0,3)','y=Vy(1,3)','b=Vy(2,3)','a=Vx(3,3)', 'p=Pm(4,1)',...
           'Grad(Exp(-p*SqNorm2(x-y))*b,x,a)');

% compute the operation
tic
g = F(x,y,b,a,p);
toc

% display part of the result
g(:,1:10)


%% Another example
%--------------------------------------%
%         create dataset               %
%--------------------------------------%

Nx = 5000;
Ny = 2000;
x0 = ones(3,Nx); x1 = ones(3,Nx); x2 = ones(3,Nx); x3 = ones(3,Nx);
y4 = ones(3,Ny); y5 = ones(3,Ny); y6 = ones(3,Ny); y7 = ones(3,Ny);
x8 = ones(3,Nx); x9 = ones(3,Nx);


%-----------------------------------------%
%           Autodiff with KeOps           %
%-----------------------------------------%

F = Kernel('x0=Vx(0,3)','x1=Vx(1,3)','x2=Vx(2,3)','x3=Vx(3,3)',...
    'y4=Vy(4,3)','y5=Vy(5,3)','y6=Vy(6,3)','y7=Vy(7,3)','x8=Vx(8,3)','x9=Vx(9,3)',...
    'Grad(x0+x1+x2+x3+y4+y5+y6+y7,x0,x8)');
tic
g = F(x0,x1,x2,x3,y4,y5,y6,y7,x8,x9);
toc

% display parts of the results
g(:,1:10)
