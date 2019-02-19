%% example of Min, ArgMin, and Min-ArgMin reductions :

path_to_lib = '..';
addpath(genpath(path_to_lib))

%% Min reduction of a gaussian kernel:
disp('Testing Min reduction')

% defining input arrays
Nx = 5000;
Ny = 2000;
x = randn(3,Nx);
y = randn(3,Ny);
b = randn(2,Ny);
p = .25;

% defining reduction operation :
F = keops_kernel('MinReduction(Exp(-p*SqNorm2(x-y))*b,0)','x=Vx(3)','y=Vy(3)','b=Vy(2)','p=Pm(1)');

% performing computation and timing it
tic
f = F(x,y,b,p);
toc

disp('first output values :')
f(:,1:5)

%% ArgMin reduction of a gaussian kernel:
disp('Testing ArgMin reduction')

% defining reduction operation :
F = keops_kernel('ArgMinReduction(Exp(-p*SqNorm2(x-y))*b,0)',...
    'x=Vx(3)','y=Vy(3)','b=Vy(2)','p=Pm(1)');

% performing computation and timing it
tic
f = F(x,y,b,p);
toc

disp('first output values :')
f(:,1:5)

%% MinArgMin reduction of a gaussian kernel:
disp('Testing MinArgMin reduction')

% defining reduction operation :
F = keops_kernel('MinArgMinReduction(Exp(-p*SqNorm2(x-y))*b,0)',...
    'x=Vx(3)','y=Vy(3)','b=Vy(2)','p=Pm(1)');

% performing computation and timing it
tic
f = F(x,y,b,p);
toc

disp('first output values : the first 2 lines give the mins')
f(1:2,1:5)
disp('the next 2 lines give the argmins')
f(3:4,1:5)

