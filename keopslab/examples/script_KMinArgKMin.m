%% example of KMin, ArgKMin, and KMin-KArgMin reductions :

path_to_lib = [fileparts(mfilename('fullpath')), filesep, '..'];
addpath(genpath(path_to_lib))

%% KMin reduction of a gaussian kernel: K minimal values for each index
disp('Testing KMin reduction, with K=3')

% defining input arrays
Nx = 5000;
Ny = 2000;
x = randn(3,Nx);
y = randn(3,Ny);
b = randn(2,Ny);
p = .25;

% defining reduction operation :
F = keops_kernel('KMin_Reduction(Exp(-p*SqNorm2(x-y))*b,3,0)',...
    'x=Vi(3)','y=Vj(3)','b=Vj(2)','p=Pm(1)');

% performing computation and timing it
tic
f = F(x,y,b,p);
toc

disp('first output values :')
disp('here output is of dimension 6, the first 2 lines give minimal values :')
f(1:2,1:5)
disp('the next 2 lines give all the 2nd minimal values :')
f(3:4,1:5)
disp('the last 2 lines give all the 3rdd minimal values :')
f(5:6,1:5)

%% ArgKMin reduction of a gaussian kernel:
disp('Testing ArgMin reduction')

% defining reduction operation :
F = keops_kernel('ArgKMin_Reduction(Exp(-p*SqNorm2(x-y))*b,3,0)',...
    'x=Vi(3)','y=Vj(3)','b=Vj(2)','p=Pm(1)');

% performing computation and timing it
tic
f = F(x,y,b,p);
toc

disp('first output values : same as before, but gives indices instead of values')
f(:,1:5)

%% KMinArgKMin reduction of a gaussian kernel:
disp('Testing KMinArgKMin reduction')

% defining reduction operation :
F = keops_kernel('KMin_ArgKMin_Reduction(Exp(-p*SqNorm2(x-y))*b,3,0)',...
    'x=Vi(3)','y=Vj(3)','b=Vj(2)','p=Pm(1)');

% performing computation and timing it
tic
f = F(x,y,b,p);
toc

disp('first output values : dimension of output is 12 : ')
disp('min values :')
f(1:2,1:5)
disp('argmins :')
f(3:4,1:5)
disp('2nd min values :')
f(5:6,1:5)
disp('corresponding indices :')
f(7:8,1:5)
disp('3rd min values :')
f(9:10,1:5)
disp('corresponding indices :')
f(11:12,1:5)

