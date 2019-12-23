%% this example shows how to use the autodiff with keops. 

path_to_lib = [fileparts(mfilename('fullpath')), filesep, '..'];
addpath(genpath(path_to_lib))

disp('Testing automatic differentiation of sum reductions with Keops')

%% Test 1 gradient of a gaussian kernel:
disp('test 1')

% defining input arrays
Nx = 5000;
Ny = 2000;
x = randn(3,Nx);
y = randn(3,Ny);
b = randn(3,Ny);
a = randn(3,Nx);
p = .25;

% defining reduction operation F and its gradient G :
F = keops_kernel('Sum_Reduction(Exp(-p*SqNorm2(x-y))*b,0)',...
    'x=Vi(3)','y=Vj(3)','b=Vj(3)','p=Pm(1)');
% Using AutoDiff engine
G = keops_grad(F,'x');
% Using direct computation
H = keops_kernel('Sum_Reduction(IntCst(2)*p*(y-x)*Exp(-p*SqNorm2(x-y))*(b|a),0)',...
'x=Vi(3)','y=Vj(3)','b=Vj(3)','p=Pm(1)', 'a=Vi(3)');
h = H(x,y,b,p,a);

tic
g = G(x,y,b,p,a); % actual computation
toc

disp('Compare the two implementations. Absolute error: ')
sum(sum(abs(g - h)))


%% Test 2 Another example, gradient with respect to a 'j indexed' variable
disp('test 2')

Nx = 5000;
Ny = 2000;
x0 = ones(3,Nx); x1 = ones(3,Nx); x2 = ones(3,Nx); x3 = ones(3,Nx);
y4 = ones(3,Ny); y5 = ones(3,Ny); y6 = ones(3,Ny); y7 = ones(3,Ny);
x8 = ones(3,Nx);

F = keops_kernel('x0=Vi(3)','x1=Vi(3)','x2=Vi(3)','x3=Vi(3)',...
    'y4=Vj(3)','y5=Vj(3)','y6=Vj(3)','y7=Vj(3)',...
    'Sum_Reduction(x0+x1+x2+x3+y4+y5+y6+y7,0)');
G = keops_grad(F,'y4');

tic
g = G(x0,x1,x2,x3,y4,y5,y6,y7,x8);
toc

disp('first output values :')
g(:,1:5)

%% Test 3 gradient with respect to a parameter variable
disp('test 3')

% defining input arrays
Nx = 5000;
Ny = 2000;
x = randn(3,Nx);
y = randn(3,Ny);
b = randn(3,Ny);
a = randn(3,Nx);
p = .25;

% defining reduction operation F and its gradient G :
F = keops_kernel('Sum_Reduction(Exp(-p*SqNorm2(x-y))*b,0)',...
    'x=Vi(3)','y=Vj(3)','b=Vj(3)','p=Pm(1)');
G = keops_grad(F,'p');

tic
g = G(x,y,b,p,a); % actual computation
toc

disp('output :')
g

