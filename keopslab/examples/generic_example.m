path_to_lib = '..';
addpath(genpath(path_to_lib))

%--------------------------------------%
%         create dataset               %
%--------------------------------------%

Nx = 50;
Ny = 20;
x = randn(3,Nx);
y = randn(3,Ny);
b = randn(3,Ny);
u = randn(4,Nx);
v = randn(4,Ny);
p = .25;


%-----------------------------------------%
%           Kernel with KeOps             %
%-----------------------------------------%

F = Kernel('x=Vx(3)','y=Vy(3)','u=Vx(4)','v=Vy(4)','b=Vy(3)', 'p=Pm(1)',...
           'Square((u,v))*Exp(-p*SqNorm2(x-y))*b');

tic
g = F(x,y,u,v,b,p);
fprintf('Time for libkp computation : %f s.\n', toc)

%-----------------------------------------%
%            Compare with matlab          %
%-----------------------------------------%

squmatrix_distance = @(x,y) sum( (repmat(reshape(x,size(x,1),1,size(x,2)),1,size(y,1),1)  - repmat(reshape(y,1,size(y,1),size(y,2)),size(x,1),1,1)) .^2,3);

tic
uv = u' * v;
g0 = ((uv .^2 .* exp(-p*squmatrix_distance(x',y'))) *b')';
fprintf('Time for pure matlab computation : %f s.\n', toc)

fprintf('\nAbsolute error: %g\n', norm(g-g0))
