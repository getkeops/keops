path_to_lib = '..';
addpath(genpath(path_to_lib))

clear

%--------------------------------------%
%         Create dataset               %
%--------------------------------------%

Nx = 5000;
Ny = 2000;
x = randn(3,Nx);
y = randn(3,1,Ny);
b = randn(3,1,Ny);
p = .25;

%-----------------------------------------%
%       Kernel with KeOps LazyTensor      %
%-----------------------------------------%

tic
X = LazyTensor(x);
Y = LazyTensor(y);
B = LazyTensor(b);
G = exp(-p*sum((X-Y).^2,1));
K = G.*b;
g = sum_reduction(K,3);
fprintf('Time for keops computation : %f s.\n', toc)

%-----------------------------------------%
%            Compare with matlab          %
%-----------------------------------------%

squmatrix_distance = @(x,y) sum( (repmat(reshape(x,size(x,1),1,size(x,2)),1,size(y,1),1)  - repmat(reshape(y,1,size(y,1),size(y,2)),size(x,1),1,1)) .^2,3);

y = reshape(y,3,Ny);
b = reshape(b,3,Ny);

tic
g0 = ((exp(-p*squmatrix_distance(x',y'))) *b')';
fprintf('Time for pure matlab computation : %f s.\n', toc)

fprintf('\nAbsolute error: %g\n', norm(g-g0))
fprintf('\nRelative error: %g\n', norm(g-g0)/norm(g0))


