% This script shows how to select the GPU device used for a KeOps
% computation

path_to_lib = '..';
addpath(genpath(path_to_lib))

%--------------------------------------%
%         create dataset               %
%--------------------------------------%

Nx = 500000;
Ny = 200000;
x = randn(3,Nx);
y = randn(3,Ny);
b = randn(3,Ny);
p = .25;

%-----------------------------------------%
%           Kernel with KeOps             %
%-----------------------------------------%

% we can specify the id of the GPU in the optional structure passed to
% "Kernel" function
options = struct('device_id',0);

F = keops_kernel('x=Vx(3)','y=Vy(3)','b=Vy(3)', 'p=Pm(1)',...
           'SumReduction(Exp(-p*SqNorm2(x-y))*b,0)',...
           options);
       
tic
g = F(x,y,b,p);
fprintf('Time for keops computation : %f s.\n', toc)
