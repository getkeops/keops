% Example script to test accuracy options

path_to_lib = '..';
addpath(genpath(path_to_lib))

% defining the kernel operation - default option (block reduction)
F = keops_kernel('Exp(-SqDist(x,y)*g)*b','x=Vi(3)','y=Vj(3)','b=Vj(3)','g=Pm(1)');

% defining the kernel operation - direct sum (no block reduction)
options.use_blockred = 0;
G = keops_kernel('Exp(-SqDist(x,y)*g)*b','x=Vi(3)','y=Vj(3)','b=Vj(3)','g=Pm(1)', options);

% defining input variables
n = 300000;
m = 10000;
x = randn(3,m);
y = randn(3,n);
b = randn(3,n);
s = .5;

% computing with default
res_blockred = F(x,y,b,1/(s*s));

% computing with direct scheme
res_direct = G(x,y,b,1/(s*s));
err = mean(abs(res_blockred(:)-res_direct(:))./abs(res_blockred(:)));
fprintf('mean relative error blockred vs direct: %f\n', err);

% start benchmark
fprintf('Start benchmarking blockred vs direct ... \n')

Ntry = 100;
tic
for i=1:Ntry
    res_blockred = F(x,y,b,1/(s*s));
end
fprintf('Average elapsed time for blockred code : %g s\n',toc/Ntry)

tic
for i=1:Ntry
    res_direct = G(x,y,b,1/(s*s));
end
fprintf('Average elapsed time for direct code : %g s\n',toc/Ntry)



