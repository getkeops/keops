% Example script to test accuracy options

path_to_lib = '..';
addpath(genpath(path_to_lib))

% defining the kernel operation - default option (block reduction)
F = keops_kernel('Exp(-SqDist(x,y)*g)*b','x=Vi(3)','y=Vj(3)','b=Vj(3)','g=Pm(1)');

% defining the kernel operation - direct sum (no block reduction)
options.sum_scheme = 0;
G = keops_kernel('Exp(-SqDist(x,y)*g)*b','x=Vi(3)','y=Vj(3)','b=Vj(3)','g=Pm(1)', options);

% defining input variables
m = 10000;
n = 1000000;
x = randn(3,m);
y = randn(3,n);
b = randn(3,n);
s = .5;

% computing with default
res_blocksum = F(x,y,b,1/(s*s));

% computing with direct scheme
res_direct = G(x,y,b,1/(s*s));
err = mean(abs(res_blocksum(:)-res_direct(:))./abs(res_blocksum(:)));
fprintf('mean relative error blocksum vs direct: %f\n', err);

% computing with direct scheme, shuffled
ind = randperm(n);
res_direct = G(x,y(:,ind),b(:,ind),1/(s*s));
err = mean(abs(res_blocksum(:)-res_direct(:))./abs(res_blocksum(:)));
fprintf('mean relative error blocksum vs direct: %f\n', err);

% start benchmark
fprintf('Start benchmarking blocksum vs direct ... \n')

Ntry = 100;
tic
for i=1:Ntry
    res_blocksum = F(x,y,b,1/(s*s));
end
fprintf('Average elapsed time for blockred code : %g s\n',toc/Ntry)

tic
for i=1:Ntry
    res_direct = G(x,y,b,1/(s*s));
end
fprintf('Average elapsed time for direct code : %g s\n',toc/Ntry)



