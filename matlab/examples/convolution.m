% 
addpath(genpath('../'))

n=2000; m=1200; d=3;

x = randn(n,d);
y = randn(m,d);
p = randn(m,d);

sig = 2.4;

gaussian = @(x,s) exp(-x / s .^2);

squmatrix_distance = @(x,y) sum( (repmat(reshape(x,size(x,1),1,size(x,2)),1,size(y,1),1)  - repmat(reshape(y,1,size(y,1),size(y,2)),size(x,1),1,1)) .^2,3);


%dry run
Mp = cudaconv(x',y',p',sig)';

% ----------- gaussian kernel -----------
tic
Mp = cudaconv(x',y',p',sig)';
fprintf('Time for cuda  : %g\n',toc)

tic
d2 = squmatrix_distance(x,y);
Mp2 = gaussian(d2,sig) * p;
fprintf('Time for matlab: %g\n',toc)

fprintf('conv absolute error: %g\n',max(abs(Mp(:) - Mp2(:))))




exit
