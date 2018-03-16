path_to_lib = '..';
addpath(genpath(path_to_lib))

n=5000; m=12000; d=3;

x = randn(n,d);
y = randn(m,d);
p = randn(m,d);

sig = 2.4;

gaussian = @(x,s) exp(-x / s .^2);
laplacian = @(x,s) exp(- sqrt(x + s .^2) );
energy = @(x,s)  (x + s .^2) .^ (-.25) ;

squmatrix_distance = @(x,y) sum( (repmat(reshape(x,size(x,1),1,size(x,2)),1,size(y,1),1)  - repmat(reshape(y,1,size(y,1),size(y,2)),size(x,1),1,1)) .^2,3);


%dry run to avoid overhead using GPU...
Mp = cudaconv(x',y',p',sig)';
% end of dry run

% ----------- gaussian kernel -----------
fprintf('\n----- Gaussian kernel\n')

tic
Mp = cudaconv(x',y',p',sig,'gaussian')';
fprintf('Time for cuda  : %g\n',toc)

tic
d2 = squmatrix_distance(x,y);
Mp2 = gaussian(d2,sig) * p;
fprintf('Time for matlab: %g\n',toc)

fprintf('conv absolute error: %g\n',max(abs(Mp(:) - Mp2(:))))

% ----------- laplacian kernel -----------
fprintf('\n----- Laplacian kernel\n')

tic
Mp = cudaconv(x',y',p',sig,'laplacian')';
fprintf('Time for cuda  : %g\n',toc)

tic
d2 = squmatrix_distance(x,y);
Mp2 = laplacian(d2,sig) * p;
fprintf('Time for matlab: %g\n',toc)

fprintf('conv absolute error: %g\n',max(abs(Mp(:) - Mp2(:))))

% ----------- Energy kernel -----------
fprintf('\n----- Energy kernel\n')

tic
Mp = cudaconv(x',y',p',sig,'energy')';
fprintf('Time for cuda  : %g\n',toc)

tic
d2 = squmatrix_distance(x,y);
Mp2 = energy(d2,sig) * p;
fprintf('Time for matlab: %g\n',toc)

fprintf('conv absolute error: %g\n',max(abs(Mp(:) - Mp2(:))))



exit
