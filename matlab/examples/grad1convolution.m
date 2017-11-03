% 
addpath(genpath('../'))

n=5000; m=12000; d=3;

x = randn(n,d);
y = randn(m,d);
p = randn(m,d);
q = randn(n,d);

sig = 2.4;

dgaussian = @(x,s) -exp(-x / s .^2) / s.^2 ;
dlaplacian = @(x,s) -exp(- sqrt(x + s .^2) ) ./ (2*sqrt(x + s .^2));
denergy = @(x,s) -.25 ./ (x + s .^2) .^ (1.25) ;

squmatrix_distance = @(x,y) sum( (repmat(reshape(x,size(x,1),1,size(x,2)),1,size(y,1),1)  - repmat(reshape(y,1,size(y,1),size(y,2)),size(x,1),1,1)) .^2,3);

% ----------- gaussian kernel -----------
fprintf('\n----- Gaussian kernel\n')

tic
Mp = cudagrad1conv(q',x',y',p',sig)';
fprintf('Time for cuda  : %g\n',toc)

tic
d2 = squmatrix_distance(x,y);
A = dgaussian(d2,sig) ;
Mp2 = zeros(n,d);
for i=1:d
    % Computation of 2*|x_i -x_j|*exp(-|x_i -x_j|^2/(lam^2))/(lam^2)
    ximyj = ( repmat(x(:,i),1,m)  -  repmat(y(:,i)',n,1) );
    Mp2(:,i) = sum(q .* ( (2 * ximyj .* A) * p),2);
end
fprintf('Time for python: %g\n',toc)

fprintf('grad1conv absolute error: %g\n',max(abs(Mp(:) - Mp2(:))))


% ----------- laplacian kernel -----------
fprintf('\n----- Laplacian kernel\n')

tic
Mp = cudagrad1conv(q',x',y',p',sig,'laplacian')';
fprintf('Time for cuda  : %g\n',toc)

tic
d2 = squmatrix_distance(x,y);
A = dlaplacian(d2,sig) ;
Mp2 = zeros(n,d);
for i=1:d
    % Computation of 2*|x_i -x_j|*exp(-|x_i -x_j|^2/(lam^2))/(lam^2)
    ximyj = ( repmat(x(:,i),1,m)  -  repmat(y(:,i)',n,1) );
    Mp2(:,i) = sum(q .* ( (2 * ximyj .* A) * p),2);
end
fprintf('Time for python: %g\n',toc)

fprintf('grad1conv absolute error: %g\n',max(abs(Mp(:) - Mp2(:))))


% ----------- energy kernel -----------
fprintf('\n----- Energy kernel\n')

tic
Mp = cudagrad1conv(q',x',y',p',sig,'energy')';
fprintf('Time for cuda  : %g\n',toc)

tic
d2 = squmatrix_distance(x,y);
A = denergy(d2,sig) ;
Mp2 = zeros(n,d);
for i=1:d
    % Computation of 2*|x_i -x_j|*exp(-|x_i -x_j|^2/(lam^2))/(lam^2)
    ximyj = ( repmat(x(:,i),1,m)  -  repmat(y(:,i)',n,1) );
    Mp2(:,i) = sum(q .* ( (2 * ximyj .* A) * p),2);
end
fprintf('Time for python: %g\n',toc)

fprintf('grad1conv absolute error: %g\n',max(abs(Mp(:) - Mp2(:))))
