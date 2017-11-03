% 
addpath(genpath('../'))

n=50; m=70; d=3;

x = randn(n,d);
y = randn(m,d);

p = randn(m,d);
q = randn(n,d);

sig = 2.4;

% ----------- matds version -----------
Mp = cudagrad1conv(q',x',y',p',sig)';


% ----------- matlab version -----------
d2 = zeros(n,m);
for i=1:d
    d2 = d2 + ( repmat(x(:,i),1,m)  -  repmat(y(:,i)',n,1) ) .^2 ;
end
A = (-2  .* exp(-d2 / sig .^2) / sig .^2) ;

Mp2 = zeros(n,d);
for i=1:d
    % Computation of 2*|x_i -x_j|*exp(-|x_i -x_j|^2/(lam^2))/(lam^2)
    ximyj = ( repmat(x(:,i),1,m)  -  repmat(y(:,i)',n,1) );
    Mp2(:,i) = sum(q .* ( ( ximyj .* A) * p),2);
end


% ---------- Compare results -----------
fprintf('grad1conv absolute error: %g\n',sum(abs(Mp(:) - Mp2(:)) .^2))

