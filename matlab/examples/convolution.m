% 
addpath(genpath('../'))

n=50;
m=50;
d=3;

x = randn(n,d);
y = randn(m,d);

p = randn(m,d);

sig = 2.4;

% ----------- matds version -----------
Mp = cudaconv(x',y',p',sig)';


% ----------- matlab version -----------
d2 = zeros(n,m);
for i=1:d
    d2 = d2 + ( repmat(x(:,i),1,m)  -  repmat(y(:,i)',n,1) ) .^2 ;
end

Mp2 = exp(-d2 / sig .^2) * p;


% ---------- Compare results -----------
fprintf('conv absolute error: %g\n',sum(abs(Mp(:) - Mp2(:)) .^2))
