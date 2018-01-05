function F = Kernel(varargin)
% Defines a Cuda kernel convolution function based on a formula
% arguments are strings defining variables and formula
%
% Examples:
%
% - Define and test a function that computes or each i the sum over j 
% of the square of the scalar products of xi and yj (both 3d vectors)
% F = Kernel('x=x0(3)','y=y1(3)','Pow2((x,y))');
% x = rand(3,2000);
% y = rand(3,5000);
% res = F(x,y);
%
% - Define and test the convolution with a Gauss kernel i.e. the sum 
% over j of e^(lambda*||xi-yj||^2)beta_j (xi,yj, beta_j 3d vectors): 
% F = Kernel('x=x0(3)','y=y1(3)','beta=y2(3)','Exp(lambda*SqNorm2(x-y))*beta');
% x = rand(3,2000);
% y = rand(3,5000);
% beta = rand(3,5000);
% res = F(x,y,beta);
%
% - Define and test the gradient of the previous function with respect 
% to the xi :
% F = Kernel('x=x0(3)','y=y1(3)','beta=y2(3)','eta=x3(3)',...
%           'Grad(Exp(lambda*SqNorm2(x-y))*beta,x,eta)');
% x = rand(3,2000);
% y = rand(3,5000);
% beta = rand(3,5000);
% eta = rand(3,2000);
% res = F(x,y,beta,eta);

obj = KernelClass(varargin{:});
    F = @obj.Eval;
end