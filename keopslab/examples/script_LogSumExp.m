%% example of LogSumExp reduction, and its gradient:
% We compute out_i = log(sum_j exp(F_ij))
% where F is a formula, then take its gradient with 
% respect to a variable contained in formula F.
% N.B. We use the KeOps "Max_SumShiftExp" reduction
% which outputs (m_i,s_i) where m_i = max_j F_ij
% and s_i = sum_j exp(m_i-F_ij)), then finalize the 
% computation as out_i = m_i + log(s_i)

path_to_lib = [fileparts(mfilename('fullpath')), filesep, '..'];
addpath(genpath(path_to_lib))

%% LogSumExp reduction of a gaussian kernel:
disp('Testing LogSumExp reduction')

% defining input arrays
Nx = 5000;
Ny = 2000;
x = randn(3,Nx);
y = randn(3,Ny);
b = randn(1,Ny);
p = .25;

% defining reduction operation :
F = keops_kernel('Max_SumShiftExp_Reduction(Exp(-p*SqNorm2(x-y))*b,0)',...
    'x=Vi(3)','y=Vj(3)','b=Vj(1)','p=Pm(1)');

% performing computation and timing it
tic
ms = F(x,y,b,p);
f = ms(1,:)+log(ms(2,:));
toc

disp('first output values :')
f(:,1:5)

% comparing with Log of Sum of Exp
F2 = keops_kernel('Sum_Reduction(Exp(Exp(-p*SqNorm2(x-y))*b),0)',...
    'x=Vi(3)','y=Vj(3)','b=Vj(1)','p=Pm(1)');
disp('Testing Log of Sum reduction of Exp')
tic
f2 = log(F2(x,y,b,p));
toc

disp('first output values :')
f2(:,1:5)

disp('relative error :')
norm(f-f2)/norm(f)


%% gradient of a LogSumExp reduction of a gaussian kernel:
disp('Testing gradient of LogSumExp reduction')

% gradient input array
a = randn(1,Nx);

% defining gradient reduction operation :
G = keops_grad(F,'x');

tic
g = G(x,y,b,p,[rand(1,Nx);a],ms);
g = g./repmat(ms(2,:),3,1);
toc

disp('first output values :')
g(:,1:5)

% comparing with hand-made gradient of Log of Sum of Exp :
disp('Testing gradient of Log of Sum reduction of Exp')
G2 = keops_grad(F2,'x');
tic
g2 = G2(x,y,b,p,a)./repmat(exp(f),size(x,1),1);
toc

disp('first output values :')
g2(:,1:5)

disp('relative error :')
norm(g-g2)/norm(g)



%% gradient wrt y of a LogSumExp reduction of a gaussian kernel:
disp('Testing gradient wrt y of LogSumExp reduction')

% gradient input array
a = randn(1,Nx);

% defining gradient reduction operation :
G = keops_grad(F,'y');

tic
g = G(x,y,b,p,[rand(1,Nx);a./ms(2,:)],ms);
toc

disp('first output values :')
g(:,1:5)

% comparing with hand-made gradient of Log of Sum of Exp :
disp('Testing gradient of Log of Sum reduction of Exp')
G2 = keops_grad(F2,'y');
tic
g2 = G2(x,y,b,p,a./exp(f));
toc

disp('first output values :')
g2(:,1:5)

disp('relative error :')
norm(g-g2)/norm(g)



