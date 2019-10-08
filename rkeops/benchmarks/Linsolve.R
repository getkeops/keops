
# =======================================================
# Kernel system solver algorithm using the generic syntax
# =======================================================
# 
# We define a dataset x (N samples) of points in R^D,
# and a dataset b of N scalar values attached to each point,
# then solve the gaussian kernel system :
# b_i = sum_j exp(-||x_i-y_j||^2/(2sigma^2)) out_j

library(class)

#setwd("~/Desktop/keops_github/keops")
#devtools::install("rkeops")

library(rkeops)

set_rkeops_option("tagCpuGpu", 1)
set_rkeops_option("precision", "double")

ConjugateGradientSolver = function(linop, b, eps=1e-6)
{
    # Conjugate gradient algorithm to solve linear system of the form
    # Ma=b where linop is a linear operation corresponding
    # to a symmetric and positive definite matrix
    delta = length(b) * eps ^ 2
    a = 0
    r = b
    nr2 = sum(r^2)
    if(nr2 < delta)
        return(0*r)
    p = r
    k = 0
    while(TRUE)
    {
        Mp = linop(p)
        alp = nr2 / sum(p*Mp)
        a = a + alp*p
        r = r - alp*Mp
        nr2new = sum(r^2)
        if(nr2new < delta)
            break
        p = r + (nr2new / nr2) * p
        nr2 = nr2new
        k = k + 1
    }
    return(a)
}


LinsolveExample = function(N,D,alpha=0)
{
    print(paste("Gaussian kernel system solver with N=",N,", D=",D,", alpha=,",alpha,sep=""))

    x = matrix(runif(N*D),D,N)
    b = matrix(runif(N),1,N)
    lambda = matrix(1 / 0.25^2)
    
    formula = paste('Sum_Reduction(Exp(-lambda*SqDist(x,y))*b,1)',sep="")
    var1 = paste('x=Vi(',D,')',sep="")  # First arg   : i-variable, of size D
    var2 = paste('y=Vj(',D,')',sep="")  # Second arg  : j-variable, of size D
    var3 = paste('b=Vj(1)',sep="")      # Third arg   : j-variable, of size 1
    var4 = paste('lambda=Pm(1)',sep="") # Fourth arg  : parameter, of size 1
    variables = c(var1,var2,var3,var4)
    
    my_routine_keops = keops_kernel(formula, variables)
    
    my_routine_nokeops = function(x,y,b,lambda)
    {
        M = ncol(x)
	N = ncol(y)
	out = matrix(0,D,N)
	for(i in 1:M)
		for(j in 1:N)
			out[,i] = out[,i] + exp(-lambda*sum((x[,i]-y[,j])^2))*b[,j]
	out
    }
    
    my_routine = my_routine_keops

    my_linop = function(b)
	    my_routine(x,x,b,lambda) + alpha*b
    
    # dummy first calls for accurate timing in case of GPU use
    dum = matrix(runif(D*10),nrow=D)
    dum2 = matrix(runif(10),nrow=1)
    my_routine(dum,dum,dum2,lambda)
    my_routine(dum,dum,dum2,lambda)
    
    start = Sys.time()
    inds = my_routine(x,y)
    cl1 = round(colMeans(matrix(cly[inds],K,Ntest)))
    end = Sys.time()
    res = end-start
    
    # compare with standard R implementation via matrices
    start = Sys.time()
        M = ncol(x)
	N = ncol(y)
        SqDist = matrix(0,M,N)
	onesM = matrix(1,M,1)
	onesN = matrix(1,N,1)
        for(k in 1:D)
            SqDist = SqDist + (onesN %*% x[k,] - t(onesM %*% y[k,]))**2
        K = exp(-lambda*SqDist)
	out2 = K %*% b   
    end = Sys.time()
    res = c(res,end-start)
    
    # compare with standard R implementation via loops
    start = Sys.time()
        M = ncol(x)
	N = ncol(y)
	out3 = matrix(0,D,N)
	for(i in 1:M)
		for(j in 1:N)
			out3[,i] = out3[,i] + exp(-lambda*sum((x[,i]-y[,j])^2))*b[,j]
    end = Sys.time()
    res = c(res,end-start)
    
    print(paste("mean errors : ",mean(abs(out1-out2)),", ",mean(abs(out2-out3)),")",sep=""))
    
    res
}

Ns = c(100,200,500,1000,2000,5000,10000,20000,50000,100000,200000,500000,1000000)
nN = length(Ns)
res = matrix(0,nN,4)
colnames(res) = c("Npoints","GaussConv(KeOps)","GaussConv(matrices)","GaussConv(loops)")
res[,1] = Ns
for(l in 1:nN)
    res[l,2:4] =GaussConvExample(M=Ns[l],N=Ns[l],D=3)
res = res[,c(1,3,4,2)]
print("")
print("Timings:")
print(res)
