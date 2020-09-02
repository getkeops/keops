
# =======================================================
# Kernel system solver algorithm using the generic syntax
# =======================================================
# 
# We define a dataset x (N samples) of points in R^D,
# and a dataset b of N scalar values attached to each point,
# then solve the gaussian kernel system :
# b_i = sum_j exp(-||x_i-y_j||^2/(2sigma^2)) out_j

devtools::install("../../rkeops")

library(rkeops)

set_rkeops_option("tagCpuGpu", 1)

ConjugateGradientSolver = function(linop, b, tol)
{
    # Conjugate gradient algorithm to solve linear system of the form
    # Ma=b where linop is a linear operation corresponding
    # to a symmetric and positive definite matrix
    delta = length(b) * tol ^ 2
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


LinsolveExample = function(N,D,alpha,tol)
{
    print(paste("Gaussian kernel system solver with N=",N,", D=",D,", alpha=",alpha,", tol=",tol,sep=""))

    x = matrix(runif(N*D),D,N)
    b = matrix(rnorm(N),1,N)
    lambda = matrix(.5 / 0.01^2)
    
    formula = paste('Sum_Reduction(Exp(-lambda*SqDist(x,y))*b,0)',sep="")
    var1 = paste('x=Vi(',D,')',sep="")  # First arg   : i-variable, of size D
    var2 = paste('y=Vj(',D,')',sep="")  # Second arg  : j-variable, of size D
    var3 = paste('b=Vj(1)',sep="")      # Third arg   : j-variable, of size 1
    var4 = paste('lambda=Pm(1)',sep="") # Fourth arg  : parameter, of size 1
    variables = c(var1,var2,var3,var4)
    
    set_rkeops_option("precision", "float")
    my_routine_keops_float = keops_kernel(formula, variables)
    
    set_rkeops_option("precision", "double")
    my_routine_keops_double = keops_kernel(formula, variables)
    
    my_routine_nokeops = function(args,nx,ny)
    {
      x = args[[1]]
      y = args[[2]]
      b = args[[3]]
      lambda = args[[4]]
      M = ncol(x)
      N = ncol(y)
      SqDist = matrix(0,M,N)
      onesM = matrix(1,M,1)
      onesN = matrix(1,N,1)
      for(k in 1:D)
        SqDist = SqDist + (onesN %*% x[k,] - t(onesM %*% y[k,]))^2
      K = exp(-lambda[1]*SqDist)
      out = t(t(K) %*% t(b)) 
    }
    
    my_routine = my_routine_keops_float

    my_linop = function(b)
	    my_routine(list(x,x,b,lambda),N,N) + alpha*b
    
    # dummy first calls for accurate timing in case of GPU use
    dum = matrix(runif(D*10),nrow=D)
    dum2 = matrix(runif(10),nrow=1)
    my_routine(list(dum,dum,dum2,lambda),10,10)
    my_routine(list(dum,dum,dum2,lambda),10,10)
    
    start = Sys.time()
    out1 = ConjugateGradientSolver(my_linop,b,tol=tol)
    end = Sys.time()
    res = end-start

    my_routine = my_routine_keops_double

    my_linop = function(b)
	    my_routine(list(x,x,b,lambda),N,N) + alpha*b

    start = Sys.time()
    out2 = ConjugateGradientSolver(my_linop,b,tol=tol)
    end = Sys.time()
    res = c(res,end-start)
    
    # compare with standard R implementation via matrices
    if(N<7000)
    {
        start = Sys.time()
        N = ncol(x)
        SqDist = matrix(0,N,N)
	onesN = matrix(1,N,1)
        for(k in 1:D)
            SqDist = SqDist + (onesN %*% x[k,] - t(onesN %*% x[k,]))**2
        K = exp(-lambda[1]*SqDist) + alpha*diag(N)
	out3 = t(solve(t(K),t(b),tol=tol))   
        end = Sys.time()
        res = c(res,end-start)
    }
    else
    {
        res = c(res,NaN)
        out3 = out1
    }
    
    print(paste("mean errors : ",mean(abs(out1-out2)),", ",mean(abs(out2-out3)),")",sep=""))
    
    res
}

Ns = c(100,200,500,1000,2000,5000,10000,20000,50000,100000,200000,500000,1000000)
nN = length(Ns)
res = matrix(0,nN,4)
colnames(res) = c("Npoints","R (K**** float)","R (K**** double)","R (solve{base})")
res[,1] = Ns
Ntry = 10
for(l in 1:nN)
{
    resl = 0
    for(i in 1:Ntry)
        resl = resl + LinsolveExample(N=Ns[l],D=3,alpha=0.8,tol=1e-6)
    res[l,2:4] = resl / Ntry
}
res = res[,c(1,4,2,3)]
print("")
print("Timings:")
print(res)
