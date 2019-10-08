
# =======================================================
# Gaussian convolution algorithm using the generic syntax
# =======================================================
# 
# We define three datasets x (M samples) and y, b (N samples) of points in R^D,
# then apply a gaussian convolution operation :
# out_i = sum_j exp(-||x_i-y_j||^2/(2sigma^2)) b_j

devtools::install("../../rkeops")

library(rkeops)

set_rkeops_option("tagCpuGpu", 1)

GaussConvExample = function(M,N,D)
{
    print(paste("Gaussian convolution with M=",M,", N=",N,", D=",D,sep=""))

    x = matrix(runif(M*D),D,M)
    y = matrix(runif(N*D),D,N)
    b = matrix(runif(N*D),D,N)
    lambda = matrix(1.0 / 0.25^2)
    
    formula = paste('Sum_Reduction(Exp(-lambda*SqDist(x,y))*b,0)',sep="")
    var1 = paste('x=Vi(',D,')',sep="")  # First arg   : i-variable, of size D
    var2 = paste('y=Vj(',D,')',sep="")  # Second arg  : j-variable, of size D
    var3 = paste('b=Vj(',D,')',sep="")  # Third arg   : j-variable, of size D
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
    
    # dummy first calls for accurate timing in case of GPU use
    dum = matrix(runif(D*10),nrow=D)
    my_routine(list(dum,dum,dum,lambda),10,10)
    my_routine(list(dum,dum,dum,lambda),10,10)
       
    start = Sys.time()
    out1 = my_routine(list(x,y,b,lambda),M,N)
    end = Sys.time()
    res = end-start

    my_routine = my_routine_keops_double
    start = Sys.time()
    out2 = my_routine(list(x,y,b,lambda),M,N)
    end = Sys.time()
    res = c(res,end-start)
    
    # compare with standard R implementation via matrices
    if(M<15000)
    {
        start = Sys.time()
        M = ncol(x)
	N = ncol(y)
        SqDist = matrix(0,M,N)
	onesM = matrix(1,M,1)
	onesN = matrix(1,N,1)
        for(k in 1:D)
            SqDist = SqDist + (onesN %*% x[k,] - t(onesM %*% y[k,]))^2
        K = exp(-lambda[1]*SqDist)
	out3 = t(t(K) %*% t(b))
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
colnames(res) = c("Npoints","R (K**** float)","R (K**** double)","R")
res[,1] = Ns
Ntry = 10
for(l in 1:nN)
{
    resl = 0
    for(i in 1:Ntry)
        resl = resl + GaussConvExample(M=Ns[l],N=Ns[l],D=3)
    res[l,2:4] = resl / Ntry
}
res = res[,c(1,4,2,3)]
print("")
print("Timings:")
print(res)
