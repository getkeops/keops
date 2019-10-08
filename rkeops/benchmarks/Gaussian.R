
# =======================================================
# Gaussian convolution algorithm using the generic syntax
# =======================================================
# 
# We define three datasets x (M samples) and y, b (N samples) of points in R^D,
# then apply a gaussian convolution operation :
# out_i = sum_j exp(-||x_i-y_j||^2/(2sigma^2)) b_j

library(class)

#setwd("~/Desktop/keops_github/keops")
#devtools::install("rkeops")

library(rkeops)

set_rkeops_option("tagCpuGpu", 1)
set_rkeops_option("precision", "double")

GaussConvExample = function(M,N,D)
{
    print(paste("Gaussian convolution with M=",N,", N=,",N,", D=,",D,sep=""))

    x = matrix(runif(M*D),D,M)
    y = matrix(runif(N*D),D,N)
    b = matrix(runif(N*D),D,N)
    
    formula = paste('Sum_Reduction(Exp(-lambda*SqDist(x,y))*b,1)',sep="")
    var1 = paste('x=Vi(',D,')',sep="")  # First arg   : i-variable, of size D
    var2 = paste('y=Vj(',D,')',sep="")  # Second arg  : j-variable, of size D
    var3 = paste('b=Vj(',D,')',sep="")  # Third arg   : j-variable, of size D
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
    
    # dummy first calls for accurate timing in case of GPU use
    dum = matrix(runif(D*10),nrow=D)
    my_routine(dum,dum)
    my_routine(dum,dum)
    
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
    
    if(all(out1==out2)&all(out2==out3))
        print("OK same results")
    else
        print(paste("NOT OK : different results !! (",100*mean(cl1==cl2)," % agree)",sep=""))
    
    res
}

Ns = c(100,200,500,1000,2000,5000,10000,20000,50000,100000,200000,500000,1000000)
nN = length(Ns)
res = matrix(0,nN,4)
colnames(res) = c("Npoints","GaussConv(KeOps)","GaussConv(matrices)","GaussConv(loops)")
res[,1] = Ns
for(l in 1:nN)
    res[,l] =GaussConvExample(M=Ns[l],N=Ns[l],D=3)
res = res[,c(1,3,4,2)]
print("")
print("Timings:")
print(res)
