
# ==========================================
# K-means algorithm using the generic syntax
# ==========================================
# 
# We define a dataset of N points in R^D, then apply a simple k-means algorithm.


#setwd("~/Desktop/keops_github/keops")
#devtools::install("rkeops")

library(rkeops)

set_rkeops_option("tagCpuGpu", 1)
set_rkeops_option("precision", "double")

indexedSum = function(V,inds,K)
{
    D = nrow(V)
    res = matrix(0,D,K)
    for(i in 1:length(inds))
    {
        ind = inds[i]
        res[,ind] = res[,ind] + V[,i]
    }
    res
}

KMeansExample = function(N,D,K,Niter=10)
{
    print(paste("k-means with N=",N,", D=,",D,", K=",K,", Niter=",Niter,sep=""))
    x = matrix(runif(N*D),D,N)
    
    formula = 'ArgMin_Reduction(SqDist(x,y),1)'
    var1 = paste('x=Vi(',D,')',sep="") # First arg   : i-variable, of size D
    var2 = paste('y=Vj(',D,')',sep="") # First arg   : j-variable, of size D
    variables = c(var1,var2)
    
    my_routine_keops = keops_kernel(formula, variables)

    my_routine_nokeops = function(args,nx,ny)
    {
	x = args[1]
	y = args[2]
        N = ncol(x)
        cl = rep(0,N)
        for(i in 1:N)
            cl[i] = which.min(colSums((x[,i]-y)^2))
        cl
    }
    
    my_routine = my_routine_keops
    
    # dummy first calls for accurate timing in case of GPU use
    dum = matrix(runif(D*10),nrow=D)
    my_routine(list(dum,dum),10,10)
    my_routine(list(dum,dum),10,10)
    
    start = Sys.time()
    C = x[,1:K]
    cl_old = rep(0,N)
    for(i in 1:Niter)
    {
        cl = as.integer(as.vector(my_routine(list(x,C),N,K)))
        if(all(cl==cl_old)) break;
        x_ = rbind(x,rep(1,N))
        C = indexedSum(x_,cl,K)
        for(d in 1:D)
            C[d,] = C[d,] / C[D+1,]
        C = C[1:D,]
        # below using aggregate ; does the same as previous 5 lines, but apparently slower
        # C = t(aggregate(t(x),list(cl),mean))[2:3,]
        cl_old = cl
    }
    cl1 = cl
    end = Sys.time()
    res = end-start
        
    # compare with standard R kmeans (library stats)
    start = Sys.time()
    cl2 = kmeans(t(x),t(x[,1:K]),iter.max = Niter)$clusters
    end = Sys.time()
    res = c(res,end-start)
    
    # compare with kmeans from caret package ?
    start = Sys.time()
    cl3 = cl2 # kmeans(t(x),K,iter.max = Niter)
    end = Sys.time()
    res = c(res,end-start)
    
    if(all(cl1==cl2)&all(cl2==cl3))
        print("OK same results")
    else
        print(paste("NOT OK : different results !! (",100*mean(cl1==cl2)," % agree)",sep=""))
    
    res
}

Ns = c(100,200,500,1000,2000,5000,10000,20000,50000,100000,200000,500000,1000000)
nN = length(Ns)
res = matrix(0,nN,4)
colnames(res) = c("Npoints","kmeans(KeOps)","kmeans{stats}","kmeans(caret)")
res[,1] = Ns
for(l in 1:nN)
    res[l,2:4] = KMeansExample(N=Ns[l],D=100,K=floor(sqrt(Ns[l])))
res = res[,c(1,3,4,2)]
print("")
print("Timings:")
print(res)
