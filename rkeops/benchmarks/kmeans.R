
# ==========================================
# K-means algorithm using the generic syntax
# ==========================================
# 
# We define a dataset of N points in R^D, then apply a simple k-means algorithm.


#devtools::install("../../rkeops")

library(rkeops)

set_rkeops_option("tagCpuGpu",1)

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
    print(paste("k-means with N=",N,", D=",D,", K=",K,", Niter=",Niter,sep=""))
    x = matrix(runif(N*D),D,N)
    
    formula = 'ArgMin_Reduction(SqDist(x,y),0)'
    var1 = paste('x=Vi(',D,')',sep="") # First arg   : i-variable, of size D
    var2 = paste('y=Vj(',D,')',sep="") # First arg   : j-variable, of size D
    variables = c(var1,var2)
    
    set_rkeops_option("precision", "float")
    my_routine_keops_float = keops_kernel(formula, variables)

    set_rkeops_option("precision", "double")
    my_routine_keops_double = keops_kernel(formula, variables)

    my_routine_nokeops = function(args,nx,ny)
    {
	x = args[[1]]
	y = args[[2]]
        N = ncol(x)
        cl = rep(0,N)
        for(i in 1:N)
            cl[i] = which.min(colSums((x[,i]-y)^2))
        cl
    }
    
    # dummy first calls for accurate timing in case of GPU use
    dum = matrix(runif(D*10),nrow=D)
    my_routine_keops_float(list(dum,dum),10,10)
    my_routine_keops_float(list(dum,dum),10,10)
    
    my_kmeans = function(my_routine)
    {
    C = x[,1:K]
    cl_old = rep(0,N)
    for(i in 1:Niter)
    {
        # 1) assignment step, done with KeOps
        cl = 1 + as.integer(as.vector(my_routine(list(x,C),N,K)))
        if(all(cl==cl_old)) break;
        # 2) recomputing centers : this step is the bottleneck for us, because here we do it with R for loops
        # over N, which is very slow; we should find someting faster
        x_ = rbind(x,rep(1,N))
        C = indexedSum(x_,cl,K)
        # other method we tried, with for loops on K, still unefficient
        #C = matrix(0,D+1,K)
        #for(k in 1:K)
        #    C[,k] = rowMeans(as.matrix(x_[,cl==k]))
        for(d in 1:D)
            C[d,] = C[d,] / C[D+1,]
        C = C[1:D,]
        # below using aggregate ; does the same, but apparently slower
        # C = t(aggregate(t(x),list(cl),mean))[2:3,]
        cl_old = cl
    }
    return(cl)
    }

    start = Sys.time()
    cl1 = my_kmeans(my_routine_keops_float)
    end = Sys.time()
    res = end-start

    start = Sys.time()
    cl2 = my_kmeans(my_routine_keops_double)
    end = Sys.time()
    res = c(res,end-start)
        
    # compare with standard R kmeans (library stats)
    if(N<300000)
    {
        start = Sys.time()
        cl3 = kmeans(t(x),t(x[,1:K]),iter.max = Niter, algorithm = "Lloyd")$cluster
        end = Sys.time()
        res = c(res,end-start)
    }
    else
    {
        res = c(res,NaN)
        cl3 = cl1
    }
    if(all(cl1==cl2)&all(cl2==cl3))
        print("OK same results")
    else
        print(paste("NOT OK : different results !! (",100*mean(cl1==cl2)," % agree)",sep=""))
    
    res
}

Ns = c(100,200,500,1000,2000,5000,10000,20000,50000,100000,200000,500000,1000000)
nN = length(Ns)
res = matrix(0,nN,4)
colnames(res) = c("Npoints","R (K**** float)","R (K**** double)","R (kmeans{stats})")
res[,1] = Ns
Ntry = 10
for(l in 1:nN)
{
    resl = 0
    for(i in 1:Ntry)
        resl = resl + KMeansExample(N=Ns[l],D=10,K=floor(sqrt(Ns[l])))
    res[l,2:4] = resl / Ntry
}
res = res[,c(1,4,2,3)]
print("")
print("Timings:")
print(res)
