
# ==========================================
# K-NN algorithm using the generic syntax
# ==========================================
# 
# We define a dataset of N points in R^D, separated in two classes,
# and a dataset of query points, then apply a simple K-NN algorithm.

library(class)

#setwd("~/Desktop/keops_github/keops")
#devtools::install("rkeops")

library(rkeops)

set_rkeops_option("tagCpuGpu", 1)
set_rkeops_option("precision", "double")

KNNExample = function(N,Ntest,D,K)
{
    print(paste("k-NN with N=",N,", Ntest=,",Ntest,", D=,",D,", K=",K,sep=""))

    x = matrix(runif(Ntest*D),D,Ntest)
    y = matrix(runif(N*D),D,N)
    cly = matrix(sample(0:1,N,rep=TRUE),1,N)
    
    formula = paste('ArgKMin_Reduction(SqDist(x,y),',K,',1)',sep="")
    var1 = paste('x = Vi(',D,')',sep="") # First arg   : i-variable, of size D
    var2 = paste('y = Vj(',D,')',sep="") # First arg   : j-variable, of size D
    variables = c(var1,var2)
    
    my_routine_keops = keops_kernel(formula, variables)
    
    my_routine_nokeops = function(x,y)
    {
        N = ncol(x)
        inds = matrix(0,K,N)
        for(i in 1:N)
            inds[,i] = order(colSums((x[,i]-y)^2))[1:K]
        inds
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
    
    # compare with standard R knn
    start = Sys.time()
    cl2 = knn(t(y),t(x),cly,k=K)
    end = Sys.time()
    res = c(res,end-start)
    
    # compare with knn from caret package ?
    start = Sys.time()
    cl3 = cl2 #knn(t(y),t(x),cly,k=K)
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
colnames(res) = c("Npoints","kNN(KeOps)","kNN{class}","kNN(caret)")
res[,1] = Ns
for(l in 1:nN)
    res[,l] = KNNExample(N=Ns[l],Ntest=10000,D=100,K=10)
res = res[,c(1,3,4,2)]
print("")
print("Timings:")
print(res)
