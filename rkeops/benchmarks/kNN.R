
# ==========================================
# K-NN algorithm using the generic syntax
# ==========================================
# 
# We define a dataset of N points in R^D, separated in two classes,
# and a dataset of query points, then apply a simple K-NN algorithm.

library(class)

devtools::install("../../rkeops")

library(rkeops)

set_rkeops_option("tagCpuGpu", 1)

KNNExample = function(N,Ntest,D,K)
{
    print(paste("k-NN with N=",N,", Ntest=",Ntest,", D=",D,", K=",K,sep=""))

    x = matrix(runif(Ntest*D),D,Ntest)
    y = matrix(runif(N*D),D,N)
    cly = matrix(sample(0:1,N,rep=TRUE),1,N)
    
    formula = paste('ArgKMin_Reduction(SqDist(x,y),',K,',0)',sep="")
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
        inds = matrix(0,K,N)
        for(i in 1:N)
            inds[,i] = order(colSums((x[,i]-y)^2))[1:K]
        inds
    }
    
    my_routine = my_routine_keops_float
    
    # dummy first calls for accurate timing in case of GPU use
    dum = matrix(runif(D*10),nrow=D)
    my_routine(list(dum,dum),10,10)
    my_routine(list(dum,dum),10,10)
    
    start = Sys.time()
    inds = 1 + as.integer(as.vector(my_routine(list(x,y),Ntest,N)))
    cl1 = round(colMeans(matrix(cly[inds],K,Ntest)))
    end = Sys.time()
    res = end-start
    
    my_routine = my_routine_keops_double
    start = Sys.time()
    inds = 1 + as.integer(as.vector(my_routine(list(x,y),Ntest,N)))
    cl2 = round(colMeans(matrix(cly[inds],K,Ntest)))
    end = Sys.time()
    res = c(res,end-start)
    
    # compare with standard R knn
    if(N<7000)
    {
        start = Sys.time()
        cl3 = knn(t(y),t(x),cly,k=K)
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
colnames(res) = c("Npoints","R (K**** float)","R (K**** double)","R (kNN{class})")
res[,1] = Ns
Ntry = 10
for(l in 1:nN)
{
    resl = 0
    for(i in 1:Ntry)
        resl = resl + KNNExample(N=Ns[l],Ntest=10000,D=100,K=10)
    res[l,2:4] = resl / Ntry
}
res = res[,c(1,4,2,3)]
print("")
print("Timings:")
print(res)
