# =======================================================
# Gaussian convolution algorithm using the generic syntax
# =======================================================
# 
# We define three datasets x (M samples) and y, b (N samples) of points in R^D,
# then apply a gaussian convolution operation :
# out_i = sum_j exp(-||x_i-y_j||^2/(2sigma^2)) b_j

library(dplyr)
library(ggplot2)
library(rkeops)

GaussConvExample = function(M, N, D) {
    
    print("--------------------")
    print(paste("Gaussian convolution with M=",M,", N=",N,", D=",D, sep=""))
    
    x = matrix(runif(M*D), M, D)
    y = matrix(runif(N*D), N, D)
    b = matrix(runif(N*D), N, D)
    lambda = matrix(1.0 / 0.25^2)
    
    formula = paste('Sum_Reduction(Exp(-lambda*SqDist(x,y))*b,0)', sep="")
    var1 = paste('x=Vi(', D, ')', sep="")  # First arg   : i-variable, of size D
    var2 = paste('y=Vj(', D, ')', sep="")  # Second arg  : j-variable, of size D
    var3 = paste('b=Vj(', D, ')', sep="")  # Third arg   : j-variable, of size D
    var4 = paste('lambda=Pm(1)', sep="")   # Fourth arg  : parameter, of size 1
    variables = c(var1, var2, var3, var4)
    
    set_rkeops_option("precision", "float")
    op_keops_float = keops_kernel(formula, variables)
    set_rkeops_option("precision", "double")
    op_keops_double = keops_kernel(formula, variables)
    
    # Using LazyTensors
    x_i <- Vi(x)
    y_j <- Vj(y)
    b_j <- Vj(b)
    lambda_for_LT <- 1.0 / 0.25^2
    Pm_lambda <- Pm(lambda_for_LT)
    
    # Standard R
    op_nokeops = function(args) {
        x = args[[1]]
        y = args[[2]]
        b = args[[3]]
        lambda = args[[4]]
        M = nrow(x)
        N = nrow(y)
        SqDist = matrix(0, M, N)
        onesM = matrix(1, M, 1)
        onesN = matrix(1, N, 1)
        for(k in 1:D)
            SqDist = SqDist + (as.matrix(x[,k]) %*% t(onesN) - onesM %*% t(as.matrix(y[,k])))^2
        K = exp(-lambda[1] * SqDist)
        out = K %*% b
        return(out)
    }
    
    op_keops = op_keops_float
    start = Sys.time()
    out1 = op_keops(list(x, y, b, lambda))
    end = Sys.time()
    time1 = as.numeric(end - start)
    
    op_keops = op_keops_double
    start = Sys.time()
    out2 = op_keops(list(x, y, b, lambda))
    end = Sys.time()
    time2 = as.numeric(end - start)
    
    # LazyTensor compilation
    set_rkeops_option("precision", "float")
    start = Sys.time()
    res_float <- sum(exp(-Pm_lambda*sqdist(x_i, y_j))*b_j, 'j')
    end = Sys.time()
    timeLT1 = as.numeric(end - start)
    
    set_rkeops_option("precision", "double")
    start = Sys.time()
    res_double <- sum(exp(-Pm_lambda*sqdist(x_i, y_j))*b_j, 'j')
    end = Sys.time()
    timeLT2 = as.numeric(end - start)
    
    # compare with standard R implementation via matrices
    out3 = NULL
    time3 = NA
    if(M<15000) {
        start = Sys.time()
        out3 = op_nokeops(list(x, y, b, lambda))
        end = Sys.time()
        time3 = as.numeric(end - start)
    }
    
    print(paste("mean errors between rkeops double and float version: ",
                mean(abs(out1-out2)), sep=""))
    
    if(!is.null(out3)) {
        print(paste("mean errors between rkeops double and R version: ",
                    mean(abs(out2-out3)), sep=""))
    }
    
    out = data.frame(
        method = c("RKeOps float", "RKeOps double", "R", "LT float", "LT double"),
        time = c(time1, time2, time3, timeLT1, timeLT2),
        dim = N,
        stringsAsFactors = FALSE
    )
    return(out)
}


dim_value = c(100,200,500,1000,2000,5000,10000,20000) #,50000,100000,200000,500000,1000000)

n_rep = 5

param_grid = expand.grid(dim = dim_value, rep = 1:n_rep, KEEP.OUT.ATTRS = FALSE)

experiment = Reduce(
    "bind_rows",
    lapply(
        split(param_grid, seq(nrow(param_grid))), 
        function(config) {
            return(GaussConvExample(config$dim, config$dim, 4))
        }
    )
)

experiment_summary <- experiment %>% 
    group_by(method, dim) %>%
    dplyr::summarize(mean_time = mean(time, na.rm=TRUE))

ggplot(experiment, aes(x=dim, y=time, group=method, col=method)) +
    geom_smooth() +
    theme_bw()

ggplot(experiment_summary, aes(x=dim, y=mean_time, group=method, col=method)) +
    geom_point() + geom_line() +
    theme_bw()
