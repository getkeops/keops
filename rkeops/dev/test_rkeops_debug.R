library(rkeops)
set_rkeops_option("verbosity", 1)
set_rkeops_option("debug", 1)

op <- keops_kernel(formula = "Sum_Reduction((x|y), 1)",
                   args = c("x=Vi(3)", "y=Vj(3)"))
# data
nx <- 1000
ny <- 1500
X <- matrix(runif(nx*3), nrow=nx, ncol=3)
Y <- matrix(runif(ny*3), nrow=ny, ncol=3)

res <- op(list(X,Y))
