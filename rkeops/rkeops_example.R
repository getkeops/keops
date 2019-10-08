# install.packages("devtools") if necessary
# devtools::install("rkeops") # or edit path to rkeops sub-directory

library(rkeops)
# clean_rkeops()

# rkeops options (compile and runtime)
get_rkeops_options()

# option names
rkeops_option_names()

# to get a specific option: get_rkeops_option(<name>)
# to set a specific option: set_rkeops_option(<name>, <value>)
# in particular, to compute on GPU: set_rkeops_option("tagCpuGpu", 1)


## exemple 1
formula = "Sum_Reduction((x|y), 1)"
args = c("x=Vi(3)", "y=Vj(3)")

op <- keops_kernel(formula, args)

n <- 10
p <- 15
x <- matrix(runif(n*3),3,n)
y <- matrix(runif(p*3),3,p)

input <- list(x, y)
res <- op(input, nx=ncol(x), ny=ncol(y))
str(res)
expected_res <- apply(t(x) %*% y, 2, sum)
sum(abs(res - expected_res))

## exemple 2
formula = "Sum_Reduction((x|y), 0)"
args = c("x=Vi(3)", "y=Vj(3)")

op <- keops_kernel(formula, args)

n <- 10
p <- 15
x <- matrix(runif(n*3),3,n)
y <- matrix(runif(p*3),3,p)

input <- list(x, y)
res <- op(input, nx=ncol(x), ny=ncol(y))
str(res)
expected_res <- apply(t(x) %*% y, 1, sum)
sum(abs(res - expected_res))

## exemple 3
formula = "Sum_Reduction(Exp(lambda*SqNorm2(x-y))*beta, 1)"
args = c("x=Vi(3)", "y=Vj(3)", "beta=Vj(3)", "lambda=Pm(1)")

op <- keops_kernel(formula, args)

nx = 1000
ny = 1500
x <- matrix(runif(nx*3),3,nx)
y <- matrix(runif(ny*3),3,ny)
beta <- matrix(runif(ny*3),3,ny)

lambda <- as.matrix(5)

input <- list(x, y, beta, lambda)

res <- op(input, nx=ncol(x), ny=ncol(y))
str(res)
