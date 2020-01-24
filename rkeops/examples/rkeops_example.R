library(rkeops)

# rkeops options (compile and runtime)
get_rkeops_options()

# option names
rkeops_option_names()

# to clean rkeops build directory
clean_rkeops()

# to get a specific option: get_rkeops_option(<name>)
# to set a specific option: set_rkeops_option(<name>, <value>)
# or direct functions:
# use_gpu() # to compute on GPU
# use_cpu() # to compute on CPU
# compile4float32() # to compile with float 32bits precision
# compile4float64() # to compile with float 64bits or double precision

## Example 1
message("## Example 1")
# Defining a function that computes for each j the sum over i
# of the scalar products between `x_i` and `y_j` (both 3d vectors),
# i.e. the sum over the rows of the result of the matrix product `X * t(Y)`
# where `x_i` and `y_j` are the respective rows of the matrices `X` and `Y`.
op <- keops_kernel(formula = "Sum_Reduction((x|y), 1)",
                   args = c("x=Vi(3)", "y=Vj(3)"))
# data
nx <- 1000
ny <- 1500
# x_i = rows of the matrix X
X <- matrix(runif(nx*3), nrow=nx, ncol=3)
# y_j = rows of the matrix Y
Y <- matrix(runif(ny*3), nrow=ny, ncol=3)
# computing the result (here, by default `inner_dim=1` and columns corresponds
# to the inner dimension)
res <- op(list(X,Y))

## Example 1 bis
message("## Example 1 bis")
# In example 1, the inner dimension (i.e. the commun dimension of vectors
# `x_i` and `y_j` corresponds to columns of the matrices `X` and `Y` resp.).
# We know consider the inner dimension to be the rows of the matrices `X`
# and `Y`.

# data
nx <- 1000
ny <- 1500
# x_i = columns of the matrix X
X <- matrix(runif(nx*3), nrow=3, ncol=nx)
# y_j = columns of the matrix Y
Y <- matrix(runif(ny*3), nrow=3, ncol=ny)
# computing the result (we specify `inner_dim=0` to indicate that rows
# corresponds to the inner dimension)
res <- op(list(X,Y), inner_dim=0)

## Example 2
message("## Example 2")
# Defining a function that computes the convolution with a Gaussian kernel
# i.e. the sum over i of `e^(lambda * ||x_i - y_j||^2) * beta_j` where `x_i`,
# `y_j` and `beta_j` are 3d vectors, and `lambda` is a scalar parameter.
op <- keops_kernel(formula = "Sum_Reduction(Exp(lambda*SqNorm2(x-y))*beta, 1)",
                   args = c("x=Vi(3)", "y=Vj(3)",
                            "beta=Vj(3)", "lambda=Pm(1)"))

# data
nx <- 1000
ny <- 1500
# x_i = rows of the matrix X
X <- matrix(runif(nx*3), nrow=nx, ncol=3)
# y _j = rows of the matrix Y
Y <- matrix(runif(ny*3), nrow=ny, ncol=3)
# beta_j = rows of the matrix beta
beta <- matrix(runif(ny*3), nrow=ny, ncol=3)
# !! important !! y and beta should have the same dimension

# parameter
lambda <- 0.25

# computing the result
res <- op(list(X, Y, beta, lambda))


## Example 3
message("## Example 3")
# define an operator
formula <- "Sum_Reduction(SqNorm2(x-y), 0)"
args <- c("x=Vi(0,3)", "y=Vj(1,3)")
op <- keops_kernel(formula, args)
# gradient regarding input variable 'x'
grad_op <- keops_grad(op, var="x")

# data
nx <- 1000
ny <- 1500
# x_i = rows of the matrix X
X <- matrix(runif(nx*3), nrow=nx, ncol=3)
# y_j = rows of the matrix Y
Y <- matrix(runif(ny*3), nrow=ny, ncol=3)
# computing the result
res <- op(list(X,Y))
# computing the gradient
eta <- matrix(1, nrow=nx, ncol=1)
res <- grad_op(list(X, Y, eta))
