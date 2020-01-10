# test definition, compilation and use of rkeops operators
project_root_dir <- system("git rev-parse --show-toplevel", intern=TRUE)
devtools::load_all(file.path(project_root_dir, "rkeops"))
set_rkeops_options()
clean_rkeops()

# rkeops options (compile and runtime)
get_rkeops_options()

# to get a specific option: get_rkeops_option(<name>)
# to set a specific option: set_rkeops_option(<name>, <value>)
# in particular, to compute on GPU: set_rkeops_option("tagCpuGpu", 1)
set_rkeops_option("precision", "double")
set_rkeops_option("verbosity", 0)

## exemple 1
message("## Example 1")
formula = "Sum_Reduction((x|y), 1)"
args = c("x=Vi(3)", "y=Vj(3)")

op <- keops_kernel(formula, args)

nx <- 10
ny <- 15
x <- matrix(runif(nx*3), nrow=3, ncol=nx)
y <- matrix(runif(ny*3), nrow=3, ncol=ny)

input <- list(x, y)
res <- op(input, inner_dim=0)

expected_res <- colSums(t(x) %*% y)
print(dim(res))
sum(abs(res - expected_res))

## exemple 1 bis
nx <- 10
ny <- 15
x <- t(matrix(runif(nx*3), nrow=3, ncol=nx))
y <- t(matrix(runif(ny*3), nrow=3, ncol=ny))

input <- list(x, y)
res <- op(input, inner_dim=1)

expected_res <- colSums(x %*% t(y))
print(dim(res))
sum(abs(res - expected_res))

## exemple 2
message("## Example 2")
formula = "Sum_Reduction((x|y), 0)"
args = c("x=Vi(3)", "y=Vj(3)")

op <- keops_kernel(formula, args)

nx <- 10
ny <- 15
x <- matrix(runif(nx*3), nrow= 3, ncol=nx)
y <- matrix(runif(ny*3), nrow=3, ncol=ny)

input <- list(x, y)
res <- op(input, inner_dim=0)
print(dim(res))
expected_res <- apply(t(x) %*% y, 1, sum)
print(cbind(t(res), expected_res))
sum(abs(res - expected_res))

## exemple 3
message("## Example 3")
formula = "Sum_Reduction(Exp(-lambda*SqNorm2(x-y))*beta, 1)"
args = c("x=Vi(3)", "y=Vj(3)", "beta=Vj(3)", "lambda=Pm(1)")

op <- keops_kernel(formula, args)

nx = 10
ny = 15
x <- matrix(runif(nx*3), nrow=3, ncol=nx)
y <- matrix(runif(ny*3), nrow=3, ncol=ny)
beta <- matrix(runif(ny*3), nrow=3, ncol=ny)

lambda <- 0.5

input <- list(x, y, beta, lambda)

res <- op(input, inner_dim=0)
print(dim(res))

expected_res <- sapply(1:ny, function(j) {
    tmp <- sapply(1:nx, function(i) {
        return(exp(- lambda * sum((x[,i]-y[,j])^2)) * beta[,j])
    })
    return(apply(tmp,1,sum))
})
sum(abs(res - expected_res))

