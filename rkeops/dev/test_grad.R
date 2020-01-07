# test definition, compilation and use of rkeops operators
project_root_dir <- system("git rev-parse --show-toplevel", intern=TRUE)
devtools::load_all(file.path(project_root_dir, "rkeops"))
set_rkeops_options()
clean_rkeops()

# Grad
message("## testing grad")
formula <- "Grad(Sum_Reduction(SqNorm2(x-y), 0), x, eta)"
args <- c("x=Vi(0,3)", "y=Vj(1,3)", "eta=Vi(2,1)")
op <- keops_kernel(formula, args)

nx <- 10
ny <- 15
x <- matrix(runif(nx*3), nrow=3, ncol=nx)
y <- matrix(runif(ny*3), nrow=3, ncol=ny)
eta <- matrix(runif(nx*1), nrow=1, ncol=nx)

input <- list(x, y, eta)
res <- op(input, inner_dim=0)


# GramFromPos
message("## testing GradFromPos")
formula <- "GradFromPos(Sum_Reduction(SqNorm2(x-y), 0), x, 2)"
args <- c("x=Vi(0,3)", "y=Vj(1,3)")
op <- keops_kernel(formula, args)
res2 <- op(input, inner_dim=0)

# GramFromPos
message("## testing GradFromInd")
formula <- "GradFromInd(Sum_Reduction(SqNorm2(x-y), 0), 0, 2)"
args <- c("x=Vi(0,3)", "y=Vj(1,3)")
op <- keops_kernel(formula, args)
res3 <- op(input, inner_dim=0)


# keops_grad
message("## keops_grad")
formula <- "Sum_Reduction(SqNorm2(x-y), 0)"
args <- c("x=Vi(0,3)", "y=Vj(1,3)")
op <- keops_kernel(formula, args)

grad_op <- keops_grad(op, var=0)

nx <- 10
ny <- 15
x <- matrix(runif(nx*3), nrow=3, ncol=nx)
y <- matrix(runif(ny*3), nrow=3, ncol=ny)
eta <- 1

input <- list(x, y, eta)
res <- grad_op(input, inner_dim=0)
