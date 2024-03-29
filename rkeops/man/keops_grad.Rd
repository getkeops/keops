% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/keops_grad.R
\name{keops_grad}
\alias{keops_grad}
\title{Compute the gradient of a rkeops operator}
\usage{
keops_grad(operator, var)
}
\arguments{
\item{operator}{a function returned by \code{keops_kernel} implementing a
formula.}

\item{var}{a text string or an integer number indicating regarding to which
variable/parameter (given by name or by position starting at 0) the
gradient of the formula should be computed.}
}
\value{
a function that can be used to compute the value of the formula
on actual data. This function takes as input a list of data corresponding
to the formula arguments and return the computed values (generally a
vector or a matrix depending on the reduction). It has an additional integer
input parameter \code{inner_dim} indicating if the inner dimension
(c.f. \code{browseVignettes("rkeops")}) corresponds to columns, i.e.
\code{inner_dim="col"} (default), or rows, i.e. \code{inner_dim="row"}, in the data.
}
\description{
The function \code{keops_grad} defines a new operator that is a partial derivative
from a previously defined KeOps operator supplied as input regarding a
specified input variable of this operator.
}
\details{
The use of the function \code{keops_grad} is detailed in the vignettes.
Run \code{browseVignettes("rkeops")} to access the vignettes.

KeOps gradient operators are defined based on KeOps formula and on operator
\code{Grad}. The function \code{keops_grad} is a wrapper to define a new formula
deriving the gradient of the formula associated to a previously defined
operator. The user just needs to choose regarding which variable (given by
name or by position starting at 0), they want to compute the partial
derivative.

The function \code{keops_grad} then calls the function \code{\link[=keops_kernel]{keops_kernel()}}
to compile a new operator corresponding to the partial derivative of the
input operator.

To decide regarding which variable the input operator should be derived,
you can specify its name or its position starting at 0 with the input
parameter \code{var}.

\strong{Important:} Formally, KeOps computes the differential conjugate operator,
thus the new corresponding operator defined will expect an additional
input variable (\code{eta} in the examples) of the some type (\code{Vi}, \code{Vj}, \code{Pm})
as the variable chosen to differentiate the formula and of dimension \code{1}
so that the final computation give the corresponding gradient.
}
\examples{
\dontrun{
set_rkeops_options()

# defining an operator (reduction on squared distance)
formula <- "Sum_Reduction(SqNorm2(x-y), 0)"
args <- c("x=Vi(0,3)", "y=Vj(1,3)")
op <- keops_kernel(formula, args)
# defining its gradient regarding x
grad_op <- keops_grad(op, var="x")

# data
nx <- 100
ny <- 150
x <- matrix(runif(nx*3), nrow=nx, ncol=3)     # matrix 100 x 3
y <- matrix(runif(ny*3), nrow=ny, ncol=3)     # matrix 150 x 3
eta <- matrix(1, nrow=nx, ncol=1)             # matrix 100 x 1

# computation
input <- list(x, y, eta)
res <- grad_op(input)

# OR you can directly define gradient in a formula
# defining a formula with a Gradient
formula <- "Grad(Sum_Reduction(SqNorm2(x-y), 0), x, eta)"
args <- c("x=Vi(0,3)", "y=Vj(1,3)", "eta=Vi(2,1)")
# compiling the corresponding operator
op <- keops_kernel(formula, args)

# data
nx <- 100
ny <- 150
x <- matrix(runif(nx*3), nrow=nx, ncol=3)     # matrix 100 x 3
y <- matrix(runif(ny*3), nrow=ny, ncol=3)     # matrix 150 x 3
eta <- matrix(1, nrow=nx, ncol=1)             # matrix 100 x 1

# computation
input <- list(x, y, eta)
res <- op(input)
}
}
\seealso{
\code{\link[=keops_kernel]{keops_kernel()}}
}
\author{
Ghislain Durif
}
