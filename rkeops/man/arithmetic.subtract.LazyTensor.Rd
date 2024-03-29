% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/lazytensor_operations.R
\name{arithmetic.subtract.LazyTensor}
\alias{arithmetic.subtract.LazyTensor}
\alias{-.LazyTensor}
\alias{-.ComplexLazyTensor}
\title{Subtraction or minus sign}
\usage{
## S3 method for class 'LazyTensor'
\special{-x}
\special{x - y}

## S3 method for class 'ComplexLazyTensor'
\special{-x}
\special{x - y}
}
\arguments{
\item{x, y}{a \code{LazyTensor}, a \code{ComplexLazyTensor}, a vector of numeric values,
or a scalar value.}
}
\value{
An object of class \code{LazyTensor}.
}
\description{
Symbolic subtraction for \code{LazyTensor} objects.
}
\details{
\strong{Binary operator}: If \code{x} or \code{y} is a \code{LazyTensor}, \code{x - y} returns a
\code{LazyTensor} that encodes, symbolically, the subtraction of \code{x} and \code{y}.
(In case one of the arguments is a vector or a scalar, it is first converted
to \code{LazyTensor}).

\strong{Unary operator}: If \code{x} is a \code{LazyTensor}, then \code{- x} returns a
\code{LazyTensor} that encodes, symbolically, the opposite of \code{x}.

\strong{Note}: \code{x} and \code{y} input arguments should have the same inner dimension
or be of dimension 1.
}
\examples{
\dontrun{
x <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows, 3 columns
y <- matrix(runif(250 * 3), 250, 3) # arbitrary R matrix, 250 rows, 3 columns
x_i <- LazyTensor(x, index = 'i')   # creating LazyTensor from matrix x, 
                                    # indexed by 'i'
y_j <- LazyTensor(y, index = 'j')   # creating LazyTensor from matrix y, 
                                    # indexed by 'j'
Sub_xy <- x_i - y_j                 # symbolic matrix
Minus_x <- -x_i                     # symbolic matrix
}
}
\seealso{
\code{\link[=-]{-()}}
}
\author{
Chloe Serre-Combe, Amelie Vernay
}
