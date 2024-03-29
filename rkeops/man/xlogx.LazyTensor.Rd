% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/lazytensor_operations.R
\name{xlogx.LazyTensor}
\alias{xlogx.LazyTensor}
\title{Element-wise \code{x*log(x)} operation}
\usage{
\method{xlogx}{LazyTensor}(x)
}
\arguments{
\item{x}{a \code{LazyTensor}, a \code{ComplexLazyTensor}, a vector of numeric values,
or a scalar value.}
}
\value{
An object of class \code{LazyTensor}.
}
\description{
Symbolic element-wise \code{x*log(x)} function for \code{LazyTensor} objects.
}
\details{
If \code{x} is a \code{LazyTensor}, \code{square(x)} returns a \code{LazyTensor} that
encodes, symbolically, the element-wise \code{x*log(x)} values for \code{x}.

\strong{Note}: by convention \code{xlogx(0)} returns \code{0}.
}
\examples{
\dontrun{
x <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows, 3 columns
x_i <- LazyTensor(x, index = 'i')   # creating LazyTensor from matrix x, 
                                    # indexed by 'i'
xlog_x <- xlogx(x_i)                # symbolic matrix, 150 rows and 3 columnsjs
}
}
\seealso{
\code{\link[=xlogx]{xlogx()}}
}
\author{
Chloe Serre-Combe, Amelie Vernay
}
