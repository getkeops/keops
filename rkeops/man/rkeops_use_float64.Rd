% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/rkeops_options.R
\name{rkeops_use_float64}
\alias{rkeops_use_float64}
\title{Use 64bit float precision in computations}
\usage{
rkeops_use_float64()
}
\value{
None
}
\description{
Set up \code{rkeops} options to use 64bit float precision in computation
when calling user-defined operators.
}
\details{
By default, \code{rkeops} uses 32bit float precision in computation.
It is mandatory to call \code{rkeops_use_float64()} to enable 64bit float
precision in computation.

\strong{Important:} 32bit float precision computations are faster than
64bit float, however the lower precision may have a huge effect on the
accuracy of your computation and validity of your results
in certain applications.
}
\examples{
\dontrun{
rkeops_use_float64()
}
}
\seealso{
\code{\link[=rkeops_use_float32]{rkeops_use_float32()}}, \code{\link[=set_rkeops_options]{set_rkeops_options()}}
}
\author{
Ghislain Durif
}
