% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/lazytensor_operations.R
\name{round.default}
\alias{round.default}
\title{Rounding function}
\usage{
\method{round}{default}(x, digits = 0)
}
\arguments{
\item{x}{a numeric vector.  Or, for \code{round} and \code{signif}, a
    complex vector.}

\item{digits}{integer indicating the number of decimal places
    (\code{round}) or significant digits (\code{signif}) to be used.
    For \code{round}, negative values are allowed (see \sQuote{Details}).}
}
\description{
\code{ceiling} takes a single numeric argument \code{x} and returns a
  numeric vector containing the smallest integers not less than the
  corresponding elements of \code{x}.

  \code{floor} takes a single numeric argument \code{x} and returns a
  numeric vector containing the largest integers not greater than the
  corresponding elements of \code{x}.

  \code{trunc} takes a single numeric argument \code{x} and returns a
  numeric vector containing the integers formed by truncating the values in
  \code{x} toward \code{0}.

  \code{round} rounds the values in its first argument to the specified
  number of decimal places (default 0).  See \sQuote{Details} about
  \dQuote{round to even} when rounding off a 5.

  \code{signif} rounds the values in its first argument to the specified
  number of \emph{significant} digits.   Hence, for \code{numeric} \code{x},
  \code{signif(x, dig)} is the same as \code{round(x, dig - ceiling(log10(abs(x))))}.
}
\details{
These are generic functions: methods can be defined for them
  individually or via the \code{\link[base:S3groupGeneric]{Math}} group
  generic.

  Note that for rounding off a 5, the \abbr{IEC} 60559 standard (see also
  \sQuote{IEEE 754}) is expected to be used, \sQuote{\emph{go to the even digit}}.
  Therefore \code{round(0.5)} is \code{0} and \code{round(-1.5)} is
  \code{-2}.  However, this is dependent on OS services and on
  representation error (since e.g.\ifelse{latex}{\out{~}}{ }{}\code{0.15} is not represented
  exactly, the rounding rule applies to the represented number and not
  to the printed number, and so \code{round(0.15, 1)} could be either
  \code{0.1} or \code{0.2}).

  Rounding to a negative number of digits means rounding to a power of
  ten, so for example \code{round(x, digits = -2)} rounds to the nearest
  hundred.

  For \code{signif} the recognized values of \code{digits} are
  \code{1...22}, and non-missing values are rounded to the nearest
  integer in that range. Each element of the vector is rounded individually, 
  unlike printing.

  These are all primitive functions.
}
\examples{
round(.5 + -2:4) # IEEE / IEC rounding: -2  0  0  2  2  4  4
## (this is *good* behaviour -- do *NOT* report it as bug !)

( x1 <- seq(-2, 4, by = .5) )
round(x1) #-- IEEE / IEC rounding !
x1[trunc(x1) != floor(x1)]
x1[round(x1) != floor(x1 + .5)]
(non.int <- ceiling(x1) != floor(x1))

x2 <- pi * 100^(-1:3)
round(x2, 3)
signif(x2, 3)
}
\seealso{
\code{\link[base:Round]{base::round()}}
}
\author{
R core team and contributors
}
