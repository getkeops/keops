% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/lazytensor_operations.R
\name{complex.default}
\alias{complex.default}
\alias{Re.default}
\alias{Im.default}
\alias{Arg.default}
\alias{Conj.default}
\alias{Mod.default}
\title{Complex Numbers and Basic Functionality}
\usage{
\method{Re}{default}(z)

\method{Im}{default}(z)

\method{Arg}{default}(z)

\method{Conj}{default}(z)

\method{Mod}{default}(z)
}
\arguments{
\item{z}{an object of mode \code{complex}, or one of a class for which
    a methods has been defined.}
}
\description{
Basic functions which support complex arithmetic in \R, in addition to
  the arithmetic operators \code{+}, \code{-}, \code{*}, \code{/}, and \code{^}.
}
\details{
Complex vectors can be created with \code{complex}.  The vector can be
  specified either by giving its length, its real and imaginary parts, or
  modulus and argument.  (Giving just the length generates a vector of
  complex zeroes.)

  \code{as.complex} attempts to coerce its argument to be of complex
  type: like \code{\link[base]{as.vector}} it strips attributes including
  names.
  Since \R version 4.4.0, \code{as.complex(x)} for \dQuote{number-like}
  \code{x}, i.e., types \code{"logical"}, \code{"integer"}, and
  \code{"double"}, will always keep imaginary part zero, now also for
  \code{NA}'s.
    Up to \R versions 3.2.x, all forms of \code{NA} and \code{NaN}
  were coerced to a complex \code{NA}, i.e., the \code{\link[base]{NA_complex_}}
  constant, for which both the real and imaginary parts are \code{NA}.
  Since \R 3.3.0, typically only objects which are \code{NA} in parts
  are coerced to complex \code{NA}, but others with \code{NaN} parts,
  are \emph{not}.  As a consequence, complex arithmetic where only
  \code{NaN}'s (but no \code{NA}'s) are involved typically will
  \emph{not} give complex \code{NA} but complex numbers with real or
  imaginary parts of \code{NaN}.
  All of these many different complex numbers fulfill \code{is.na(.)} but
  only one of them is identical to \code{NA_complex_}.

  Note that \code{is.complex} and \code{is.numeric} are never both
  \code{TRUE}.

  The functions \code{Re}, \code{Im}, \code{Mod}, \code{Arg} and
  \code{Conj} have their usual interpretation as returning the real
  part, imaginary part, modulus, argument and complex conjugate for
  complex values.  The modulus and argument are also called the \emph{polar
  coordinates}.  If \eqn{z = x + i y} with real \eqn{x} and \eqn{y}, for
  \eqn{r = Mod(z) = \sqrt{x^2 + y^2}}{r = Mod(z) = \\sqrt(x^2 + y^2)},
  and \eqn{\phi = Arg(z)}, \eqn{x = r \cos(\phi)}{x = r*cos(\\phi)} and
  \eqn{y = r \sin(\phi)}{y = r*sin(\\phi)}.  They are all
  \link[base]{internal generic} \link[base]{primitive} functions: methods can be
  defined for them
  individually or \emph{via} the \code{\link[base:S3groupGeneric]{Complex}}
  group generic.

  In addition to the arithmetic operators (see \link[base]{Arithmetic})
  \code{+}, \code{-}, \code{*}, \code{/}, and \code{^}, the elementary
  trigonometric, logarithmic, exponential, square root and hyperbolic
  functions are implemented for complex values.

  Matrix multiplications (\code{\link[base]{\%*\%}}, \code{\link[base]{crossprod}},
  \code{\link[base]{tcrossprod}}) are also defined for complex matrices
  (\code{\link[base]{matrix}}), and so are \code{\link[base]{solve}},
  \code{\link[base]{eigen}} or \code{\link[base]{svd}}.

  Internally, complex numbers are stored as a pair of \link[base]{double}
  precision numbers, either or both of which can be \code{\link[base]{NaN}}
  (including \code{NA}, see \code{\link[base]{NA_complex_}} and above) or
  plus or minus infinity.
}
\examples{
require(graphics)

0i ^ (-3:3)

matrix(1i^ (-6:5), nrow = 4) #- all columns are the same
0 ^ 1i # a complex NaN

## create a complex normal vector
z <- complex(real = stats::rnorm(100), imaginary = stats::rnorm(100))
## or also (less efficiently):
z2 <- 1:2 + 1i*(8:9)

## The Arg(.) is an angle:
zz <- (rep(1:4, length.out = 9) + 1i*(9:1))/10
zz.shift <- complex(modulus = Mod(zz), argument = Arg(zz) + pi)
plot(zz, xlim = c(-1,1), ylim = c(-1,1), col = "red", asp = 1,
     main = expression(paste("Rotation by "," ", pi == 180^o)))
abline(h = 0, v = 0, col = "blue", lty = 3)
points(zz.shift, col = "orange")

## as.complex(<some NA>): numbers keep Im = 0:
stopifnot(identical(as.complex(NA_real_), NA_real_ + 0i)) # has always been true
NAs <- vapply(list(NA, NA_integer_, NA_real_, NA_character_, NA_complex_),
              as.complex, 0+0i)
stopifnot(is.na(NAs), is.na(Re(NAs))) # has always been true
showC <- function(z) noquote(paste0("(", Re(z), ",", Im(z), ")"))
showC(NAs)
Im(NAs) # [0 0 0 NA NA]  \\ in R <= 4.3.x was [NA NA 0 NA NA]
stopifnot(Im(NAs)[1:3] == 0)


## The exact result of this *depends* on the platform, compiler, math-library:
(NpNA <- NaN + NA_complex_) ; str(NpNA) # *behaves* as 'cplx NA' ..
stopifnot(is.na(NpNA), is.na(NA_complex_), is.na(Re(NA_complex_)), is.na(Im(NA_complex_)))
showC(NpNA)# but does not always show '(NaN,NA)'
## and this is not TRUE everywhere:
identical(NpNA, NA_complex_)
showC(NA_complex_) # always == (NA,NA)
}
\seealso{
\code{\link[base:complex]{base::Re()}}

\code{\link[base:complex]{base::Im()}}

\code{\link[base:complex]{base::Arg()}}

\code{\link[base:complex]{base::Conj()}}

\code{\link[base:complex]{base::Mod()}}
}
\author{
R core team and contributors
}
