
# ARITHMETIC OPERATIONS ========================================================


# addition ---------------------------------------------------------------------

#' @title Default arithmetic operations
#' @name default.arithmetic.fun
#' @aliases +.default
#' @usage
#' ## Default S3 method:
#' \special{+x}
#' \special{x + y}
#' @inherit base::Arithmetic description
#' @inherit base::Arithmetic details
#' @inherit base::Arithmetic params
#' @inherit base::Arithmetic return
#' @inherit base::Arithmetic examples
#' @seealso [base::Arithmetic]
#' @author R core team and contributors
#' @export
"+.default" <- function(x, y = NULL) {
    if(!is.null(y)) {
        return(base::"+"(x, y))
    }
    else {
        return(base::"+"(x))
    }
}


#' Addition
#' @name arithmetic.add
#' @aliases +
#' @author Chloe Serre-Combe, Amelie Vernay
#' @description
#' Symbolic addition for `LazyTensor` objects or default addition for other
#' types.
#' @usage
#' \special{+x}
#' \special{x + y}
#' @details
#' If `x` or `y` is a `LazyTensor`, see [rkeops::+.LazyTensor()], else see
#' [rkeops::+.default()].
#'
#' **Note:**
#' - the `+` operand is only a binary operator for `LazyTensor`s: `x + y`.
#' - the `+` operand can be a unary or a binary operator for other types: `+ x`
#' or `x + y`.
#' @param x,y input for [rkeops::+.default()] or [rkeops::+.LazyTensor()].
#' @return See value of [rkeops::+.default()] or [rkeops::+.LazyTensor()].
#' @seealso [rkeops::+.default()], [rkeops::+.LazyTensor()],
#' [rkeops::+.ComplexLazyTensor()]
#' @examples
#' \dontrun{
#' # R base operation
#' +5
#' 1 + 3
#' # LazyTensor symbolic addition
#' x <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows, 3 columns
#' y <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows, 3 columns
#' x_i <- LazyTensor(x, index = 'i')   # creating LazyTensor from matrix x,
#'                                     # indexed by 'i'
#' y_j <- LazyTensor(y, index = 'j')   # creating LazyTensor from matrix y,
#'                                     # indexed by 'j'
#' Sum_xy <- x_i + y_j                 # symbolic matrix, 150 rows and 3 columns
#' }
#' @export
"+" <- function(x, y = NULL) {
    if(!is.null(y) && !is.ComplexLazyTensor(x) && is.LazyTensor(y))
        UseMethod("+", y)
    else
        UseMethod("+", x)
}

#' Addition
#' @name arithmetic.add.LazyTensor
#' @aliases
#' +.LazyTensor
#' @author Chloe Serre-Combe, Amelie Vernay
#' @usage
#' ## S3 method for class 'LazyTensor'
#' \special{+x}
#' \special{x + y}
#' @description
#' Symbolic addition for `LazyTensor` objects.
#' @details If `x` or `y` is a `LazyTensor`, `x + y` returns a `LazyTensor`
#' that encodes, symbolically, the addition of `x` and `y`.
#' (In case one of the arguments is a vector or a scalar, it is first converted 
#' to `LazyTensor`).
#' 
#' **Note**: `x` and `y` input arguments should have the same inner dimension 
#' or be of dimension 1.
#' @param x,y a `LazyTensor`, a `ComplexLazyTensor`, a vector of numeric values,
#' or a scalar value.
#' @return An object of class `LazyTensor`.
#' @seealso [rkeops::+()]
#' @examples
#' \dontrun{
#' x <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows, 3 columns
#' y <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows, 3 columns
#' x_i <- LazyTensor(x, index = 'i')   # creating LazyTensor from matrix x, 
#'                                     # indexed by 'i'
#' y_j <- LazyTensor(y, index = 'j')   # creating LazyTensor from matrix y, 
#'                                     # indexed by 'j'
#' Sum_xy <- x_i + y_j                 # symbolic matrix, 150 rows and 3 columns
#' }
#' @export
"+.LazyTensor" <- function(x, y = NULL) {
    
    if(is.null(y))
        stop("'+' operand not implemented for a single LazyTensor argument")
    else
        res <- binaryop.LazyTensor(
            x, y, "+", is_operator = TRUE, dim_check_type = "sameor1")
    return(res)
}

#' Addition
#' @name arithmetic.add.LazyTensor
#' @aliases +.ComplexLazyTensor
#' @author Chloe Serre-Combe, Amelie Vernay
#' @usage
#' ## S3 method for class 'ComplexLazyTensor'
#' \special{+x}
#' \special{x + y}
#' @export
"+.ComplexLazyTensor" <- function(x, y = NULL) {
    
    if(is.null(y)) {
        msg <- paste0(
            "'+' operand not implemented for",
            "a single ComplexLazyTensor argument"
        )
        stop(msg)
    }
    
    if(!is.LazyTensor(x) && !is.matrix(x)) {
        x <- LazyTensor(x)
    }
    
    if(!is.LazyTensor(y) && !is.matrix(y)) {
        y <- LazyTensor(y)
    }
    
    # convert in complex when needed 
    if(!is.ComplexLazyTensor(y)) {
        res <- x + real2complex(y)
        return(res)
    }
    else if(!is.ComplexLazyTensor(x)) {
        res <- real2complex(x) + y
        return(res)
    }
    
    res <- binaryop.LazyTensor(x, y, "Add")
    return(res)
    
}

# subtraction ------------------------------------------------------------------

#' @title Default arithmetic operations
#' @name default.arithmetic.fun
#' @aliases -.default
#' @usage
#' ## Default S3 method:
#' \special{-x}
#' \special{x - y}
#' @export
"-.default" <- function(x, y = NULL) {
    if(!is.null(y)) {
        return(base::"-"(x, y))
    }
    else {
        return(base::"-"(x))
    }
}

#' Subtraction or minus sign
#' @name arithmetic.subtract
#' @aliases -
#' @author Chloe Serre-Combe, Amelie Vernay
#' @description
#' Symbolic subtraction for `LazyTensor` objects or default subtraction for 
#' otherwise.
#' @usage
#' \special{-x}
#' \special{x - y}
#' @details
#' If `x` or `y` is a `LazyTensor`, see [rkeops::-.LazyTensor()], else see
#' [rkeops::-.default()].
#' 
#' **Note:** the `-` operand can be a binary operator, e.g. `x - y`, 
#' implementing the subtraction, or a unary operator, e.g. `- x`, implementing 
#' the "minus sign", both for `LazyTensor` objects or other types.
#' @param x,y input for [rkeops::-.default()] or [rkeops::-.LazyTensor()].
#' @return See value of [rkeops::-.default()] or [rkeops::-.LazyTensor()].
#' @seealso [rkeops::-.default()], [rkeops::-.LazyTensor()], 
#' [rkeops::-.ComplexLazyTensor()]
#' @examples
#' \dontrun{
#' # R base operation
#' +5
#' 1 + 3
#' # LazyTensor symbolic subtraction
#' x <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows, 3 columns
#' y <- matrix(runif(250 * 3), 250, 3) # arbitrary R matrix, 250 rows, 3 columns
#' x_i <- LazyTensor(x, index = 'i')   # creating LazyTensor from matrix x, 
#'                                     # indexed by 'i'
#' y_j <- LazyTensor(y, index = 'j')   # creating LazyTensor from matrix y, 
#'                                     # indexed by 'j'
#' Sub_xy <- x_i - y_j                 # symbolic matrix
#' Minus_x <- -x_i                     # symbolic matrix
#' }
#' @export
"-" <- function(x, y = NULL) { 
    if(!is.null(y) && (!is.ComplexLazyTensor(x) && is.LazyTensor(y)))
        UseMethod("-", y)
    else
        UseMethod("-", x)
}

#' Subtraction or minus sign
#' @name arithmetic.subtract.LazyTensor
#' @aliases
#' -.LazyTensor
#' @author Chloe Serre-Combe, Amelie Vernay
#' @description
#' Symbolic subtraction for `LazyTensor` objects.
#' @usage
#' ## S3 method for class 'LazyTensor'
#' \special{-x}
#' \special{x - y}
#' @details 
#' **Binary operator**: If `x` or `y` is a `LazyTensor`, `x - y` returns a 
#' `LazyTensor` that encodes, symbolically, the subtraction of `x` and `y`.
#' (In case one of the arguments is a vector or a scalar, it is first converted 
#' to `LazyTensor`).
#' 
#' **Unary operator**: If `x` is a `LazyTensor`, then `- x` returns a 
#' `LazyTensor` that encodes, symbolically, the opposite of `x`.
#' 
#' **Note**: `x` and `y` input arguments should have the same inner dimension 
#' or be of dimension 1.
#' @param x,y a `LazyTensor`, a `ComplexLazyTensor`, a vector of numeric values,
#' or a scalar value.
#' @return An object of class `LazyTensor`.
#' @seealso [rkeops::-()]
#' @examples
#' \dontrun{
#' x <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows, 3 columns
#' y <- matrix(runif(250 * 3), 250, 3) # arbitrary R matrix, 250 rows, 3 columns
#' x_i <- LazyTensor(x, index = 'i')   # creating LazyTensor from matrix x, 
#'                                     # indexed by 'i'
#' y_j <- LazyTensor(y, index = 'j')   # creating LazyTensor from matrix y, 
#'                                     # indexed by 'j'
#' Sub_xy <- x_i - y_j                 # symbolic matrix
#' Minus_x <- -x_i                     # symbolic matrix
#' }
#' @export
"-.LazyTensor" <- function(x, y = NULL) {
    
    if(is.null(y))
        res <- unaryop.LazyTensor(x, "Minus")
    else
        res <- binaryop.LazyTensor(
            x, y, "-", is_operator = TRUE, dim_check_type = "sameor1")
    return(res)
}

#' Subtraction or minus sign
#' @name arithmetic.subtract.LazyTensor
#' @aliases -.ComplexLazyTensor
#' @author Chloe Serre-Combe, Amelie Vernay
#' @usage
#' ## S3 method for class 'ComplexLazyTensor'
#' \special{-x}
#' \special{x - y}
#' @export
"-.ComplexLazyTensor" <- function(x, y = NULL) {
    
    if(is.null(y)) {
        res <- unaryop.LazyTensor(x, "Minus")
        return(res)
    }
    
    if(!is.LazyTensor(x) && !is.matrix(x)) {
        x <- LazyTensor(x)
    }
    
    if(!is.LazyTensor(y) && !is.matrix(y)) {
        y <- LazyTensor(y)
    }
    
    # convert in complex when needed 
    if(!is.ComplexLazyTensor(y)) {
        res <- x - real2complex(y)
        return(res)
    }
    else if(!is.ComplexLazyTensor(x)) {
        res <- real2complex(x) - y
        return(res)
    }
    
    res <- binaryop.LazyTensor(x, y, "Subtract")
    return(res)
    
}


# multiplication  --------------------------------------------------------------

#' @title Default arithmetic operations
#' @name default.arithmetic.fun
#' @aliases *.default
#' @export
"*.default" <- function(x, y) {
    return(base::"*"(x, y))
}

#' Multiplication
#' @name arithmetic.multiply
#' @aliases *
#' @author Chloe Serre-Combe, Amelie Vernay
#' @description
#' Symbolic multiplication for `LazyTensor` objects or default multiplication 
#' otherwise.
#' @details
#' If `x` or `y` is a `LazyTensor`, see [rkeops::*.LazyTensor()], else see
#' [rkeops::*.default()].
#' @param x,y input for [rkeops::*.default()] or [rkeops::*.LazyTensor()].
#' @return See value of [rkeops::*.default()] or [rkeops::*.LazyTensor()].
#' @seealso [rkeops::*.default()], [rkeops::*.LazyTensor()], 
#' [rkeops::*.ComplexLazyTensor()]
#' @examples
#' \dontrun{
#' # R base operation
#' +5
#' 1 + 3
#' # LazyTensor symbolic multiplication
#' x <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows, 3 columns
#' y <- matrix(runif(250 * 3), 250, 3) # arbitrary R matrix, 250 rows, 3 columns
#' x_i <- LazyTensor(x, index = 'i')   # creating LazyTensor from matrix x, 
#'                                     # indexed by 'i'
#' y_j <- LazyTensor(y, index = 'j')   # creating LazyTensor from matrix y, 
#'                                     # indexed by 'j'
#' x_times_y <- x_i * y_j              # symbolic matrix
#' }
#' @export
"*" <- function(x, y) { 
    if(!is.ComplexLazyTensor(x) && is.LazyTensor(y))
        UseMethod("*", y)
    else
        UseMethod("*", x)
}

#' Multiplication
#' @name arithmetic.multiply.LazyTensor
#' @aliases *.LazyTensor
#' @author Chloe Serre-Combe, Amelie Vernay
#' @description
#' Symbolic multiplication for `LazyTensor` objects.
#' @details If `x` or `y` is a `LazyTensor`, `x * y` returns a 
#' `LazyTensor` that encodes, symbolically, the multiplication of `x` and `y`.
#' (In case one of the arguments is a vector or a scalar, it is first converted 
#' to `LazyTensor`).
#' 
#' **Note**: `x` and `y` input arguments should have the same inner dimension 
#' or be of dimension 1.
#' @param x,y a `LazyTensor`, a `ComplexLazyTensor`, a vector of numeric values,
#' or a scalar value.
#' @return An object of class `LazyTensor`.
#' @seealso [rkeops::*()]
#' @examples
#' \dontrun{
#' x <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows, 3 columns
#' y <- matrix(runif(250 * 3), 250, 3) # arbitrary R matrix, 250 rows, 3 columns
#' x_i <- LazyTensor(x, index = 'i')   # creating LazyTensor from matrix x, 
#'                                     # indexed by 'i'
#' y_j <- LazyTensor(y, index = 'j')   # creating LazyTensor from matrix y, 
#'                                     # indexed by 'j'
#' x_times_y <- x_i * y_j              # symbolic matrix
#' }
#' @export
"*.LazyTensor" <- function(x, y) {
    res <- binaryop.LazyTensor(x, y, "*", is_operator = TRUE,
                               dim_check_type = "sameor1")
    return(res)
}

#' Multiplication
#' @name arithmetic.multiply.LazyTensor
#' @aliases *.ComplexLazyTensor
#' @author Chloe Serre-Combe, Amelie Vernay
#' @export
"*.ComplexLazyTensor" <- function(x, y) {
    if(!is.LazyTensor(x) && !is.matrix(x)) {
        x <- LazyTensor(x)
    }
    
    if(!is.LazyTensor(y) && !is.matrix(y)) {
        y <- LazyTensor(y)
    }
    
    # multiplication between a ComplexLazyTensor and a LazyParameter
    if(is.LazyParameter(x) || is.LazyParameter(y)) {
        res <- binaryop.LazyTensor(x, y, "ComplexRealScal")
    }
    
    # multiplication between a ComplexLazyTensor and a LazyTensor
    # the LazyTensor is converted in ComplexLazyTensor
    else if(!is.ComplexLazyTensor(y)) {
        res <- x * real2complex(y)
    }
    else if(!is.ComplexLazyTensor(x)) {
        res <- real2complex(x) * y
    }
    
    # multiplication between 2 'ComplexLazyParameters'
    else if(is.ComplexLazyParameter(x) || is.ComplexLazyParameter(y)) {
        res <- binaryop.LazyTensor(x, y, "ComplexScal", dim_check_type = NA)
    }
    
    # multiplication between 2 ComplexLazyTensors
    else {
        res <- binaryop.LazyTensor(x, y, "ComplexMult")
    }
    
    return(res)
}


# division ---------------------------------------------------------------------

#' @title Default arithmetic operations
#' @name default.arithmetic.fun
#' @aliases /.default
#' @export
"/.default" <- function(x, y) {
    return(base::"/"(x, y))
}

#' Division
#' @name arithmetic.divide
#' @aliases /
#' @author Chloe Serre-Combe, Amelie Vernay
#' @description
#' Symbolic division for `LazyTensor` objects or default division 
#' otherwise.
#' @details
#' If `x` or `y` is a `LazyTensor`, see [rkeops::/.LazyTensor()], else see
#' [rkeops::/.default()].
#' @param x,y input for [rkeops::/.default()] or [rkeops::/.LazyTensor()].
#' @return See value of [rkeops::/.default()] or [rkeops::/.LazyTensor()].
#' @seealso [rkeops::/.default()], [rkeops::/.LazyTensor()], 
#' [rkeops::/.ComplexLazyTensor()]
#' @examples
#' \dontrun{
#' # R base operation
#' +5
#' 1 + 3
#' # LazyTensor symbolic division
#' x <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows, 3 columns
#' y <- matrix(runif(250 * 3), 250, 3) # arbitrary R matrix, 250 rows, 3 columns
#' x_i <- LazyTensor(x, index = 'i')   # creating LazyTensor from matrix x, 
#'                                     # indexed by 'i'
#' y_j <- LazyTensor(y, index = 'j')   # creating LazyTensor from matrix y,
#'                                     # indexed by 'j'
#' x_div_y <- x_i / y_j                # symbolic matrix
#' }
#' @export
"/" <- function(x, y) { 
    if(!is.ComplexLazyTensor(x) && is.LazyTensor(y))
        UseMethod("/", y)
    else
        UseMethod("/", x)
}

#' Division
#' @name arithmetic.divide.LazyTensor
#' @aliases /.LazyTensor
#' @author Chloe Serre-Combe, Amelie Vernay
#' @description
#' Symbolic multiplication for `LazyTensor` objects.
#' @details If `x` or `y` is a `LazyTensor`, `x / y` returns a 
#' `LazyTensor` that encodes, symbolically, the division of `x` and `y`.
#' (In case one of the arguments is a vector or a scalar, it is first converted 
#' to `LazyTensor`).
#' 
#' **Note**: `x` and `y` input arguments should have the same inner dimension 
#' or be of dimension 1.
#' @param x,y a `LazyTensor`, a `ComplexLazyTensor`, a vector of numeric values,
#' or a scalar value.
#' @return An object of class `LazyTensor`.
#' @seealso [rkeops::/()]
#' @examples
#' \dontrun{
#' # LazyTensor symbolic division
#' x <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows, 3 columns
#' y <- matrix(runif(250 * 3), 250, 3) # arbitrary R matrix, 250 rows, 3 columns
#' x_i <- LazyTensor(x, index = 'i')   # creating LazyTensor from matrix x, 
#'                                     # indexed by 'i'
#' y_j <- LazyTensor(y, index = 'j')   # creating LazyTensor from matrix y,
#'                                     # indexed by 'j'
#' x_div_y <- x_i / y_j                # symbolic matrix
#' }
#' @export
"/.LazyTensor" <- function(x, y) {
    res <- binaryop.LazyTensor(x, y, "/", is_operator = TRUE,
                               dim_check_type = "sameor1")
    return(res)
}

#' Division
#' @name arithmetic.divide.LazyTensor
#' @aliases /.ComplexLazyTensor
#' @author Chloe Serre-Combe, Amelie Vernay
#' @export
"/.ComplexLazyTensor" <- function(x, y) {
    if(!is.LazyTensor(x) && !is.matrix(x)) {
        x <- LazyTensor(x)
    }
    
    if(!is.LazyTensor(y) && !is.matrix(y)) {
        y <- LazyTensor(y)
    }
    
    if(!is.ComplexLazyTensor(y)) {
        res <- x / real2complex(y)
        return(res)
    }
    
    else if(!is.ComplexLazyTensor(x)) {
        res <- real2complex(x) / y
        return(res)
    }
    
    res <- binaryop.LazyTensor(x, y, "ComplexDivide")
    return(res)
}


# square -----------------------------------------------------------------------

#' Element-wise square (power-2) operation
#' @author Chloe Serre-Combe, Amelie Vernay
#' @description
#' Element-wise square or power-2 operations on numeric or complex vectors 
#' (or objects which can be coerced to them).
#' @details
#' `square(x)` is equivalent to `x^2` where `^` is the default power function, 
#' i.e. \eqn{x^2}.
#' 
#' From [base::Arithmetic]: if applied to arrays the result will be an array if 
#' this is sensible 
#' (for example it will not if the recycling rule has been invoked).
#' @param x numeric or complex vectors or objects which can be coerced to such, 
#' or other objects for which methods have been written.
#' @return Vector or array of elements from `x` elevated to the power-2.
#' @seealso [base::Arithmetic]
#' @examples
#' square(4)
#' square(1:10)
#' @export
square.default <- function(x) {
    return(x^2)
}

#' Element-wise square (power-2) operation
#' @author Chloe Serre-Combe, Amelie Vernay
#' @description
#' Symbolic element-wise square operation for `LazyTensor` objects or
#' standard element-wise square operation otherwise, i.e. \eqn{x^2}.
#' @details If `x` is a `LazyTensor`, see [rkeops::square.LazyTensor()], else 
#' see [rkeops::square.default()].
#' @param x input for [rkeops::square.default()] or 
#' [rkeops::square.LazyTensor()].
#' @return See value of [rkeops::square.default()] or 
#' [rkeops::square.LazyTensor()]
#' @seealso [rkeops::square.default()], [rkeops::square.LazyTensor()]
#' @examples
#' \dontrun{
#' # Numerical input
#' square(4)
#' square(1:4)
#' # LazyTensor symbolic element-wise square
#' x <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows, 3 columns
#' x_i <- LazyTensor(x, index = 'i')   # creating LazyTensor from matrix x, 
#'                                     # indexed by 'i'
#' Square_x <- square(x_i)             # symbolic matrix, 150 rows and 3 columns
#' }
#' @export
square <- function(x) {
    UseMethod("square", x)
}

#' Element-wise square (power-2) operation
#' @author Chloe Serre-Combe, Amelie Vernay
#' @description
#' Symbolic element-wise square operation for `LazyTensor` objects.
#' @details If `x` is a `LazyTensor`, `square(x)` returns a `LazyTensor` that 
#' encodes, symbolically, the element-wise square of `x`, i.e. \eqn{x^2}.
#' @param x a `LazyTensor`, a `ComplexLazyTensor`, a vector of numeric values,
#' or a scalar value.
#' @return An object of class `LazyTensor`.
#' @seealso [rkeops::square()]
#' @examples
#' \dontrun{
#' x <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows, 3 columns
#' x_i <- LazyTensor(x, index = 'i')   # creating LazyTensor from matrix x, 
#'                                     # indexed by 'i'
#' Square_x <- square(x_i)             # symbolic matrix, 150 rows and 3 columns
#' }
#' @export
square.LazyTensor <- function(x) {
    return(unaryop.LazyTensor(x, "Square"))
}



# square root ------------------------------------------------------------------

#' @title Miscellaneous Mathematical Functions
#' @name default.math.fun
#' @aliases sqrt.default
#' @inherit base::sqrt description
#' @inherit base::sqrt details
#' @inherit base::sqrt params
#' @inherit base::sqrt return
#' @inherit base::sqrt examples
#' @seealso [base::sqrt()]
#' @author R core team and contributors
#' @export
sqrt.default <- function(x) {
    return(base::sqrt(x))
}

#' Element-wise square root operation
#' @author Chloe Serre-Combe, Amelie Vernay
#' @description
#' Symbolic element-wise square root operation for `LazyTensor` objects or
#' standard element-wise square root operation otherwise, i.e. 
#' \eqn{\sqrt{x}}.
#' @details If `x` is a `LazyTensor`, see [rkeops::sqrt.LazyTensor()], else 
#' see [rkeops::sqrt.default()].
#' 
#' **Note**: for complex input argument `z`, then the absolute value 
#' corresponds to the complex modulus `sqrt(z) = z^0.5`.
#' @param x input for [rkeops::sqrt.default()] or [rkeops::sqrt.LazyTensor()].
#' @return See value of [rkeops::sqrt.default()] or [rkeops::sqrt.LazyTensor()].
#' @seealso [rkeops::sqrt.default()], [rkeops::sqrt.LazyTensor()]
#' @examples
#' \dontrun{
#' # R base operation
#' sqrt(4)
#' sqrt(c(1,4,9,16))
#' # LazyTensor symbolic element-wise square root
#' x <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows, 3 columns
#' x_i <- LazyTensor(x, index = 'i')   # creating LazyTensor from matrix x, 
#'                                     # indexed by 'i'
#' Sqrt_x <- sqrt(x_i)                 # symbolic matrix, 150 rows and 3 columns
#' }
#' @export
sqrt <- function(x) { 
    UseMethod("sqrt", x)
}

#' Element-wise square root operation
#' @author Chloe Serre-Combe, Amelie Vernay
#' @description
#' Symbolic element-wise square root operation for `LazyTensor` objects.
#' @details If `x` is a `LazyTensor`, `sqrt(x)` returns a `LazyTensor` that 
#' encodes, symbolically, the element-wise square root of `x`, i.e. 
#' \eqn{\sqrt{x}}.
#' @param x a `LazyTensor`, a `ComplexLazyTensor`, a vector of numeric values,
#' or a scalar value.
#' @return An object of class `LazyTensor`.
#' @seealso [rkeops::sqrt()]
#' @examples
#' \dontrun{
#' x <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows, 3 columns
#' x_i <- LazyTensor(x, index = 'i')   # creating LazyTensor from matrix x, 
#'                                     # indexed by 'i'
#' Sqrt_x <- sqrt(x_i)                 # symbolic matrix, 150 rows and 3 columns
#' }
#' @export
sqrt.LazyTensor <- function(x) {
    return(unaryop.LazyTensor(x, "Sqrt"))
}


# Rsqrt ------------------------------------------------------------------------

#' Element-wise inverse square root operation
#' @author Chloe Serre-Combe, Amelie Vernay
#' @description
#' Element-wise inverse square root on numeric or complex vectors 
#' (or objects which can be coerced to them).
#' @details
#' `rsqrt(x)` is equivalent to `1 / sqrt(x)` where `sqrt()` is the 
#' default square root function, i.e. \eqn{1/\sqrt{x}}.
#' @inheritParams base::sqrt
#' @return a numeric or complex vector or array.
#' @seealso [rkeops::rsqrt()], [rkeops::sqrt()], [base::sqrt()]
#' @examples
#' rsqrt(4)
#' rsqrt(c(1,4,9,16))
#' @export
rsqrt.default <- function(x) {
    return(1 / sqrt(x))
}

#' Element-wise inverse square root operation
#' @author Chloe Serre-Combe, Amelie Vernay
#' @description
#' Symbolic element-wise inverse square root operation for `LazyTensor` 
#' objects or standard element-wise inverse square root operation otherwise, 
#' i.e. \eqn{1/\sqrt{x}}.
#' @details If `x` is a `LazyTensor`, see [rkeops::rsqrt.LazyTensor()], else 
#' see [rkeops::rsqrt.default()].
#' @param x input for [rkeops::rsqrt.default()] or 
#' [rkeops::rsqrt.LazyTensor()].
#' @return See value of [rkeops::rsqrt.default()] or 
#' [rkeops::rsqrt.LazyTensor()].
#' @seealso [rkeops::rsqrt.default()], [rkeops::rsqrt.LazyTensor()]
#' @examples
#' \dontrun{
#' # Numerical input
#' rsqrt(4)
#' rsqrt(c(1,4,9,16))
#' # LazyTensor symbolic element-wise inverse square root
#' x <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows, 3 columns
#' x_i <- LazyTensor(x, index = 'i')   # creating LazyTensor from matrix x, 
#'                                     # indexed by 'i'
#' Rsqrt_x <- rsqrt(x_i)               # symbolic matrix, 150 rows and 3 columns
#' }
#' @export
rsqrt <- function(x) {
    UseMethod("rsqrt", x)
}

#' Element-wise inverse square root operation
#' @author Chloe Serre-Combe, Amelie Vernay
#' @description
#' Symbolic element-wise inverse square root operation for `LazyTensor` 
#' objects.
#' @details If `x` is a `LazyTensor`, `rsqrt(x)` returns a `LazyTensor` that 
#' encodes, symbolically, the element-wise inverse square root of `x`, i.e. 
#' \eqn{1/\sqrt{x}}.
#' @param x a `LazyTensor`, a `ComplexLazyTensor`, a vector of numeric values,
#' or a scalar value.
#' @return An object of class `LazyTensor`.
#' @seealso [rkeops::rsqrt()]
#' @examples
#' \dontrun{
#' x <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows, 3 columns
#' x_i <- LazyTensor(x, index = 'i')   # creating LazyTensor from matrix x, 
#'                                     # indexed by 'i'
#' Rsqrt_x <- rsqrt(x_i)               # symbolic matrix, 150 rows and 3 columns
#' }
#' @export
rsqrt.LazyTensor <- function(x) {
    res <- unaryop.LazyTensor(x, "Rsqrt")
    return(res)
}


# power ------------------------------------------------------------------------

#' @title Default arithmetic operations
#' @name default.arithmetic.fun
#' @aliases ^.default
#' @export
"^.default" <- function(x, y) {
    return(base::"^"(x, y))
}

#' Element-wise power operation
#' @name arithmetic.power
#' @aliases ^
#' @author Chloe Serre-Combe, Amelie Vernay
#' @description
#' Symbolic element-wise power binary operation for `LazyTensor` objects or 
#' default element-wise power operation otherwise, i.e. 
#' \eqn{x^y}.
#' @details
#' If `x` or `y` is a `LazyTensor`, see [rkeops::^.LazyTensor()], else see
#' [rkeops::^.default()].
#' @param x,y input for [rkeops::^.default()] or [rkeops::^.LazyTensor()].
#' @return See value of [rkeops::^.default()] or [rkeops::^.LazyTensor()].
#' @seealso [rkeops::^.default()], [rkeops::^.LazyTensor()]
#' @examples
#' \dontrun{
#' # R base operation
#' 3^2
#' (1:10)^3
#' 2^(1:10)
#' (1:10)^(1:2)
#' # LazyTensor symbolic power operation
#' x <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows, 3 columns
#' y <- matrix(runif(250 * 3), 250, 3) # arbitrary R matrix, 250 rows, 3 columns
#' x_i <- LazyTensor(x, index = 'i')   # creating LazyTensor from matrix x, 
#'                                     # indexed by 'i'
#' y_j <- LazyTensor(y, index = 'j')   # creating LazyTensor from matrix y, 
#'                                     # indexed by 'j'
#' x_pow_y <- x_i^y_j                  # symbolic matrix
#' }
#' @export
"^" <- function(x, y) { 
    if(!is.LazyTensor(x))
        UseMethod("^", y)
    else
        UseMethod("^", x)
}

#' Element-wise power operation
#' @name arithmetic.power.LazyTensor
#' @author Chloe Serre-Combe, Amelie Vernay
#' @aliases ^.lazyTensor
#' @description
#' Symbolic power binary operation for `LazyTensor` objects.
#' @details If `x` or `y` is a `LazyTensor`, `x^y` returns a `LazyTensor`
#' that encodes, symbolically, the element-wise value of `x` to the power 
#' of `y`.
#' (In case one of the arguments is a vector or a scalar, it is first converted 
#' to `LazyTensor`).
#' 
#' **Note**:
#' - if `y = 2`, `x^y` relies on the `"Square"` `KeOps` operation;
#' - if `y = 0.5`, `x^y` uses on the `"Sqrt"` `KeOps` operation;
#' - if `y = -0.5`, `x^y` uses on the `"Rsqrt"` `KeOps` operation.
#' @param x,y a `LazyTensor`, a `ComplexLazyTensor`, a vector of numeric values,
#' or a scalar value.
#' @return An object of class `LazyTensor`.
#' @examples
#' \dontrun{
#' x <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows, 3 columns
#' y <- matrix(runif(250 * 3), 250, 3) # arbitrary R matrix, 250 rows, 3 columns
#' x_i <- LazyTensor(x, index = 'i')   # creating LazyTensor from matrix x, 
#'                                     # indexed by 'i'
#' y_j <- LazyTensor(y, index = 'j')   # creating LazyTensor from matrix y, 
#'                                     # indexed by 'j'
#' x_pow_y <- x_i^y_j                  # symbolic matrix
#' }
#' @export
"^.LazyTensor" <- function(x, y) {   
    if(is.numeric(y) && length(y) == 1) {
        if(is.int(y)) {
            if(y == 2) {
                return(unaryop.LazyTensor(x, "Square"))
            } else {
                return(unaryop.LazyTensor(x, "Pow", y))
            }
        } else if(y == 0.5) {
            # element-wise square root
            return(unaryop.LazyTensor(x, "Sqrt"))
        } else if(y == (-0.5)) {
            # element-wise inverse square root
            return(unaryop.LazyTensor(x, "Rsqrt"))
        # check if Powf with y a float number has to be like Powf(var1,var2) 
        # or Powf(var,y) (Powf(var, 0.5))
        } else {
            return(binaryop.LazyTensor(x, y, "Powf")) # power operation
        }
    } else {
        return(binaryop.LazyTensor(x, y, "Powf")) # power operation
    }
}


# Euclidean scalar product -----------------------------------------------------

#' @title Logical.or
#' @name logical
#' @aliases |.default
#' @description
#' Logical "or" operator.
#' @details
#' See [base::Logic] for more details.
#' @inherit base::Logic params
#' @inherit base::Logic return
#' @seealso [base::Logic]
#' @author R core team and contributors
#' @examples
#' TRUE | FALSE
#' x <- 1:10
#' (x < 2) | (x > 8)
#' @export
"|.default" <- function(x, y) {
    return(base::"|"(x, y))
}

#' Euclidean scalar product (for LazyTensors) or default logical "or"
#' @name ScalarProduct.or.OR
#' @aliases |
#' @description
#' Euclidean scalar product symbolic binary operation for `LazyTensor` objects,
#' i.e. \eqn{\langle x, y\rangle}, or default element-wise logical "or" for 
#' other types, i.e. `x OR y`.
#' @details
#' If `x` or `y` is a `LazyTensor`, see [rkeops::|.LazyTensor()], else see
#' [rkeops::|.default()].
#' @param x,y input for [rkeops::|.default()] or [rkeops::|.LazyTensor()].
#' @return See value of [rkeops::|.default()] or [rkeops::|.LazyTensor()].
#' @seealso [rkeops::|.default()], [rkeops::|.LazyTensor()]
#' @examples
#' \dontrun{
#' # R base element-wise logical or operation
#' TRUE | FALSE
#' x <- 1:10
#' (x < 2) | (x > 8)
#' # LazyTensor symbolic scalar product operation
#' x <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows, 3 columns
#' y <- matrix(runif(250 * 3), 250, 3) # arbitrary R matrix, 250 rows, 3 columns
#' x_i <- LazyTensor(x, index = 'i')   # creating LazyTensor from matrix x, 
#'                                     # indexed by 'i'
#' y_j <- LazyTensor(y, index = 'j')   # creating LazyTensor from matrix y, 
#'                                     # indexed by 'j'
#' 
#' x_sp_y <- x_i | y_j                 # symbolic matrix
#' }
#' @export
"|" <- function(x, y) { 
    if(!is.LazyTensor(x))
        UseMethod("|", y)
    else
        UseMethod("|", x)
}

#' Euclidean scalar product operation
#' @name scalar.product.LazyTensor
#' @aliases |.LazyTensor
#' @author Chloe Serre-Combe, Amelie Vernay
#' @description
#' Symbolic Euclidean scalar product binary operation for `LazyTensor` objects.
#' @usage
#' \special{x | y}
#' \special{(x | y)}
#' @details If `x` or `y` is a `LazyTensor`, `x|y` returns a 
#' `LazyTensor` that encodes, symbolically, the Euclidean scalar product between 
#' `x` and `y`, which must have the same shape. (In case one of the arguments is 
#' a vector or a scalar, it is first converted to `LazyTensor`).
#'
#' **Note**: `x` and `y` input arguments should have the same inner dimension.
#' @param x,y a `LazyTensor`, a `ComplexLazyTensor`, a vector of numeric values,
#' or a scalar value.
#' @return An object of class `LazyTensor`.
#' @examples
#' \dontrun{
#' x <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows, 3 columns
#' y <- matrix(runif(250 * 3), 250, 3) # arbitrary R matrix, 250 rows, 3 columns
#' x_i <- LazyTensor(x, index = 'i')   # creating LazyTensor from matrix x, 
#'                                     # indexed by 'i'
#' y_j <- LazyTensor(y, index = 'j')   # creating LazyTensor from matrix y, 
#'                                     # indexed by 'j'
#' x_sp_y <- x_i | y_j                 # symbolic matrix
#' }
#' @export
"|.LazyTensor" <- function(x, y) {
    res <- binaryop.LazyTensor(x, y, "|", is_operator = TRUE,
                               dim_check_type = "same", dim_res = 1)
    res$formula <- paste("(", res$formula, ")", sep = "")
    return(res)
}


# Matrix product ---------------------------------------------------------------

#' @title Matrix multiplication
#' @name matmult.default
#' @aliases %*%.default
#' @inherit base::matmult description
#' @inherit base::matmult details
#' @inherit base::matmult params
#' @inherit base::matmult return
#' @inherit base::matmult examples
#' @seealso [base::matmult]
#' @author R core team and contributors
#' @export
"%*%.default" <- function(x, y) {
    return(base::"%*%"(x, y))
}

#' Matrix multiplication
#' @name matmult
#' @aliases %*%
#' @author Chloe Serre-Combe, Amelie Vernay
#' @description
#' Matrix multiplication binary operation for `LazyTensor` objects or 
#' default matrix multiplication operation for R matrices.
#' @details
#' If `x` or `y` is a `LazyTensor`, see [rkeops::%*%.LazyTensor()], else see
#' [rkeops::%*%.default()].
#' @param x,y input for [rkeops::%*%.default()] or [rkeops::%*%.LazyTensor()].
#' @return See value of [rkeops::%*%.default()] or [rkeops::%*%.LazyTensor()].
#' @seealso [rkeops::%*%.default()], [rkeops::%*%.LazyTensor()]
#' @examples
#' \dontrun{
#' # R base operation
#' x <- matrix(runif(10 * 5), 10, 5)   # arbitrary R matrix, 10 rows, 5 columns
#' x <- matrix(runif(5 * 3), 5, 3)     # arbitrary R matrix, 5 rows, 3 columns
#' x %*% y                             # product matrix, 10 rows, 3 columns
#' 
#' # LazyTensor matrix multiplication
#' x <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows, 3 columns
#' y <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows, 3 columns
#' x_i <- LazyTensor(x, index = 'i')   # creating LazyTensor from matrix x, 
#'                                     # indexed by 'i'
#' y_j <- LazyTensor(y, index = 'j')   # creating LazyTensor from matrix y, 
#'                                     # indexed by 'j'
#' x_mult_y <- x_i %*% y_j             # FIXME
#' }
#' @export
"%*%" <- function(x, y) { 
    if(!is.LazyTensor(x))
        UseMethod("%*%", y)
    else
        UseMethod("%*%", x)
}

#' Matrix multiplication
#' @name matmult.LazyTensor
#' @description
#' Matrix multiplication binary operation for `LazyTensor` objects 
#' corresponding to combination of multiplication operator `*` and sum 
#' reduction.
#' @details If `x` or `y` is a `LazyTensor`, `x %*% y` returns the sum 
#' reduction of the product `x * y`.
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param x,y a `LazyTensor`, a `ComplexLazyTensor`.
#' @return A matrix.
#' @examples
#' \dontrun{
#' x <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows, 3 columns
#' y <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows, 3 columns
#' x_i <- LazyTensor(x, index = 'i')   # creating LazyTensor from matrix x, 
#'                                     # indexed by 'i'
#' y_j <- LazyTensor(y, index = 'j')   # creating LazyTensor from matrix y, 
#'                                     # indexed by 'j'
#' x_mult_y <- x_i %*% y_j             # FIXME
#' }
#' @export
"%*%.LazyTensor" <- function(x, y) {
    # FIXME
    return(sum(x * y, index = "j")) # sum reduction 
}


# exponential ------------------------------------------------------------------

#' @title Logarithms and Exponentials
#' @name default.log.exp.fun
#' @aliases exp.default
#' @inherit base::log description
#' @inherit base::log details
#' @inherit base::log params
#' @inherit base::log return
#' @inherit base::log examples
#' @seealso [base::log]
#' @author R core team and contributors
#' @export
exp.default <- function(x) {
    return(base::exp(x))
}

#' Element-wise exponential operation
#' @author Chloe Serre-Combe, Amelie Vernay
#' @description
#' Symbolic element-wise exponential operation for `LazyTensor` objects or 
#' default element-wise exponential operation otherwise, i.e. 
#' \eqn{e^x}.
#' @details If `x` is a `LazyTensor`, see [rkeops::exp.LazyTensor()], else 
#' see [rkeops::exp.default()].
#' @param x input for [rkeops::exp.default()] or 
#' [rkeops::exp.LazyTensor()].
#' @return See value of [rkeops::exp.default()] or 
#' [rkeops::exp.LazyTensor()].
#' @seealso [rkeops::exp.default()], [rkeops::exp.LazyTensor()]
#' @examples
#' \dontrun{
#' # R base operation
#' exp(0)
#' exp(1:10)
#' # LazyTensor symbolic element-wise exponential
#' x <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows, 3 columns
#' x_i <- LazyTensor(x, index = 'i')   # creating LazyTensor from matrix x, 
#'                                     # indexed by 'i'
#' Exp_x <- exp(x_i)                   # symbolic matrix, 150 rows and 3 columns
#' }
#' @export
exp <- function(x) {
    UseMethod("exp")
}

#' Element-wise exponential operation
#' @name exp.LazyTensor
#' @author Chloe Serre-Combe, Amelie Vernay
#' @description
#' Symbolic element-wise exponential operation for `LazyTensor` objects.
#' @details 
#' **Different use cases**:
#' - If `x` is a `LazyTensor`, `exp(x)` returns a `LazyTensor` that 
#' encodes, symbolically, the element-wise exponential of `x`, i.e. 
#' \eqn{e^x}.
#' - If `x` is a `ComplexLazyTensor`, `exp(x)` returns a `LazyTensor` that 
#' encodes, symbolically, the element-wise complex exponential of `x`
#' @param x a `LazyTensor`, a `ComplexLazyTensor`, a vector of numeric values,
#' or a scalar value.
#' @return An object of class `LazyTensor` or `ComplexLazyTensor` depending on
#' input.
#' @seealso [rkeops::exp()]
#' @examples
#' \dontrun{
#' # basic example
#' x <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows, 3 columns
#' x_i <- LazyTensor(x, index = 'i')   # creating LazyTensor from matrix x, 
#'                                     # indexed by 'i'
#' Exp_x <- exp(x_i)                   # symbolic matrix, 150 rows and 3 columns
#' 
#' # basic example with complex exponential
#' z <- matrix(1i^ (-6:5), nrow = 4)        # create a complex 4x3 matrix
#' z_i <- LazyTensor(z, index = 'i', is_complex = TRUE) # create a ComplexLazyTensor
#' Exp_z_i <- exp(z_i)                                  # symbolic matrix
#' }
#' @export
exp.LazyTensor <- function(x) {
    return(unaryop.LazyTensor(x, "Exp"))
}

#' Element-wise exponential operation
#' @name exp.LazyTensor
#' @aliases exp.ComplexLazyTensor
#' @author Chloe Serre-Combe, Amelie Vernay
#' @export
exp.ComplexLazyTensor <- function(x) {
    return(unaryop.LazyTensor(x, "ComplexExp"))
}


# logarithm --------------------------------------------------------------------

#' @title Logarithms and Exponentials
#' @name default.log.exp.fun
#' @aliases log.default
#' @export
log.default <- function(x, base = exp(1)) {
    return(base::log(x, base))
}

#' Element-wise natural logarithm operation
#' @author Chloe Serre-Combe, Amelie Vernay
#' @description
#' Symbolic element-wise natural logarithm operation for `LazyTensor` 
#' objects or default element-wise natural logarithm operation otherwise, 
#' i.e. \eqn{\log(x)}.
#' @details If `x` is a `LazyTensor`, see [rkeops::log.LazyTensor()], else 
#' see [rkeops::log.default()].
#' @param x input for [rkeops::log.default()] or 
#' [rkeops::log.LazyTensor()].
#' @param base numeric, see [rkeops::log.default()], not used with `LazyTensor`
#' input.
#' @return See value of [rkeops::log.default()] or 
#' [rkeops::log.LazyTensor()].
#' @seealso [rkeops::log.default()], [rkeops::log.LazyTensor()]
#' @examples
#' \dontrun{
#' # R base operation
#' log(1)
#' log(1:10)
#' # LazyTensor symbolic element-wise natural logarithm
#' x <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows, 3 columns
#' x_i <- LazyTensor(x, index = 'i')   # creating LazyTensor from matrix x, 
#'                                     # indexed by 'i'
#' Log_x <- log(x_i)                   # symbolic matrix, 150 rows and 3 columns
#' }
#' @export
log <- function(x, base = exp(1)) {
    UseMethod("log")
}

#' Element-wise natural logarithm operation
#' @author Chloe Serre-Combe, Amelie Vernay
#' @usage
#' \special{log(x)}
#' @description
#' Symbolic element-wise natural logarithm operation for `LazyTensor` 
#' objects.
#' @details If `x` is a `LazyTensor`, `log(x)` returns a `LazyTensor` that 
#' encodes, symbolically, the element-wise natural logarithm of `x`, i.e. 
#' \eqn{\log{x}}.
#' @param x a `LazyTensor`, a `ComplexLazyTensor`, a vector of numeric values,
#' or a scalar value.
#' @param base not used, only present for method compatibility with
#' corresponding generic function.
#' @return An object of class `LazyTensor`.
#' @seealso [rkeops::log()]
#' @examples
#' \dontrun{
#' x <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows, 3 columns
#' x_i <- LazyTensor(x, index = 'i')   # creating LazyTensor from matrix x, 
#'                                     # indexed by 'i'
#' Log_x <- log(x_i)                   # symbolic matrix, 150 rows and 3 columns
#' }
#' @export
log.LazyTensor <- function(x, base = exp(1)) {
    return(unaryop.LazyTensor(x, "Log"))
}


# inverse ----------------------------------------------------------------------

#' Element-wise inverse operation
#' @author Chloe Serre-Combe, Amelie Vernay
#' @description
#' Element-wise inverse operations on numeric or complex vectors 
#' (or objects which can be coerced to them), i.e. `1/x`.
#' @details
#' `inv(x)` is equivalent to `1/x` where `/` is the default division operator.
#' 
#' From [base::Arithmetic]: if applied to arrays the result will be an array if 
#' this is sensible 
#' (for example it will not if the recycling rule has been invoked).
#' @param x numeric or complex vectors or objects which can be coerced to such, 
#' or other objects for which methods have been written.
#' @return Vector or array of inverted elements from `x`.
#' @seealso [base::Arithmetic]
#' @examples
#' inv(4)
#' inv(1:10)
#' @export
inv.default <- function(x) {
    return(1 / x)
}

#' Element-wise inverse operation.
#' @author Chloe Serre-Combe, Amelie Vernay
#' @description
#' Symbolic element-wise inverse operation for `LazyTensor` objects or 
#' standard element-wise inverse operation otherwise, i.e. `1/x`.
#' @details If `x` is a `LazyTensor`, see [rkeops::inv.LazyTensor()], else 
#' see [rkeops::inv.default()].
#' @param x input for [rkeops::inv.default()] or 
#' [rkeops::inv.LazyTensor()].
#' @return See value of [rkeops::inv.default()] or 
#' [rkeops::inv.LazyTensor()].
#' @seealso [rkeops::inv.default()], [rkeops::inv.LazyTensor()]
#' @examples
#' \dontrun{
#' # Numerical input
#' inv(4)
#' inv(1:10)
#' # LazyTensor symbolic element-wise inverse
#' x <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows, 3 columns
#' x_i <- LazyTensor(x, index = 'i')   # creating LazyTensor from matrix x, 
#'                                     # indexed by 'i'
#' Inv_x <- inv(x_i)                   # symbolic matrix, 150 rows and 3 columns
#' }
#' @export
inv <- function(x) {
    UseMethod("inv")
}

#' Element-wise inverse operation
#' @author Chloe Serre-Combe, Amelie Vernay
#' @description
#' Symbolic element-wise inverse operation for `LazyTensor` 
#' objects.
#' @details If `x` is a `LazyTensor`, `inv(x)` returns a `LazyTensor` that 
#' encodes, symbolically, the element-wise inverse of `x`, i.e. `1/x`.
#' @param x a `LazyTensor`, a `ComplexLazyTensor`, a vector of numeric values,
#' or a scalar value.
#' @return An object of class `LazyTensor`.
#' @seealso [rkeops::inv()]
#' @examples
#' \dontrun{
#' x <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows, 3 columns
#' x_i <- LazyTensor(x, index = 'i')   # creating LazyTensor from matrix x, 
#'                                     # indexed by 'i'
#' Inv_x <- inv(x_i)                   # symbolic matrix, 150 rows and 3 columns
#' }
#' @export
inv.LazyTensor <- function(x) {
    res <- unaryop.LazyTensor(x, "Inv")
    return(res)
}


# cosine -----------------------------------------------------------------------

#' @title Trigonometric Functions
#' @name default.trigo.fun
#' @aliases cos.default
#' @inherit base::cos description
#' @inherit base::cos details
#' @inherit base::cos params
#' @inherit base::cos return
#' @inherit base::cos examples
#' @seealso [base::Trig]
#' @author R core team and contributors
#' @export
cos.default <- function(x) {
    return(base::cos(x))
}

#' Element-wise cosine operation.
#' @author Chloe Serre-Combe, Amelie Vernay
#' @description
#' Symbolic element-wise cosine operation for `LazyTensor` 
#' objects or standard element-wise cosine operation otherwise.
#' @details If `x` is a `LazyTensor`, see [rkeops::cos.LazyTensor()], else 
#' see [rkeops::cos.default()].
#' @param x input for [rkeops::cos.default()] or 
#' [rkeops::cos.LazyTensor()].
#' @return See value of [rkeops::cos.default()] or 
#' [rkeops::cos.LazyTensor()].
#' @seealso [rkeops::cos.default()], [rkeops::cos.LazyTensor()]
#' @examples
#' \dontrun{
#' # R base operation
#' cos(0)
#' cos(pi)
#' # LazyTensor symbolic element-wise cosine
#' x <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows, 3 columns
#' x_i <- LazyTensor(x, index = 'i')   # creating LazyTensor from matrix x, 
#'                                     # indexed by 'i'
#' Cos_x <- cos(x_i)                   # symbolic matrix, 150 rows and 3 columns
#' }
#' @export
cos <- function(x) {
    UseMethod("cos")
}

#' Element-wise cosine operation
#' @author Chloe Serre-Combe, Amelie Vernay
#' @description
#' Symbolic element-wise cosine operation for `LazyTensor` objects.
#' @details If `x` is a `LazyTensor`, `cos(x)` returns a `LazyTensor` that 
#' encodes, symbolically, the element-wise cosine of `x`.
#' @param x a `LazyTensor`, a `ComplexLazyTensor`, a vector of numeric values,
#' or a scalar value.
#' @return An object of class `LazyTensor`.
#' @seealso [rkeops::cos()]
#' @examples
#' \dontrun{
#' x <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows, 3 columns
#' x_i <- LazyTensor(x, index = 'i')   # creating LazyTensor from matrix x, 
#'                                     # indexed by 'i'
#' Cos_x <- cos(x_i)                   # symbolic matrix, 150 rows and 3 columns
#' }
#' @export
cos.LazyTensor <- function(x) {
    return(unaryop.LazyTensor(x, "Cos"))
}


# sine -------------------------------------------------------------------------

#' @title Trigonometric Functions
#' @name default.trigo.fun
#' @aliases sin.default
#' @export
sin.default <- function(x) {
    return(base::sin(x))
}

#' Element-wise sine operation.
#' @author Chloe Serre-Combe, Amelie Vernay
#' @description
#' Symbolic element-wise sine operation for `LazyTensor` 
#' objects or standard element-wise sine operation otherwise.
#' @details If `x` is a `LazyTensor`, see [rkeops::sin.LazyTensor()], else 
#' see [rkeops::sin.default()].
#' @param x input for [rkeops::sin.default()] or 
#' [rkeops::sin.LazyTensor()].
#' @return See value of [rkeops::sin.default()] or 
#' [rkeops::sin.LazyTensor()].
#' @seealso [rkeops::sin.default()], [rkeops::sin.LazyTensor()]
#' @examples
#' \dontrun{
#' # R base operation
#' sin(0)
#' sin(pi)
#' # LazyTensor symbolic element-wise sine
#' x <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows, 3 columns
#' x_i <- LazyTensor(x, index = 'i')   # creating LazyTensor from matrix x, 
#'                                     # indexed by 'i'
#' Sin_x <- sin(x_i)                   # symbolic matrix, 150 rows and 3 columns
#' }
#' @export
sin <- function(x) {
    UseMethod("sin")
}

#' Element-wise sine operation
#' @author Chloe Serre-Combe, Amelie Vernay
#' @description
#' Symbolic element-wise sine operation for `LazyTensor` objects.
#' @details If `x` is a `LazyTensor`, `sin(x)` returns a `LazyTensor` that 
#' encodes, symbolically, the element-wise sine of `x`.
#' @param x a `LazyTensor`, a `ComplexLazyTensor`, a vector of numeric values,
#' or a scalar value.
#' @return An object of class `LazyTensor`.
#' @seealso [rkeops::sin()]
#' @examples
#' \dontrun{
#' x <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows, 3 columns
#' x_i <- LazyTensor(x, index = 'i')   # creating LazyTensor from matrix x, 
#'                                     # indexed by 'i'
#' Sin_x <- sin(x_i)                   # symbolic matrix, 150 rows and 3 columns
#' }
#' @export
sin.LazyTensor  <- function(x){
    return(unaryop.LazyTensor(x, "Sin"))
}


# arc-cosine --------------------------------------------------------------------

#' @title Trigonometric Functions
#' @name default.trigo.fun
#' @aliases acos.default
#' @export
acos.default <- function(x) {
    return(base::acos(x))
}

#' Element-wise arc-cosine operation.
#' @author Chloe Serre-Combe, Amelie Vernay
#' @description
#' Symbolic element-wise arc-cosine operation for `LazyTensor` 
#' objects or standard element-wise arc-cosine operation otherwise.
#' @details If `x` is a `LazyTensor`, see [rkeops::acos.LazyTensor()], else 
#' see [rkeops::acos.default()].
#' @param x input for [rkeops::acos.default()] or 
#' [rkeops::acos.LazyTensor()].
#' @return See value of [rkeops::acos.default()] or 
#' [rkeops::acos.LazyTensor()].
#' @seealso [rkeops::acos.default()], [rkeops::acos.LazyTensor()]
#' @examples
#' \dontrun{
#' # R base operation
#' acos(0)
#' acos(-1)
#' # LazyTensor symbolic element-wise arc-cosine
#' x <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows, 3 columns
#' x_i <- LazyTensor(x, index = 'i')   # creating LazyTensor from matrix x, 
#'                                     # indexed by 'i'
#' Acos_x <- acos(x_i)                 # symbolic matrix, 150 rows and 3 columns
#' }
#' @export
acos <- function(x) {
    UseMethod("acos")
}

#' Element-wise arc-cosine operation
#' @author Chloe Serre-Combe, Amelie Vernay
#' @description
#' Symbolic element-wise arc-cosine operation for `LazyTensor` objects.
#' @details If `x` is a `LazyTensor`, `acos(x)` returns a `LazyTensor` that 
#' encodes, symbolically, the element-wise arc-cosine of `x`.
#' @param x a `LazyTensor`, a `ComplexLazyTensor`, a vector of numeric values,
#' or a scalar value.
#' @return An object of class `LazyTensor`.
#' @seealso [rkeops::acos()]
#' @examples
#' \dontrun{
#' x <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows, 3 columns
#' x_i <- LazyTensor(x, index = 'i')   # creating LazyTensor from matrix x, 
#'                                     # indexed by 'i'
#' Acos_x <- acos(x_i)                 # symbolic matrix, 150 rows and 3 columns
#' }
#' @export
acos.LazyTensor <- function(x) {
    res <- unaryop.LazyTensor(x, "Acos")
    return(res)
}


# arc-sine ----------------------------------------------------------------------

#' @title Trigonometric Functions
#' @name default.trigo.fun
#' @aliases asin.default
#' @export
asin.default <- function(x) {
    return(base::asin(x))
}

#' Element-wise arc-sine operation.
#' @author Chloe Serre-Combe, Amelie Vernay
#' @description
#' Symbolic element-wise arc-sine operation for `LazyTensor` 
#' objects or standard element-wise arc-sine operation otherwise.
#' @details If `x` is a `LazyTensor`, see [rkeops::asin.LazyTensor()], else 
#' see [rkeops::asin.default()].
#' @param x input for [rkeops::asin.default()] or 
#' [rkeops::asin.LazyTensor()].
#' @return See value of [rkeops::asin.default()] or 
#' [rkeops::asin.LazyTensor()].
#' @seealso [rkeops::asin.default()], [rkeops::asin.LazyTensor()]
#' @examples
#' \dontrun{
#' # R base operation
#' asin(0)
#' asin(-1)
#' # LazyTensor symbolic element-wise arc-sine
#' x <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows, 3 columns
#' x_i <- LazyTensor(x, index = 'i')   # creating LazyTensor from matrix x, 
#'                                     # indexed by 'i'
#' Asin_x <- asin(x_i)                 # symbolic matrix, 150 rows and 3 columns
#' }
#' @export
asin <- function(x) {
    UseMethod("asin")
}

#' Element-wise arc-sine operation
#' @author Chloe Serre-Combe, Amelie Vernay
#' @description
#' Symbolic element-wise arc-sine operation for `LazyTensor` objects.
#' @details If `x` is a `LazyTensor`, `asin(x)` returns a `LazyTensor` that 
#' encodes, symbolically, the element-wise arc-sine of `x`.
#' @param x a `LazyTensor`, a `ComplexLazyTensor`, a vector of numeric values,
#' or a scalar value.
#' @return An object of class `LazyTensor`.
#' @seealso [rkeops::asin()]
#' @examples
#' \dontrun{
#' x <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows, 3 columns
#' x_i <- LazyTensor(x, index = 'i')   # creating LazyTensor from matrix x, 
#'                                     # indexed by 'i'
#' Asin_x <- asin(x_i)                 # symbolic matrix, 150 rows and 3 columns
#' }
#' @export
asin.LazyTensor <- function(x) {
    res <- unaryop.LazyTensor(x, "Asin")
    return(res)
}


# arc-tangent -------------------------------------------------------------------

#' @title Trigonometric Functions
#' @name default.trigo.fun
#' @aliases atan.default
#' @export
atan.default <- function(x) {
    return(base::atan(x))
}

#' Element-wise arc-tangent operation.
#' @author Chloe Serre-Combe, Amelie Vernay
#' @description
#' Symbolic element-wise arc-tangent operation for `LazyTensor` 
#' objects or standard element-wise arc-tangent operation otherwise.
#' @details If `x` is a `LazyTensor`, see [rkeops::atan.LazyTensor()], else 
#' see [rkeops::atan.default()].
#' @param x input for [rkeops::atan.default()] or 
#' [rkeops::atan.LazyTensor()].
#' @return See value of [rkeops::atan.default()] or 
#' [rkeops::atan.LazyTensor()].
#' @seealso [rkeops::atan.default()], [rkeops::atan.LazyTensor()]
#' @examples
#' \dontrun{
#' # R base operation
#' atan(0)
#' atan(-1)
#' # LazyTensor symbolic element-wise arc-tangent
#' x <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows, 3 columns
#' x_i <- LazyTensor(x, index = 'i')   # creating LazyTensor from matrix x, 
#'                                     # indexed by 'i'
#' Atan_x <- atan(x_i)                 # symbolic matrix, 150 rows and 3 columns
#' }
#' @export
atan <- function(x) {
    UseMethod("atan")
}

#' Element-wise arc-tangent operation
#' @author Chloe Serre-Combe, Amelie Vernay
#' @description
#' Symbolic element-wise arc-tangent operation for `LazyTensor` objects.
#' @details If `x` is a `LazyTensor`, `atan(x)` returns a `LazyTensor` that 
#' encodes, symbolically, the element-wise arc-tangent of `x`.
#' @param x a `LazyTensor`, a `ComplexLazyTensor`, a vector of numeric values,
#' or a scalar value.
#' @return An object of class `LazyTensor`.
#' @seealso [rkeops::atan()]
#' @examples
#' \dontrun{
#' x <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows, 3 columns
#' x_i <- LazyTensor(x, index = 'i')   # creating LazyTensor from matrix x, 
#'                                     # indexed by 'i'
#' Atan_x <- atan(x_i)                 # symbolic matrix, 150 rows and 3 columns
#' }
#' @export
atan.LazyTensor <- function(x) {
    res <- unaryop.LazyTensor(x, "Atan")
    return(res)
}


# arc-tan2 ---------------------------------------------------------------------

#' @title Trigonometric Functions
#' @name default.trigo.fun
#' @aliases atan2.default
#' @export
atan2.default <- function(x, y) {
    return(base::atan2(x, y))
}

#' Element-wise 2-argument arc-tangent operation.
#' @author Chloe Serre-Combe, Amelie Vernay
#' @description
#' Symbolic element-wise 2-argument arc-tangent operation for `LazyTensor` 
#' objects or standard element-wise 2-argument arc-tangent operation 
#' otherwise.
#' @details If `x` is a `LazyTensor`, see [rkeops::atan2.LazyTensor()], else 
#' see [rkeops::atan2.default()].
#' 
#' **Note**: the arc-tangent of two arguments `atan2(y, x)` returns the angle
#' between the x-axis and the vector from the origin to (x, y), i.e.,
#' for positive arguments `atan2(y, x) == atan(y/x)`.
#' @param x,y input for [rkeops::atan2.default()] or 
#' [rkeops::atan2.LazyTensor()].
#' @return See value of [rkeops::atan2.default()] or 
#' [rkeops::atan2.LazyTensor()].
#' @seealso [rkeops::atan2.default()], [rkeops::atan2.LazyTensor()]
#' @examples
#' \dontrun{
#' # R base operation
#' atan2(0, 1)
#' atan2(1, 0)
#' # LazyTensor symbolic element-wise 2-argument arc-tangent
#' x <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows, 3 columns
#' y <- matrix(runif(250 * 3), 250, 3) # arbitrary R matrix, 250 rows, 3 columns
#' x_i <- LazyTensor(x, index = 'i')   # creating LazyTensor from matrix x, 
#'                                     # indexed by 'i'
#' y_j <- LazyTensor(y, index = 'j')   # creating LazyTensor from matrix y, 
#'                                     # indexed by 'j'
#' Atan2_xy <- atan2(x_i, y_j)         # symbolic matrix
#' }
#' @export
atan2 <- function(x, y) {
    if(!is.LazyTensor(x) && !is.ComplexLazyTensor(x)) 
        UseMethod("atan2", y)
    else
        UseMethod("atan2", x)
}

#' Element-wise 2-argument arc-tangent operation
#' @author Chloe Serre-Combe, Amelie Vernay
#' @description
#' Symbolic element-wise 2-argument arc-tangent operation for 
#' `LazyTensor` objects.
#' @details If `x` or `y` is a `LazyTensor`, `atan2(x, y)` returns a 
#' `LazyTensor` that encodes, symbolically, the element-wise 2-argument 
#' arc-tangent of `x` and `y`. i.e. `atan2(x, y) == atan(x/y)`.
#' (In case one of the arguments is 
#' a vector or a scalar, it is first converted to `LazyTensor`).
#'
#' **Note**: `x` and `y` input arguments should have the same inner dimension.
#' @param x,y a `LazyTensor`, a `ComplexLazyTensor`, a vector of numeric values,
#' or a scalar value.
#' @return An object of class `LazyTensor`.
#' @seealso [rkeops::atan2()]
#' @examples
#' \dontrun{
#' x <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows, 3 columns
#' y <- matrix(runif(250 * 3), 250, 3) # arbitrary R matrix, 250 rows, 3 columns
#' x_i <- LazyTensor(x, index = 'i')   # creating LazyTensor from matrix x, 
#'                                     # indexed by 'i'
#' y_j <- LazyTensor(y, index = 'j')   # creating LazyTensor from matrix y, 
#'                                     # indexed by 'j'
#' Atan2_xy <- atan2(x_i, y_j)         # symbolic matrix
#' }
#' @export
atan2.LazyTensor <- function(x, y) {
    return(binaryop.LazyTensor(x, y, "Atan2", dim_check_type = "same"))
}


# absolute value ---------------------------------------------------------------

#' @title Miscellaneous Mathematical Functions
#' @name default.math.fun
#' @aliases abs.default
#' @seealso [base::abs()]
#' @export
abs.default <- function(x) {
    return(base::abs(x))
}

#' Element-wise absolute value (or modulus) operation
#' @author Chloe Serre-Combe, Amelie Vernay
#' @description
#' Symbolic element-wise absolute value (or modulus) operation for 
#' `LazyTensor` objects or default element-wise absolute value (or modulus)
#' operation otherwise, i.e. \eqn{\vert x \vert}.
#' @details If `x` is a `LazyTensor`, see [rkeops::abs.LazyTensor()], else 
#' see [rkeops::abs.default()].
#' 
#' **Note**: for complex input argument `z`, then the absolute value 
#' corresponds to the complex modulus `abs(z) = Mod(z)`.
#' @param x input for [rkeops::abs.default()] or [rkeops::abs.LazyTensor()].
#' @return See value of [rkeops::abs.default()] or [rkeops::abs.LazyTensor()].
#' @seealso [rkeops::abs.default()], [rkeops::abs.LazyTensor()], 
#' [rkeops::Mod.default()], [rkeops::Mod.LazyTensor()]
#' @examples
#' \dontrun{
#' # R base operation
#' abs(5)
#' abs(-5)
#' abs(-10:10)
#' # LazyTensor symbolic element-wise absolute value
#' x <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows, 3 columns
#' x_i <- LazyTensor(x, index = 'i')   # creating LazyTensor from matrix x, 
#'                                     # indexed by 'i'
#' Abs_x <- abs(x_i)                   # symbolic matrix, 150 rows and 3 columns
#' }
#' @export
abs <- function(x) {
    UseMethod("abs")
}

#' Element-wise absolute value operation
#' @name abs.LazyTensor
#' @author Chloe Serre-Combe, Amelie Vernay
#' @description
#' Symbolic element-wise absolute value (or modulus) operation for 
#' `LazyTensor` objects.
#' @details If `x` is a `LazyTensor`, `abs(x)` returns a `LazyTensor` that 
#' encodes, symbolically, the element-wise absolute value of `x`.
#' 
#' If `x` is a `ComplexLazyTensor`, `abs(x)` returns a `LazyTensor` that 
#' encodes, symbolically, the complex absolute value or modulus of `x`, i.e.
#' `abs(x) = Mod(x)`.
#' @param x a `LazyTensor`, a `ComplexLazyTensor`, a vector of numeric values,
#' or a scalar value.
#' @return An object of class `LazyTensor`.
#' @seealso [rkeops::abs()], [rkeops::Mod()]
#' @examples
#' \dontrun{
#' x <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows, 3 columns
#' x_i <- LazyTensor(x, index = 'i')   # creating LazyTensor from matrix x, 
#'                                     # indexed by 'i'
#'                                     
#' Abs_x <- abs(x_i)                   # symbolic matrix, 150 rows and 3 columns
#' }
#' @export
abs.LazyTensor <- function(x) {
    return(unaryop.LazyTensor(x, "Abs"))
}

#' Element-wise absolute value operation
#' @name abs.LazyTensor
#' @aliases abs.ComplexLazyTensor
#' @author Chloe Serre-Combe, Amelie Vernay
#' @export
abs.ComplexLazyTensor <- function(x) {
    return(unaryop.LazyTensor(x, "ComplexAbs", res_type = "LazyTensor"))
}


# sign function ----------------------------------------------------------------

#' Sign function
#' @inherit base::sign description
#' @inherit base::sign details
#' @inherit base::sign params
#' @inherit base::sign return
#' @inherit base::sign examples
#' @seealso [base::sign()]
#' @author R core team and contributors
#' @export
sign.default <- function(x) {
    return(base::sign(x))
}

#' Element-wise sign operation
#' @author Chloe Serre-Combe, Amelie Vernay
#' @description
#' Symbolic element-wise sign operation for `LazyTensor` objects or 
#' default element-wise sign operation otherwise.
#' @details If `x` is a `LazyTensor`, see [rkeops::sign.LazyTensor()], else 
#' see [rkeops::sign.default()].
#' 
#' The sign function is defined as follows:
#' \deqn{\text{sign}(x) = \left\{
#' \begin{array}{rcl}
#' 0 & \text{if } & x = 0\\
#' -1 & \text{if } & x < 0\\
#' 1 & \text{if } & x > 0 
#' \end{array}\right.}
#' @param x input for [rkeops::sign.default()] or [rkeops::sign.LazyTensor()].
#' @return See value of [rkeops::sign.default()] or [rkeops::sign.LazyTensor()].
#' @seealso [rkeops::sign.default()], [rkeops::sign.LazyTensor()]
#' @examples
#' \dontrun{
#' # R base operation
#' sign(5)
#' sign(-5)
#' sign(-10:10)
#' # LazyTensor symbolic element-wise sign
#' x <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows, 3 columns
#' x_i <- LazyTensor(x, index = 'i')   # creating LazyTensor from matrix x, 
#'                                     # indexed by 'i'
#' Sign_x <- sign(x_i)                 # symbolic matrix, 150 rows and 3 columns
#' }
#' @export
sign <- function(x) {
    UseMethod("sign")
}

#' Element-wise sign operation
#' @author Chloe Serre-Combe, Amelie Vernay
#' @description
#' Symbolic element-wise sign operation for `LazyTensor` objects.
#' @details If `x` is a `LazyTensor`, `sign(x)` returns a `LazyTensor` that 
#' encodes, symbolically, the element-wise sign of `x`, i.e. 
#' \deqn{\text{sign}(x) = \left\{
#' \begin{array}{rcl}
#' 0 & \text{if } & x = 0\\
#' -1 & \text{if } & x < 0\\
#' 1 & \text{if } & x > 0 
#' \end{array}\right.}
#' @param x a `LazyTensor`, a `ComplexLazyTensor`, a vector of numeric values,
#' or a scalar value.
#' @return An object of class `LazyTensor`.
#' @seealso [rkeops::sign()]
#' @examples
#' \dontrun{
#' x <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows, 3 columns
#' x_i <- LazyTensor(x, index = 'i')   # creating LazyTensor from matrix x, 
#'                                     # indexed by 'i'
#' Sign_x <- sign(x_i)                 # symbolic matrix, 150 rows and 3 columns
#' }
#' @export
sign.LazyTensor <- function(x) {
    return(unaryop.LazyTensor(x, "Sign"))
}


# round function ---------------------------------------------------------------

#' Rounding function
#' @inherit base::round description
#' @inherit base::round details
#' @inherit base::round params
#' @inherit base::round return
#' @inherit base::round examples
#' @seealso [base::round()]
#' @author R core team and contributors
#' @export
round.default <- function(x, digits = 0) {
    return(base::round(x, digits))
}

#' Element-wise rounding function
#' @author Chloe Serre-Combe, Amelie Vernay
#' @description
#' Symbolic element-wise rounding function for `LazyTensor` objects or 
#' default element-wise rounding function otherwise.
#' @details If `x` is a `LazyTensor`, see [rkeops::round.LazyTensor()], else 
#' see [rkeops::round.default()].
#' @param x input for [rkeops::round.default()] or [rkeops::round.LazyTensor()].
#' @param digits integer indicating the number of decimal places to be used 
#' when rounding.
#' @return See value of [rkeops::round.default()] or [rkeops::round.LazyTensor()].
#' @seealso [rkeops::round.default()], [rkeops::round.LazyTensor()]
#' @examples
#' \dontrun{
#' # R base operation
#' round(5)
#' round(runif(10), 4)
#' # LazyTensor symbolic element-wise rounding
#' x <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows, 3 columns
#' x_i <- LazyTensor(x, index = 'i')   # creating LazyTensor from matrix x, 
#'                                     # indexed by 'i'
#' Round_x <- round(x_i, 2)            # symbolic matrix
#' }
#' @export
round <- function(x, digits = 0) {
    UseMethod("round", x)
}

#' Element-wise rounding function
#' @author Chloe Serre-Combe, Amelie Vernay
#' @description
#' Symbolic element-wise rounding function for `LazyTensor` objects.
#' @details If `x` is a `LazyTensor`, `round(x, digits)` returns a `LazyTensor` 
#' that encodes, symbolically, the element-wise rounding of `x` to `digits` 
#' decimal places.
#' @param x a `LazyTensor`, a `ComplexLazyTensor`, a vector of numeric values,
#' or a scalar value.
#' @param digits integer indicating the number of decimal places to be used 
#' when rounding.
#' @return An object of class `LazyTensor`.
#' @seealso [rkeops::round()]
#' @examples
#' \dontrun{
#' x <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows, 3 columns
#' x_i <- LazyTensor(x, index = 'i')   # creating LazyTensor from matrix x, 
#'                                     # indexed by 'i'
#' Round_x <- round(x_i, 2)            # symbolic matrix
#' }
#' @export
round.LazyTensor <- function(x, digits = 0) {
    if(is.numeric(digits) && length(digits) == 1)
        res <- unaryop.LazyTensor(x, "Round", digits)
    else
        stop("`digits` input argument should be a scalar.")
    return(res)
}


# xlogx function ---------------------------------------------------------------

#' x*log(x) function
#' @author Chloe Serre-Combe, Amelie Vernay
#' @description
#' Element-wise \eqn{x \times \log(x)} function on numeric or complex vectors 
#' (or objects which can be coerced to them).
#' @details
#' See [base::Arithmetic] and [base::log()] for details about the 
#' multiplication operation and log function.
#' 
#' **Note**: by convention `xlogx(0)` returns `0`.
#' @param x numeric or complex vectors or objects which can be coerced to such.
#' @return Vector or array containing the values of `x*log(x)`.
#' @seealso [base::Arithmetic], [base::log()]
#' @examples
#' xlogx(4)
#' xlogx(1:10)
#' @export
xlogx.default <- function(x) {
    return(ifelse(x == 0, 0, x * log(x)))
}

#' Element-wise `x*log(x)` operation
#' @author Chloe Serre-Combe, Amelie Vernay
#' @description
#' Symbolic element-wise `x*log(x)` function for `LazyTensor` objects or
#' standard element-wise `x*log(x)` function otherwise.
#' @details If `x` is a `LazyTensor`, see [rkeops::xlogx.LazyTensor()], 
#' else see [rkeops::xlogx.default()].
#' 
#' **Note**: by convention `xlogx(0)` returns `0`.
#' @param x input for [rkeops::xlogx.default()] or 
#' [rkeops::xlogx.LazyTensor()].
#' @return See value of [rkeops::xlogx.default()] or 
#' [rkeops::xlogx.LazyTensor()]
#' @seealso [rkeops::xlogx.default()], [rkeops::xlogx.LazyTensor()]
#' @examples
#' \dontrun{
#' # Numerical input
#' xlog(4)
#' xlog(1:10)
#' # LazyTensor symbolic element-wise `x*log(x)`
#' x <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows, 3 columns
#' x_i <- LazyTensor(x, index = 'i')   # creating LazyTensor from matrix x, 
#'                                     # indexed by 'i'
#' xlog_x <- xlogx(x_i)                # symbolic matrix, 150 rows and 3 columns
#' }
#' @export
xlogx <- function(x) {
    UseMethod("xlogx", x)
}

#' Element-wise `x*log(x)` operation
#' @author Chloe Serre-Combe, Amelie Vernay
#' @description
#' Symbolic element-wise `x*log(x)` function for `LazyTensor` objects.
#' @details If `x` is a `LazyTensor`, `square(x)` returns a `LazyTensor` that 
#' encodes, symbolically, the element-wise `x*log(x)` values for `x`.
#' 
#' **Note**: by convention `xlogx(0)` returns `0`.
#' @param x a `LazyTensor`, a `ComplexLazyTensor`, a vector of numeric values,
#' or a scalar value.
#' @return An object of class `LazyTensor`.
#' @seealso [rkeops::xlogx()]
#' @examples
#' \dontrun{
#' x <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows, 3 columns
#' x_i <- LazyTensor(x, index = 'i')   # creating LazyTensor from matrix x, 
#'                                     # indexed by 'i'
#' xlog_x <- xlogx(x_i)                # symbolic matrix, 150 rows and 3 columnsjs
#' }
#' @export
xlogx.LazyTensor <- function(x) {
    return(unaryop.LazyTensor(x, "XLogX"))
}


# sinxdivx function ------------------------------------------------------------

#' sin(x)/x function
#' @author Chloe Serre-Combe, Amelie Vernay
#' @description
#' Element-wise \eqn{\sin(x) / x} function on numeric or complex vectors 
#' (or objects which can be coerced to them).
#' @details
#' See [base::sin()], [base::Arithmetic] for details about the sine function
#' and division operation.
#' 
#' **Note**: by convention `sinxdivx(0)` returns `1`.
#' @param x numeric or complex vectors or objects which can be coerced to such.
#' @return Vector or array containing the values of `sin(x)/x`.
#' @seealso [base::Arithmetic], [base::sin()]
#' @examples
#' sinxdivx(pi)
#' sinxdivx(1:10)
#' @export
sinxdivx.default <- function(x) {
    return(ifelse(x == 0, 1, sin(x) / x))
}

#' Element-wise `sin(x)/x` function
#' @author Chloe Serre-Combe, Amelie Vernay
#' @description
#' Symbolic element-wise `sin(x)/x` function for `LazyTensor` objects or
#' standard element-wise `sin(x)/x` function otherwise.
#' @details If `x` is a `LazyTensor`, see [rkeops::sinxdivx.LazyTensor()], 
#' else see [rkeops::sinxdivx.default()].
#' 
#' **Note**: by convention `sinxdivx(0)` returns `1`.
#' @param x input for [rkeops::sinxdivx.default()] or 
#' [rkeops::sinxdivx.LazyTensor()].
#' @return See value of [rkeops::sinxdivx.default()] or 
#' [rkeops::sinxdivx.LazyTensor()]
#' @seealso [rkeops::sinxdivx.default()], [rkeops::sinxdivx.LazyTensor()]
#' @examples
#' \dontrun{
#' # Numerical input
#' sinxdivx(4)
#' sinxdivx(1:10)
#' # LazyTensor symbolic element-wise `sin(x)/x`
#' x <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows, 3 columns
#' x_i <- LazyTensor(x, index = 'i')   # creating LazyTensor from matrix x, 
#'                                     # indexed by 'i'
#' sindiv_x <- sinxdivx(x_i)           # symbolic matrix, 150 rows and 3 columns
#' }
#' @export
sinxdivx <- function(x) {
    UseMethod("sinxdivx", x)
}

#' Element-wise `sin(x)/x` operation
#' @author Chloe Serre-Combe, Amelie Vernay
#' @description
#' Symbolic element-wise `sin(x)/x` function for `LazyTensor` objects.
#' @details If `x` is a `LazyTensor`, `square(x)` returns a `LazyTensor` that 
#' encodes, symbolically, the element-wise `sin(x)/x` values for `x`.
#' 
#' **Note**: by convention `sinxdivx(0)` returns `1`.
#' @param x a `LazyTensor`, a `ComplexLazyTensor`, a vector of numeric values,
#' or a scalar value.
#' @return An object of class `LazyTensor`.
#' @seealso [rkeops::sinxdivx()]
#' @examples
#' \dontrun{
#' x <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows, 3 columns
#' x_i <- LazyTensor(x, index = 'i')   # creating LazyTensor from matrix x, 
#'                                     # indexed by 'i'
#' sindiv_x <- sinxdivx(x_i)           # symbolic matrix, 150 rows and 3 columns
#' }
#' @export
sinxdivx.LazyTensor <- function(x) {
    return(unaryop.LazyTensor(x, "SinXDivX"))
}


# step function ----------------------------------------------------------------

#' Choose a model by AIC in a Stepwise Algorithm
#' @inherit stats::step description
#' @inherit stats::step details
#' @inherit stats::step params
#' @inherit stats::step return
#' @inherit stats::step examples
#' @seealso [stats::step()]
#' @author R core team and contributors
#' @export
step.default <- function(object, ...) {
    return(stats::step(object, ...))
}

#' Element-wise 0-1 step function (for LazyTensors) or default stepwise model 
#' selection
#' @author Chloe Serre-Combe, Amelie Vernay
#' @description
#' Symbolic element-wise 0-1 step function for `LazyTensor` objects 
#' (i.e. `0` if `x < 0`, `1` if `x >= 0` for an input `x`) or stepwise 
#' model selection otherwise.
#' @details If `x` is a `LazyTensor`, see [rkeops::step.LazyTensor()], else 
#' see [rkeops::step.default()].
#' @param object input for [rkeops::step.default()] or 
#' [rkeops::step.LazyTensor()].
#' @param ... optional additional input arguments.
#' @return See value of [rkeops::step.default()] or [rkeops::step.LazyTensor()].
#' @seealso [rkeops::step.default()], [rkeops::step.LazyTensor()]
#' @examples
#' \dontrun{
#' # R base operation: see `?step.default`
#' # LazyTensor symbolic element-wise sign
#' x <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows, 3 columns
#' x_i <- LazyTensor(x, index = 'i')   # creating LazyTensor from matrix x, 
#'                                     # indexed by 'i'
#' Step_x <- step.LazyTensor(x_i)      # symbolic matrix, 150 rows and 3 columns
#' }
#' @export
step <- function(object, ...){
    UseMethod("step", object)
}

#' Element-wise 0-1 step function
#' @author Chloe Serre-Combe, Amelie Vernay
#' @description
#' Symbolic element-wise 0-1 step function for `LazyTensor` objects.
#' @details If `x` is a `LazyTensor`, `step(x)` returns a `LazyTensor` 
#' that encodes, symbolically, the element-wise 0-1 step of `x`, i.e. 
#' `0` if `x < 0`, `1` if `x >= 0`.
#' @param object a `LazyTensor`, a `ComplexLazyTensor`, a vector of numeric 
#' values, or a scalar value.
#' @param ... not used, only present for method compatibility with the
#' corresponding generic function.
#' @return An object of class `LazyTensor`.
#' @seealso [rkeops::step()]
#' @examples
#' \dontrun{
#' x <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows, 3 columns
#' x_i <- LazyTensor(x, index = 'i')   # creating LazyTensor from matrix x, 
#'                                     # indexed by 'i'
#' Step_x <- step.LazyTensor(x_i)      # symbolic matrix, 150 rows and 3 columns
#' }
#' @export
step.LazyTensor <- function(object, ...) {
    return(unaryop.LazyTensor(object, "Step"))
}


# relu function ----------------------------------------------------------------

#' Element-wise ReLU function
#' @author Chloe Serre-Combe, Amelie Vernay
#' @description
#' Element-wise ReLU (Rectified Linear Unit) function on numeric vectors 
#' (or objects which can be coerced to them).
#' @details
#' The ReLU function \insertCite{fukushima_cognitron_1975}{rkeops} is defined 
#' as follows: `relu(x)` returns `0` if `x < 0`, `x` if `x >= 0`.
#' @param x numeric vectors or objects which can be coerced to such.
#' @return Vector or array containing the element-wise ReLU of `x`.
#' @references
#' \insertRef{fukushima_cognitron_1975}{rkeops}
#' @examples
#' relu(4)
#' relu(-10:10)
#' @export
relu.default <- function(x) {
    return(ifelse(x < 0, 0, x))
}

#' Element-wise ReLU function
#' @author Chloe Serre-Combe, Amelie Vernay
#' @description
#' Symbolic element-wise ReLU function for `LazyTensor` objects or
#' standard element-wise ReLU function otherwise.
#' @details The ReLU function \insertCite{fukushima_cognitron_1975}{rkeops} 
#' is defined as follows: `relu(x)` returns `0` if `x < 0`, `x` if `x >= 0`.
#' 
#' If `x` is a `LazyTensor`, see [rkeops::relu.LazyTensor()], else 
#' see [rkeops::relu.default()].
#' @param x input for [rkeops::relu.default()] or 
#' [rkeops::relu.LazyTensor()].
#' @return See value of [rkeops::relu.default()] or 
#' [rkeops::relu.LazyTensor()]
#' @seealso [rkeops::relu.default()], [rkeops::relu.LazyTensor()]
#' @references
#' \insertRef{fukushima_cognitron_1975}{rkeops}
#' @examples
#' \dontrun{
#' # Numerical input
#' relu(4)
#' relu(-10:10)
#' # LazyTensor symbolic element-wise square
#' x <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows, 3 columns
#' x_i <- LazyTensor(x, index = 'i')   # creating LazyTensor from matrix x, 
#'                                     # indexed by 'i'
#' ReLU_x <- relu(x_i)                 # symbolic matrix, 150 rows and 3 columns
#' }
#' @export
relu <- function(x) {
    UseMethod("relu", x)
}

#' Element-wise ReLU function
#' @author Chloe Serre-Combe, Amelie Vernay
#' @description
#' Symbolic element-wise ReLU (Rectified Linear Unit) function for 
#' `LazyTensor` objects.
#' @details If `x` is a `LazyTensor`, `relu(x)` returns a `LazyTensor` that 
#' encodes, symbolically, the element-wise ReLU of `x`.
#' 
#' The ReLU function \insertCite{fukushima_cognitron_1975}{rkeops} is defined 
#' as follows: `relu(x)` returns `0` if `x < 0`, `x` if `x >= 0`.
#' @param x a `LazyTensor`, a `ComplexLazyTensor`, a vector of numeric values,
#' or a scalar value.
#' @return An object of class `LazyTensor`.
#' @seealso [rkeops::relu()]
#' @references
#' \insertRef{fukushima_cognitron_1975}{rkeops}
#' @examples
#' \dontrun{
#' x <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows, 3 columns
#' x_i <- LazyTensor(x, index = 'i')   # creating LazyTensor from matrix x, 
#'                                     # indexed by 'i'
#' ReLU_x <- relu(x_i)                 # symbolic matrix, 150 rows and 3 columns
#' }
#' @export
relu.LazyTensor <- function(x) {
    return(unaryop.LazyTensor(x, "ReLU"))
}


# clamp function ---------------------------------------------------------------

#' Element-wise clamp function
#' @description
#' Symbolic element-wise clamp function (ternary operation) for `LazyTensor` 
#' objects.
#' @details `clamp(x, a, b)` returns a `LazyTensor` that encodes, symbolically,
#' the element-wise clamping of  `x` in `(a, b)`, i.e. `a` if `x < a`, 
#' `x` if `a <= x <= b`, and `b` if  `b < x`.
#' Broadcasting rules apply.
#' 
#' **Note**: If `a` and `b` are not scalar values, these should have the same 
#' inner dimension as `x`.
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param x,a,b A `LazyTensor`, a vector of numeric values, or a scalar value.
#' @return An object of class `LazyTensor`.
#' @examples
#' \dontrun{
#' # basic example
#' D <- 3
#' M <- 100
#' N <- 150
#' P <- 200
#' x <- matrix(runif(M * D), M, D)
#' y <- matrix(runif(N * D), N, D)
#' z <- matrix(runif(P * D), P, D)
#' x_i <- LazyTensor(x, index = 'i')
#' y_j <- LazyTensor(y, index = 'j')
#' z_i <- LazyTensor(z, index = 'i')
#' 
#' # call clamp function
#' clp <- clamp(x_i, y_j, z_i)
#' }
#' @export
clamp <- function(x, a, b) {
    if((is.ComplexLazyTensor(x) || is.ComplexLazyTensor(a)) 
       || (is.ComplexLazyTensor(b))) {
        stop(paste0("`x`, `a` and `b` input arguments ", 
                    "cannot be ComplexLazyTensors."))
    }
    
    if(is.int(a) && is.int(b))
        res <- unaryop.LazyTensor(x, "ClampInt", a, b)
    else
        res <- ternaryop.LazyTensor(x, a, b, "Clamp")
    return(res)
}


# clampint function ------------------------------------------------------------

#' Element-wise clampint function
#' @description
#' Symbolic element-wise clampint function (ternary operation) for `LazyTensor` 
#' objects.
#' @details `clampint(x, y, z)` returns a `LazyTensor` that encodes, 
#' symbolically, the element-wise clamping of `x` in `(y, z)` which are 
#' integers. See [rkeops::clamp()] for more details.
#' Broadcasting rules apply.
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param x A `LazyTensor`, a vector of numeric values, or a scalar value.
#' @param y An `integer`.
#' @param z An `integer`.
#' @return An object of class `LazyTensor`.
#' @examples
#' \dontrun{
#' # basic example
#' D <- 3
#' M <- 100
#' x <- matrix(runif(M * D), M, D)
#' x_i <- LazyTensor(x, index = 'i')
#' 
#' # call clampint function
#' clp_int <- clamp(x_i, 7, 2)
#' }
#' @export
clampint <- function(x, y, z) {
    if(is.ComplexLazyTensor(x)) {
        stop(paste0("`x` cannot be a ComplexLazyTensor."))
    }
    
    if((!is.int(y) || !is.int(z))) {
        stop(
            paste(
                "`clampint(x, y, z)` expects integer arguments for `y` and `z`.", 
                "Use clamp(x, y, z) for different `y` and `z` types.", 
                sep = " "
            )
        )
    }
    res <- unaryop.LazyTensor(x, "ClampInt", y, z)
}


# if-else function -------------------------------------------------------------

#' Conditional Element Selection
#' @inherit base::ifelse description
#' @inherit base::ifelse details
#' @inherit base::ifelse params
#' @inherit base::ifelse return
#' @inherit base::ifelse examples
#' @seealso [base::ifelse()]
#' @author R core team and contributors
#' @export
ifelse.default <- function(test, yes, no) {
    return(base::ifelse(test, yes, no))
}


#' Element-wise if-else function
#' @author Chloe Serre-Combe, Amelie Vernay
#' @description
#' Symbolic element-wise if-else function for `LazyTensor` objects or
#' standard element-wise if-else function otherwise.
#' @details If `test` is a `LazyTensor`, see [rkeops::ifelse.LazyTensor()], else 
#' see [rkeops::ifelse.default()].
#' @param test,yes,no input for [rkeops::ifelse.default()] or 
#' [rkeops::ifelse.LazyTensor()].
#' @return See value of [rkeops::ifelse.default()] or 
#' [rkeops::ifelse.LazyTensor()]
#' @seealso [rkeops::ifelse.default()], [rkeops::ifelse.LazyTensor()]
#' @examples
#' \dontrun{
#' # R base operation
#' x <- c(6:-4)
#' sqrt(ifelse(x >= 0, x, NA))
#' # LazyTensor symbolic element-wise square
#' D <- 3
#' M <- 100
#' N <- 150
#' P <- 200
#' x <- matrix(runif(M * D), M, D)
#' y <- matrix(runif(N * D), N, D)
#' z <- matrix(runif(P * D), P, D)
#' x_i <- LazyTensor(x, index = 'i')
#' y_j <- LazyTensor(y, index = 'j')
#' z_i <- LazyTensor(z, index = 'i')
#' 
#' if_else_xyz <- ifelse(x_i, y_j, z_i)
#' }
#' @export
ifelse <- function(test, yes, no) {
    UseMethod("ifelse", test)
}

#' Element-wise if-else function.
#' @author Chloe Serre-Combe, Amelie Vernay
#' @description
#' Symbolic element-wise if-else function  (ternary operation) for 
#' `LazyTensor` objects.
#' @details If `test` is a `LazyTensor`, `ifelse(test, yes, no)` returns a
#'  `LazyTensor` that encodes, symbolically, `yes` if `test >= 0` and 
#'  `no` if `test < 0`.  
#' Broadcasting rules apply. 
#' `yes` and `no` may be fixed integers or floats, or other `LazyTensor`.
#' 
#' **Note**: If `yes` and `no` are not scalar values, these should have the same 
#' inner dimension as `test`.
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param test,yes,no a `LazyTensor`, a vector of numeric values, 
#' or a scalar value.
#' @return An object of class `LazyTensor`.
#' @examples
#' \dontrun{
#' # basic example
#' D <- 3
#' M <- 100
#' N <- 150
#' P <- 200
#' x <- matrix(runif(M * D), M, D)
#' y <- matrix(runif(N * D), N, D)
#' z <- matrix(runif(P * D), P, D)
#' x_i <- LazyTensor(x, index = 'i')
#' y_j <- LazyTensor(y, index = 'j')
#' z_i <- LazyTensor(z, index = 'i')
#' 
#' # call ifelse function
#' if_else_xyz <- ifelse(x_i, y_j, z_i)
#' }
#' @export
ifelse.LazyTensor <- function(test, yes, no) {
    if((is.ComplexLazyTensor(test) || is.ComplexLazyTensor(yes)) 
       || (is.ComplexLazyTensor(yes))) {
        stop(paste0("`test`, `yes` and `no` input arguments ", 
                    "cannot be ComplexLazyTensors."))
    }
    return(ternaryop.LazyTensor(test, yes, no, "IfElse"))
}


# mod function -----------------------------------------------------------------

#' Element-wise modulo with offset function
#' @rdname modulo
#' @author Chloe Serre-Combe, Amelie Vernay
#' @description
#' Symbolic element-wise modulo with offset function  (ternary operation) for 
#' `LazyTensor` objects  or any other `mod` function (if existing).
#' @details If `x` is a `LazyTensor`, see [rkeops::ifelse.LazyTensor()], else 
#' see the corresponding `mod` function (if existing) help page.
#' 
#' **Warning**: Do not confuse with [rkeops::Mod()].
#' @param x input for [rkeops::ifelse.default()] or any other `mod` function 
#' (if existing).
#' @param ... optional additional input arguments.
#' @return See value of [rkeops::mod.LazyTensor()] or any other `mod` function
#' (if existing).
#' @seealso [rkeops::mod.LazyTensor()]
#' @examples
#' \dontrun{
#' # basic example
#' D <- 3
#' M <- 100
#' N <- 150
#' P <- 200
#' x <- matrix(runif(M * D), M, D)
#' y <- matrix(runif(N * D), N, D)
#' z <- matrix(runif(P * D), P, D)
#' x_i <- LazyTensor(x, index = 'i')
#' y_j <- LazyTensor(y, index = 'j')
#' z_i <- LazyTensor(z, index = 'i')
#' 
#' # call mod function
#' mod_x72 <- mod(x_i, 7, 2)
#' 
#' # works also with LazyTensors with same inner dimension or dimension 1
#' mod_xyz <- mod(x_i, y_j, z_i)
#' }
#' @export
mod <- function(x, ...) {
    UseMethod("mod", x)
}

#' Element-wise modulo with offset function
#' @rdname modulo.LazyTensor
#' @author Chloe Serre-Combe, Amelie Vernay
#' @description
#' Symbolic element-wise modulo with offset function (ternary operation) for 
#' `LazyTensor` objects
#' @details If `x` is a `LazyTensor`, `mod(x, a, b)` returns a `LazyTensor` 
#' that encodes, symbolically, the element-wise modulo of `x` with 
#' divisor `a` and offset `b`, i.e. `x - a * floor((x - b)/a)`.
#' By default `b = 0`, `mod(x, a)` corresponds to the standard R 
#' function `%%`.
#' 
#' `a` and `b` may be fixed integers or floats, or other `LazyTensor`.
#' Broadcasting rules apply.
#' 
#' **Note**: If `a` and `b` are not scalar values, these should have the same 
#' inner dimension as `x`.
#' 
#' **Warning**: Do not confuse with [rkeops::Mod()].
#' @param x,a,b a `LazyTensor`, a `ComplexLazyTensor`, a vector of numeric 
#' values, or a scalar value.
#' @param ... not used, only present for method compatibility with
#' corresponding generic function.
#' @return An object of class `LazyTensor`.
#' @examples
#' \dontrun{
#' # basic example
#' D <- 3
#' M <- 100
#' N <- 150
#' P <- 200
#' x <- matrix(runif(M * D), M, D)
#' y <- matrix(runif(N * D), N, D)
#' z <- matrix(runif(P * D), P, D)
#' x_i <- LazyTensor(x, index = 'i')
#' y_j <- LazyTensor(y, index = 'j')
#' z_i <- LazyTensor(z, index = 'i')
#' 
#' # call mod function
#' mod_x72 <- mod(x_i, 7, 2)
#' 
#' # works also with LazyTensors with same inner dimension or dimension 1
#' mod_xyz <- mod(x_i, y_j, z_i)
#' }
#' @export
mod.LazyTensor <- function(x, a, b = 0, ...) {
    return(ternaryop.LazyTensor(x, a, b, "Mod"))
}


# SIMPLE NORM AND DISTANCE OPERATIONS ==========================================

# Squared Euclidean norm -------------------------------------------------------

#' Squared Euclidean norm
#' @description
#' Symbolic squared Euclidean norm operation for `LazyTensor` objects.
#' @details `sqnorm2(x)` returns a `LazyTensor` that encodes, symbolically,
#' the squared Euclidean norm of `x`, same as `(x|x)`.
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param x a `LazyTensor`, a `ComplexLazyTensor`, a vector of numeric values, 
#' or a scalar value.
#' @return An object of class `LazyTensor`.
#' @seealso [rkeops::|.LazyTensor()]
#' @examples
#' \dontrun{
#' x <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows, 3 columns
#' x_i <- LazyTensor(x, index = 'i')   # creating LazyTensor from matrix x, 
#'                                     # indexed by 'i'
#' SqN_x <- sqnorm2(x_i)               # symbolic matrix, 150 rows, 1 columns
#' }
#' @export
sqnorm2 <- function(x) {
    return(
        unaryop.LazyTensor(x, "SqNorm2", res_type = "LazyTensor", dim_res = 1))
}


# Euclidean norm ---------------------------------------------------------------

#' Euclidean norm
#' @description
#' Symbolic Euclidean norm operation for `LazyTensor` objects.
#' @details `norm2(x)` returns a `LazyTensor` that encodes, symbolically,
#' the Euclidean norm of `x`, same as `sqrt(x|x)`.
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param x a `LazyTensor`, a `ComplexLazyTensor`, a vector of numeric values, 
#' or a scalar value.
#' @return An object of class `LazyTensor`.
#' @seealso [rkeops::|.LazyTensor()], [rkeops::sqrt()]
#' @examples
#' \dontrun{
#' x <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows, 3 columns
#' x_i <- LazyTensor(x, index = 'i')   # creating LazyTensor from matrix x, 
#'                                     # indexed by 'i'
#' N_x <- norm2(x_i)                   # symbolic matrix, 150 rows and 3 columns
#' }
#' @export
norm2 <- function(x) {
    return(
        unaryop.LazyTensor(x, "Norm2", res_type = "LazyTensor", dim_res = 1))
}


# Vector normalization ---------------------------------------------------------

#' Vector normalization
#' @description
#' Symbolic vector normalization operation for `LazyTensor` objects.
#' @details `normalize(x)` returns a `LazyTensor` that encodes, symbolically,
#' the vector normalization of `x` (with the Euclidean norm),
#' same as `rsqrt(sqnorm2(x)) * x`.
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param x a `LazyTensor`, a `ComplexLazyTensor`, a vector of numeric values, 
#' or a scalar value.
#' @return An object of class `LazyTensor`.
#' @seealso [rkeops::sqnorm2()], [rkeops::rsqrt()]
#' @examples
#' \dontrun{
#' x <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows, 3 columns
#' x_i <- LazyTensor(x, index = 'i')   # creating LazyTensor from matrix x, 
#'                                     # indexed by 'i'
#' N_x <- norm2(x_i)                   # symbolic matrix, 150 rows and 3 columns
#' }
#' @export
normalize <- function(x) {
    return(unaryop.LazyTensor(x, "Normalize"))
}



# Squared distance -------------------------------------------------------------

#' Squared distance
#' @description
#' Symbolic squared Euclidean distance binary operation for `LazyTensor` 
#' objects.
#' @details `sqdist(x)` returns a `LazyTensor` that encodes, symbolically,
#' the squared Euclidean distance between `x` and `y`, same as `sqnorm2(x - y)`.
#' 
#' **Note**: `x` and `y` input arguments should have the same inner dimension 
#' or be of dimension 1.
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param x a `LazyTensor`, a `ComplexLazyTensor`, a vector of numeric values, 
#' or a scalar value.
#' @param y a `LazyTensor`, a `ComplexLazyTensor`, a vector of numeric values, 
#' or a scalar value.
#' @return An object of class `LazyTensor`.
#' @seealso [rkeops::sqnorm2()]
#' @examples
#' \dontrun{
#' x <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows, 3 columns
#' x_i <- LazyTensor(x, index = 'i')   # creating LazyTensor from matrix x, 
#'                                     # indexed by 'i'
#'                                     
#' SqD_x <- sqdist(x_i)                # symbolic matrix, 150 rows and 3 columns
#' }
#' @export
sqdist <- function(x, y) {
    return(binaryop.LazyTensor(x, y, "SqDist", res_type = "LazyTensor", 
                               dim_res = 1))
}


# Weighted squared norm --------------------------------------------------------

#' Generic weighted squared Euclidean norm
#' @description
#' Symbolic weighted squared norm of a LazyTensor binary operation for 
#' `LazyTensor` objects.
#' @details `weightedsqnorm(x, s)` returns a `LazyTensor` that encodes, 
#' symbolically, the weighted squared norm of a vector `x` with weights stored 
#' in the LazyTensor `s`.
#' 
#' **Note**: `x`, and `s` should have the same inner dimension.
#' 
#' **Note**: Run `browseVignettes("rkeops")` to access the vignettes and find 
#' details about this function in the *"RKeOps LazyTensor"* vignette, at 
#' section *"Simple vector operations"*.
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param x a `vector` of numeric values or a scalar value.
#' @param s a `LazyTensor`, a `ComplexLazyTensor`, a vector of numeric values, 
#' or a scalar value. 
#' @return An object of class `LazyTensor`.
#' @seealso [rkeops::sqnorm2()], [rkeops::weightedsqdist()]
#' @examples
#' \dontrun{
#' x <- matrix(runif(100 * 3), 100, 3) # arbitrary R matrix, 100 rows, 3 columns
#' s <- matrix(runif(100 * 3), 100, 3) # arbitrary R matrix, 100 rows, 3 columns
#' x_i <- LazyTensor(x, index = 'i')   # creating LazyTensor from matrix x, 
#'                                     # indexed by 'i'
#' s_j <- LazyTensor(s, index = 'j')   # creating LazyTensor from matrix s, 
#'                                     # indexed by 'j'
#' wsqn_xy <- weightedsqnorm(x_i, s_j) # symbolic matrix, 100 rows,1 columns
#' }
#' @export
weightedsqnorm <- function(x, s) {
    return(binaryop.LazyTensor(s, x, "WeightedSqNorm",
                               dim_check_type = NA,
                               res_type = "LazyTensor",
                               dim_res = 1))
}


# Weighted squared distance ----------------------------------------------------

#' Generic weighted squared distance
#' @description
#' Symbolic weighted squared distance binary operation for `LazyTensor` 
#' objects.
#' @details `weightedsqdist(x, y, s)` returns a `LazyTensor` that encodes, 
#' symbolically, the weighted squared distance of a vector `x` with weights 
#' stored in the LazyTensor `s`, same as `weightedsqnorm(x - y, s)`.
#' 
#' **Note**: `x`, `y` and `s` should all have the same inner dimension.
#' 
#' **Note**: Run `browseVignettes("rkeops")` to access the vignettes and find 
#' details about this function in the *"RKeOps LazyTensor"* vignette, at 
#' section *"Simple vector operations"*.
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param x A `vector` of numeric values or a scalar value.
#' @param y a `LazyTensor`, a `ComplexLazyTensor`, a vector of numeric values, 
#' or a scalar value.
#' @param s a `LazyTensor`, a `ComplexLazyTensor`, a vector of numeric values, 
#' or a scalar value.
#' @return An object of class `LazyTensor`.
#' @seealso [rkeops::weightedsqnorm()], [rkeops::sqdist()]
#' @examples
#' \dontrun{
#' x <- matrix(runif(100 * 3), 100, 3) # arbitrary R matrix, 100 rows, 3 columns
#' y <- matrix(runif(100 * 3), 100, 3) # arbitrary R matrix, 100 rows, 3 columns
#' s <- matrix(runif(100 * 3), 100, 3) # arbitrary R matrix, 100 rows, 3 columns
#' 
#' x_i <- LazyTensor(x, index = 'i')   # creating LazyTensor from matrix x, 
#'                                     # indexed by 'i'
#' y_j <- LazyTensor(y, index = 'j')   # creating LazyTensor from matrix y, 
#'                                     # indexed by 'j'
#' s_i <- LazyTensor(s, index = 'i')   # creating LazyTensor from matrix s, 
#'                                     # indexed by 'i'
#' 
#' wsqd_xy <- weightedsqdist(x_i, y_j, s_i)    # symbolic matrix
#' }
#' @export
weightedsqdist <- function(x, y, s) {
    return(weightedsqnorm(x - y, s))
}


# COMPLEX FUNCTIONS ============================================================


# real -------------------------------------------------------------------------

#' Complex Numbers and Basic Functionality
#' @name complex.default
#' @aliases Re.default
#' @inherit base::Re description
#' @inherit base::Re details
#' @inherit base::Re params
#' @inherit base::Re return
#' @inherit base::Re examples
#' @seealso [base::Re()]
#' @author R core team and contributors
#' @export
Re.default <- function(z) {
    return(base::Re(z))
}

#' Element-wise complex real part operation
#' @author Chloe Serre-Combe, Amelie Vernay
#' @description
#' Symbolic element-wise complex real part operation for `ComplexLazyTensor`
#' objects or default R element-wise complex real part operation otherwise.
#' @details If `z` is a `ComplexLazyTensor`, see 
#' [rkeops::Re.LazyTensor()], else see [rkeops::Re.default()].
#' @param z input for [rkeops::Re.default()] or [rkeops::Re.LazyTensor()].
#' @return See value of [rkeops::Re.default()] or [rkeops::Re.LazyTensor()].
#' @seealso [rkeops::Re.default()], [rkeops::Re.LazyTensor()], 
#' [rkeops::Im()], [rkeops::Arg()], [rkeops::Mod()], [rkeops::Conj()]
#' @examples
#' \dontrun{
#' # R base operation
#' Re(1)
#' Re(1+1i)
#' # LazyTensor symbolic element-wise complex real part
#' z <- matrix(2 + 1i^ (-6:5), nrow = 4)        # complex matrix
#' z_i <- LazyTensor(z, "i", is_complex = TRUE) # creating ComplexLazyTensor
#' Re_z <- Re(z_i)                              # symbolic matrix
#' }
#' @export
Re <- function(z) {
    UseMethod("Re", z)
}

#' Element-wise complex real part
#' @name Re.LazyTensor
#' @description
#' Symbolic element-wise complex real part operation for `ComplexLazyTensor`
#' objects.
#' @details If `z` is a `ComplexLazyTensor`, `Re(z)` returns a 
#' `ComplexLazyTensor` that encodes, symbolically, the element-wise real part 
#' of complex `z`.
#' 
#' **Note**: `Re(z)` will exit with an error if `z` is a `LazyTensor` but not
#' a `ComplexLazyTensor`.
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param z A `ComplexLazyTensor` or any type of values accepted by R default 
#' [base::Re()] function.
#' @return An object of class `ComplexLazyTensor`.
#' @seealso [rkeops::Re()]
#' @examples
#' \dontrun{
#' z <- matrix(2 + 1i^ (-6:5), nrow = 4)        # complex matrix
#' z_i <- LazyTensor(z, "i", is_complex = TRUE) # creating ComplexLazyTensor
#' Re_z <- Re(z_i)                              # symbolic matrix
#' }
#' @export
Re.LazyTensor <- function(z) {
    msg <- paste(
        "`Re` cannot be applied to a LazyTensor.", 
        "See `?Re` for compatible types."
    )
    stop(msg)
}

#' Element-wise complex real part
#' @name Re.LazyTensor
#' @aliases Re.ComplexLazyTensor
#' @author Chloe Serre-Combe, Amelie Vernay
#' @export
Re.ComplexLazyTensor <- function(z) {
    return(unaryop.LazyTensor(z, "ComplexReal"))
}


# imaginary --------------------------------------------------------------------

#' Complex Numbers and Basic Functionality
#' @name complex.default
#' @aliases Im.default
#' @seealso [base::Im()]
#' @export
Im.default <- function(z) {
    return(base::Im(z))
}

#' Element-wise complex imaginery part operation
#' @author Chloe Serre-Combe, Amelie Vernay
#' @description
#' Symbolic element-wise complex imaginery part operation for 
#' `ComplexLazyTensor` objects or default element-wise complex imaginery part
#' operation otherwise.
#' @details If `z` is a `LazyTensor`, see [rkeops::Im.LazyTensor()], else 
#' see [rkeops::Im.default()].
#' @param z input for [rkeops::Im.default()] or [rkeops::Im.LazyTensor()].
#' @return See value of [rkeops::Im.default()] or [rkeops::Im.LazyTensor()].
#' @seealso [rkeops::Im.default()], [rkeops::Im.LazyTensor()], 
#' [rkeops::Re()], [rkeops::Arg()], [rkeops::Mod()], [rkeops::Conj()]
#' @examples
#' \dontrun{
#' # R base operation
#' Im(1)
#' Im(1+1i)
#' # LazyTensor symbolic element-wise complex imaginery part
#' z <- matrix(2 + 1i^ (-6:5), nrow = 4)         # complex matrix
#' z_i <- LazyTensor(z, "i", is_complex = TRUE)  # creating ComplexLazyTensor
#' Im_z <- Im(z_i)                               # symbolic matrix
#' }
#' @export
Im <- function(z) {
    UseMethod("Im", z)
}

#' Element-wise complex imaginery part
#' @name Im.LazyTensor
#' @description
#' Symbolic element-wise complex imaginery part operation for 
#' `ComplexLazyTensor` objects.
#' @details If `z` is a `ComplexLazyTensor`, `Im(z)` returns a 
#' `ComplexLazyTensor` that encodes, symbolically, the element-wise imaginary
#' part of complex `z`.
#' 
#' **Note**: `Im(z)` will exit with an error if `z` is a `LazyTensor` but not
#' a `ComplexLazyTensor`.
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param z A `ComplexLazyTensor` or any type of values accepted by R default 
#' [base::Im()] function.
#' @return An object of class `ComplexLazyTensor`.
#' @seealso [rkeops::Im()]
#' @examples
#' \dontrun{
#' z <- matrix(2 + 1i^ (-6:5), nrow = 4)        # complex matrix
#' z_i <- LazyTensor(z, "i", is_complex = TRUE) # creating ComplexLazyTensor
#' Im_z <- Im(z_i)                              # symbolic matrix
#' }
#' @export
Im.LazyTensor <- function(z) {
    msg <- paste(
        "`Im` cannot be applied to a LazyTensor.",
        "See `?Im` for compatible types."
    )
    stop(msg)
}

#' Element-wise complex imaginery part
#' @name Im.LazyTensor
#' @aliases Im.ComplexLazyTensor
#' @author Chloe Serre-Combe, Amelie Vernay
#' @export
Im.ComplexLazyTensor <- function(z) {
    return(unaryop.LazyTensor(z, "ComplexImag"))
}


# angle ------------------------------------------------------------------------

#' Complex Numbers and Basic Functionality
#' @name complex.default
#' @aliases Arg.default
#' @seealso [base::Arg()]
#' @export
Arg.default <- function(z) {
    return(base::Arg(z))
}

#' Element-wise argument (or angle) of complex operation
#' @author Chloe Serre-Combe, Amelie Vernay
#' @description
#' Symbolic element-wise argument (or angle) of complex operation for 
#' `ComplexLazyTensor` objects or default element-wise argument (or angle) of 
#' complex operation otherwise.
#' @details If `z` is a `LazyTensor`, see [rkeops::Arg.LazyTensor()], else 
#' see [rkeops::Arg.default()].
#' @param z input for [rkeops::Arg.default()] or [rkeops::Arg.LazyTensor()].
#' @return See value of [rkeops::Arg.default()] or [rkeops::Arg.LazyTensor()].
#' @seealso [rkeops::Arg.default()], [rkeops::Arg.LazyTensor()], 
#' [rkeops::Re()], [rkeops::Im()], [rkeops::Mod()], [rkeops::Conj()]
#' @examples
#' \dontrun{
#' # R base operation
#' Arg(1)
#' Arg(1+1i)
#' # LazyTensor symbolic element-wise argument (or angle) of complex
#' z <- matrix(2 + 1i^ (-6:5), nrow = 4)         # complex matrix
#' z_i <- LazyTensor(z, "i", is_complex = TRUE)  # creating ComplexLazyTensor
#' Arg_z <- Arg(z_i)                             # symbolic matrix
#' }
#' @export
Arg <- function(z) {
    UseMethod("Arg", z)
}

#' Element-wise argument (or angle) of complex
#' @name Arg.LazyTensor
#' @description
#' Symbolic element-wise argument (or angle) of complex operation for 
#' `LazyTensor` objects.
#' @details If `z` is a `ComplexLazyTensor`, `Arg(z)` returns a 
#' `ComplexLazyTensor` that encodes, symbolically, the element-wise imaginary
#' part of complex `z`.
#' 
#' **Note**: `Arg(z)` will exit with an error if `z` is a `LazyTensor` but not
#' a `ComplexLazyTensor`.
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param z A `ComplexLazyTensor` or any type of values accepted by R default 
#' [base::Arg()] function.
#' @return An object of class `ComplexLazyTensor`.
#' @seealso [rkeops::Arg()]
#' @examples
#' \dontrun{
#' z <- matrix(2 + 1i^ (-6:5), nrow = 4)        # complex matrix
#' z_i <- LazyTensor(z, "i", is_complex = TRUE) # creating ComplexLazyTensor
#' Arg_z <- Arg(z_i)                            # symbolic matrix
#' }
#' @export
Arg.LazyTensor <- function(z) {
    msg <- paste(
        "`Arg` cannot be applied to a LazyTensor.",
        "See `?Arg` for compatible types."
    )
    stop(msg)
}

#' Element-wise argument (or angle) of complex
#' @name Arg.LazyTensor
#' @aliases Arg.ComplexLazyTensor
#' @author Chloe Serre-Combe, Amelie Vernay
#' @export
Arg.ComplexLazyTensor <- function(z) {
    return(unaryop.LazyTensor(z, "ComplexAngle"))
}


# real to complex --------------------------------------------------------------

#' Element-wise "real to complex" operation.
#' @name real2complex
#' @description
#' Symbolic element-wise "real to complex" operation.
#' @details `real2complex(x)` returns a `ComplexLazyTensor` that encodes, 
#' symbolically, the element-wise "real to complex" transformation of `x` 
#' (i.e. with additional zero imaginary part: `x + 0*i`).
#' 
#' **Note**: `real2complex(x)` will exit with an error if `x` is 
#' a `ComplexLazyTensor`.
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param x a `LazyTensor`.
#' @return An object of class `ComplexLazyTensor`.
#' @examples
#' \dontrun{
#' x <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows, 3 columns
#' x_i <- LazyTensor(x, "i")           # creating LazyTensor from matrix x, 
#'                                     # indexed by 'i'
#' z <- real2complex(x_i)              # ComplexLazyTensor object
#' }
#' @export
real2complex <- function(x) {
    UseMethod("real2complex", x)
}

#' Element-wise "real 2 complex" operation
#' @name real2complex
#' @aliases real2complex.LazyTensor
#' @author Chloe Serre-Combe, Amelie Vernay
#' @export
real2complex.LazyTensor <- function(x) {
    cplx_warning()
    return(
        unaryop.LazyTensor(x, "Real2Complex", res_type = "ComplexLazyTensor"))
}

#' Element-wise "real to complex" operation
#' @name real2complex
#' @aliases real2complex.LazyTensor
#' @author Chloe Serre-Combe, Amelie Vernay
#' @export
real2complex.ComplexLazyTensor <- function(x) {
    stop("`real2complex` cannot be applied to a ComplexLazyTensor.")
}


# imaginary to complex ---------------------------------------------------------

#' Element-wise "imaginary to complex" operation.
#' @name imaginary2complex
#' @description
#' Symbolic operation for element-wise "imaginary to complex".
#' @details `imag2complex(x)` returns a `ComplexLazyTensor` that encodes,
#' symbolically, the element-wise "imaginary to complex" transformation 
#' of `x` (i.e. with additional zero real part: `0 + x*i`).
#' 
#' **Note**: `imag2complex(x)` will exit with an error if `x` is 
#' a `ComplexLazyTensor`.
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param x a `LazyTensor`.
#' @return An object of class `ComplexLazyTensor`.
#' @examples
#' \dontrun{
#' x <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows, 3 columns
#' x_i <- LazyTensor(x, "i")           # creating LazyTensor from matrix x,
#'                                     # indexed by 'i'
#' z <- imag2complex(x_i)              # ComplexLazyTensor object
#' }
#' @export
imag2complex <- function(x) {
    UseMethod("imag2complex", x)
}

#' Element-wise "imag 2 complex" operation
#' @name imaginary2complex
#' @author Chloe Serre-Combe, Amelie Vernay
#' @export
imag2complex.LazyTensor <- function(x) {
    return(
        unaryop.LazyTensor(x, "Imag2Complex", res_type = "ComplexLazyTensor"))
}

#' Element-wise "imag 2 complex" operation
#' @name imaginary2complex
#' @author Chloe Serre-Combe, Amelie Vernay
#' @export
imag2complex.ComplexLazyTensor <- function(x) {
    stop("`imag2complex` cannot be applied to a ComplexLazyTensor.")
}


# complex exponential of 1j x --------------------------------------------------

#' Element-wise "complex exponential of 1j x" operation.
#' @name exp1j
#' @description
#' Symbolic element-wise "complex exponential of 1j x" operation for 
#' `LazyTensor` object.
#' @details `exp1j(x)` returns a `ComplexLazyTensor` that encodes, symbolically,
#' the multiplication of `1j` with `x`.
#' 
#' **Note**: `exp1j(x)` will exit with an error if `x` is 
#' a `ComplexLazyTensor`.
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param x A `LazyTensor`.
#' @return An object of class "ComplexLazyTensor".
#' @examples
#' \dontrun{
#' x <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows, 3 columns
#' x_i <- LazyTensor(x, "i")           # creating LazyTensor from matrix x, 
#'                                     # indexed by 'i'
#' z <- exp1j(x_i)                     # ComplexLazyTensor object
#' }
#' @export
exp1j <- function(x) {
    UseMethod("exp1j", x)
}

#' Element-wise "complex exponential of 1j x" operation.
#' @name exp1j
#' @author Chloe Serre-Combe, Amelie Vernay
#' @export
exp1j.LazyTensor <- function(x) {
    return(
        unaryop.LazyTensor(x, "ComplexExp1j", res_type = "ComplexLazyTensor"))
}

#' Element-wise "complex exponential of 1j x" operation
#' @name exp1j
#' @author Chloe Serre-Combe, Amelie Vernay
#' @export
exp1j.ComplexLazyTensor <- function(x) {
    stop("`exp1j` cannot be applied to a ComplexLazyTensor.")
}


# complex conjugate ------------------------------------------------------------

#' Complex Numbers and Basic Functionality
#' @name complex.default
#' @aliases Conj.default
#' @seealso [base::Conj()]
#' @export
Conj.default <- function(z) {
    return(base::Conj(z))
}

#' Element-wise complex conjugate operation
#' @author Chloe Serre-Combe, Amelie Vernay
#' @description
#' Symbolic element-wise complex conjugate operation for 
#' `ComplexLazyTensor` objects or default element-wise complex conjugate
#' operation otherwise.
#' @details If `z` is a `LazyTensor`, see [rkeops::Conj.LazyTensor()], else 
#' see [rkeops::Conj.default()].
#' @param z input for [rkeops::Conj.default()] or [rkeops::Conj.LazyTensor()].
#' @return See value of [rkeops::Conj.default()] or [rkeops::Conj.LazyTensor()].
#' @seealso [rkeops::Conj.default()], [rkeops::Conj.LazyTensor()], 
#' [rkeops::Re()], [rkeops::Im()], [rkeops::Arg()], [rkeops::Mod()]
#' @examples
#' \dontrun{
#' # R base operation
#' Conj(1)
#' Conj(1+1i)
#' # LazyTensor symbolic element-wise complex conjugate
#' z <- matrix(1i^ (-6:5), nrow = 4)                     # complex 4x3 matrix
#' z_i <- LazyTensor(z, index = 'i', is_complex = TRUE)  # ComplexLazyTensor
#' Conj_z_i <- Conj(z_i)                                 # symbolic matrix
#' }
#' @export
Conj <- function(z) { 
    UseMethod("Conj", z)
}

#' Element-wise complex conjugate
#' @name Conj.LazyTensor
#' @description
#' Symbolic element-wise complex conjugate operation for 
#' `ComplexLazyTensor` objects.
#' @details If `z` is a `ComplexLazyTensor`, `Conj(z)` returns a 
#' `ComplexLazyTensor` that encodes, symbolically, the element-wise complex 
#' conjugate of `z`.
#' 
#' **Note**: `Conj(z)` will exit with an error if `z` is a `LazyTensor` but not
#' a `ComplexLazyTensor`.
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param z A `ComplexLazyTensor` or any type of values accepted by R default 
#' [base::Conj()] function.
#' @return An object of class `ComplexLazyTensor`.
#' @seealso [rkeops::Conj()]
#' @examples
#' \dontrun{
#' z <- matrix(1i^ (-6:5), nrow = 4)                     # complex 4x3 matrix
#' z_i <- LazyTensor(z, index = 'i', is_complex = TRUE)  # ComplexLazyTensor
#' Conj_z_i <- Conj(z_i)                                 # symbolic matrix
#' }
#' @export
Conj.LazyTensor <- function(z) {
    msg <- paste(
        "`Conj` cannot be applied to a LazyTensor.",
        "See `?Conj` for compatible types."
    )
    stop(msg)
}

#' Element-wise complex conjugate
#' @name Conj.LazyTensor
#' @aliases Conj.ComplexLazyTensor
#' @author Chloe Serre-Combe, Amelie Vernay
#' @export
Conj.ComplexLazyTensor <- function(z) {
    return(unaryop.LazyTensor(z, "Conj"))
}


# complex modulus --------------------------------------------------------------

#' Complex Numbers and Basic Functionality
#' @name complex.default
#' @aliases Mod.default
#' @seealso [base::Mod()]
#' @export
Mod.default <- function(z) {
    return(base::Mod(z))
}

#' Element-wise complex modulus (absolute value) operation
#' @author Chloe Serre-Combe, Amelie Vernay
#' @description
#' Symbolic element-wise complex modulus (absolute value) operation for 
#' `ComplexLazyTensor` objects or default element-wise complex modulus 
#' (absolute value) operation otherwise.
#' @details If `z` is a `LazyTensor`, see [rkeops::Mod.LazyTensor()], else 
#' see [rkeops::Mod.default()].
#' 
#' **Warning**: Do not confuse with [rkeops::mod()].
#' @param z input for [rkeops::Mod.default()] or [rkeops::Mod.LazyTensor()].
#' @return See value of [rkeops::Mod.default()] or [rkeops::Mod.LazyTensor()].
#' @seealso [rkeops::Mod.default()], [rkeops::Mod.LazyTensor()], 
#' [rkeops::Re()], [rkeops::Im()], [rkeops::Arg()], [rkeops::Conj()]
#' @examples
#' \dontrun{
#' # R base operation
#' Mod(1)
#' Mod(1+1i)
#' # LazyTensor symbolic element-wise complex modulus (absolute value)
#' z <- matrix(1i^ (-6:5), nrow = 4)                     # complex 4x3 matrix
#' z_i <- LazyTensor(z, index = 'i', is_complex = TRUE)  # ComplexLazyTensor
#' Mod_z_i <- Mod(z_i)                                   # symbolic matrix
#' }
#' @export
Mod <- function(z) { 
    UseMethod("Mod", z)
}

#' Element-wise complex modulus (absolute value)
#' @name Mod.LazyTensor
#' @description
#' Symbolic element-wise complex modulus (absolute value) operation for 
#' `ComplexLazyTensor` objects.
#' @details If `z` is a `ComplexLazyTensor`, `Mod(z)` returns a 
#' `ComplexLazyTensor` that encodes, symbolically, the element-wise conjugate of
#' complex `z`.
#' 
#' **Note**: `Mod(z)` will exit with an error if `z` is a `LazyTensor` but not
#' a `ComplexLazyTensor`.
#' 
#' **Warning**: Do not confuse with [rkeops::mod()].
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param z A `ComplexLazyTensor` or any type of values accepted by R default 
#' [base::Mod()] function.
#' @return An object of class `ComplexLazyTensor`.
#' @seealso [rkeops::Mod()]
#' @examples
#' \dontrun{
#' z <- matrix(1i^ (-6:5), nrow = 4)                     # complex 4x3 matrix
#' z_i <- LazyTensor(z, index = 'i', is_complex = TRUE)  # ComplexLazyTensor
#' Mod_z_i <- Mod(z_i)                                   # symbolic matrix
#' }
#' @export
Mod.LazyTensor <- function(z) {
    msg <- paste(
        "`Mod` cannot be applied to a LazyTensor.",
        "See `?Mod` for compatible types."
    )
    stop(msg)
}

#' Element-wise complex modulus (absolute value)
#' @name Mod.LazyTensor
#' @aliases Mod.ComplexLazyTensor
#' @author Chloe Serre-Combe, Amelie Vernay
#' @export
Mod.ComplexLazyTensor <- function(z) {
    return(unaryop.LazyTensor(z, "ComplexAbs", res_type = "LazyTensor"))
}


# CONSTANT AND PADDING/CONCATENATION OPERATIONS ================================

# Elem -------------------------------------------------------------------------

#' Extract an element
#' @description
#' Symbolic element extraction operation for `LazyTensor` objects.
#' @details 
#' If `x` is a `LazyTensor`, `elem(x, m)` returns a `LazyTensor` that 
#' encodes, symbolically, the `m+1`-th element of `x`, i.e. `x[m+1]` in 
#' standard R notation.
#' 
#' **IMPORTANT**: IN THIS CASE, INDICES START AT ZERO, therefore, `m` should 
#' be in `[0, n)`, where `n` is the inner dimension of `x`.
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param x a `LazyTensor` or a `ComplexLazyTensor`.
#' @param m an `integer` corresponding to the index (starting from 0) of 
#' the element of `x` that will be extracted.
#' @return a `LazyTensor` or a `ComplexLazyTensor`.
#' @examples
#' \dontrun{
#' x <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows, 3 columns
#' x_i <- LazyTensor(x, index = 'i')   # LazyTensor from matrix x, indexed by 'i'
#' m <- 2
#' 
#' elem_x <- elem(x_i, m)  # symbolic `m+1`-th element of `x_i`.
#' }
#' @export
elem <- function(x, m) {
    if(!is.int(m)) {
        stop("`m` input argument should be an integer.")
    }
    D <- x$dimres
    if(m < 0 || m >= D) {
        stop(paste("Index `m` is out of bounds. Should be in [0, ",
                   D, ").", sep = ""))
    }
    return(unaryop.LazyTensor(x, "Elem", m, dim_res = 1))
}


# ElemT ------------------------------------------------------------------------

#' Insert element in a vector of zeros
#' @description
#' Insert a given value (stored in a `LazyTensor`) in a symbolic vector of 
#' zeros.
#' @details 
#' `elemT(x, m, n)` insert scalar value `x` (encoded as a `LazyTensor`) at
#' position `m+1` in a vector of zeros of length `n` (encoded as a `LazyTensor`
#' of inner dimension `n`).
#' 
#' **Note**: Input `x` should be a `LazyTensor` encoding a single parameter
#' value.
#' 
#' **IMPORTANT**: IN THIS CASE, INDICES START AT ZERO, therefore, `m` should 
#' be in `[0, n)`.
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param x a `LazyTensor` or a `ComplexLazyTensor` encoding a single
#' parameter value.
#' @param m an `integer` corresponding to the position `m` (starting from 0) 
#' of the created vector of zeros at which we want to insert the value `x`.
#' @param n an `integer` corresponding to the length of the vector of zeros.
#' @return a `LazyTensor` or a `ComplexLazyTensor`.
#' @examples
#' \dontrun{
#' # basic example
#' x <- 3.14              # arbitrary value
#' Pm_x <- LazyTensor(x)  # creating scalar parameter LazyTensor from x
#' 
#' m <- 2
#' n <- 3
#' 
#' elemT_x <- elemT(Pm_x, m, n)  # symbolic vector
#' }
#' @export
elemT <- function(x, m, n) {
    if(!is.LazyParameter(x) && !is.ComplexLazyParameter(x)) {
        stop(paste("`x` input argument should be a `LazyTensor`",
                   " encoding a single value.", sep = ""))
    }
    if(!is.int(m)) {
        stop("`m` input argument should be an integer.")
    }
    if(!is.int(n)) {
        stop("`n` input argument should be an integer.")
    }
    if(m < 0 || m >= n) {
        stop(paste("Index `m` is out of bounds. Should be in [0, ",
                   n, ").", sep = ""))
    }
    
    return(unaryop.LazyTensor(x, "ElemT", opt_arg = n, opt_arg2 = m))
}


# Extract ----------------------------------------------------------------------

#' Extract a range of elements
#' @description
#' Symbolic extraction range of elements for `LazyTensor` objects.
#' @details `extract(x_i, m, d)` encodes, symbolically, the extraction
#' of a range of values `x[m:m+d]` in the `LazyTensor` `x`; (`m` is the 
#' starting index, and `d` is the dimension of the extracted sub-vector).
#'
#' **IMPORTANT**: IN THIS CASE, INDICES START AT ZERO, therefore, `m` should 
#' be in `[0, n)`, where `n` is the inner dimension of `x`. And `d` should be 
#' in `[0, n-m]`.
#' 
#' **Note**: See the examples for a more concrete explanation regarding the 
#' use of `extract()`.
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param x a `LazyTensor` or a `ComplexLazyTensor`.
#' @param m an `integer` corresponding to the starting index (starting from 0) 
#' of the range of elements from `x` that will be extracted.
#' @param d an `integer` corresponding to the output dimension, i.e. the number 
#' of consecutive elements from `x` that will be extracted.
#' @return a `LazyTensor`.
#' @examples
#' \dontrun{
#' # Two very rudimentary examples
#' # -----------------------------
#' 
#' # Let's say that you have a matrix `g` looking like this:
#' #      [,1] [,2] [,3] [,4]
#' # [1,]    1    8    1    3
#' # [2,]    2    1    2    7
#' # [3,]    3    7    4    5
#' # [4,]    1    3    3    0
#' # [5,]    5    4    9    4
#' 
#' # Convert it to LazyTensor:
#' g_i <- LazyTensor(g, index = 'i')
#' 
#' # Then extract some elements:
#' ext_g <- extract(g_i, 1, 3)
#' 
#' # In this case, `ext_g` is a LazyTensor encoding, symbolically,
#' # the following part of g:
#' #       [,1] [,2] [,3]
#' # [1,]    8    1    3
#' # [2,]    1    2    7
#' # [3,]    7    4    5
#' # [4,]    3    3    0
#' # [5,]    4    9    4
#' 
#' 
#' # Same principle with a LazyTensor encoding a vector:
#' v <- c(1, 2, 3, 1, 5)
#' Pm_v <- LazyTensor(v)
#' 
#' ext_Pm_v <- extract(Pm_v, 2, 3)
#' 
#' # In this case, `ext_Pm_v` is a LazyTensor encoding, symbolically,
#' # the following part of v:
#' #       [,1] [,2] [,3]
#' # [1,]    3    1    5
#' 
#' 
#' # A more general example
#' # ----------------------
#' 
#' x <- matrix(runif(150 * 5), 150, 5) # arbitrary R matrix, 150 rows, 5 columns
#' x_i <- LazyTensor(x, index = 'i')   # LazyTensor from matrix x, indexed by 'i'
#' m <- 2
#' d <- 2
#' 
#' extract_x <- extract(x_i, m, d) # symbolic matrix
#' }
#' @export
extract <- function(x, m, d) {
    # type check
    if(!is.int(m))
        stop("`m` input argument should be an integer.")
    if(!is.int(d)) 
        stop("`d` input argument should be an integer.")
    # dim check
    D <- x$dimres
    if(m < 0 || m >= D) {
        stop(paste("Index `m` is out of bounds. Should be in [0, ",
                   D, ").", sep = ""))
    }
    if(d < 0 || (D < (m + d))) {
        stop(
            paste("Slice dimension is out of bounds. Input `d` should be in [0, ",
                  D, "-m] where `m` is the starting index.", sep = "")
        )
    }
    return(unaryop.LazyTensor(x, "Extract",
                              opt_arg = m, opt_arg2 = d,
                              dim_res = d))
}


# ExtractT ---------------------------------------------------------------------

#' Insert a range of elements
#' @description
#' Insert a given value, vector of values or matrix of values (stored in a 
#' `LazyTensor`) in a symbolic vector or matrix of zeros.
#' @details If `x` is a `LazyTensor` encoding a vector (resp. a matrix),
#' `extractT(x, m, d)` encodes, symbolically, a `d`-inner-dimensional
#' vector (resp. matrix) of zeros in which is inserted `x`,
#' at starting position `m+1`.
#' 
#' **IMPORTANT**: IN THIS CASE, INDICES START AT ZERO, therefore, `m` should 
#' be in `[0, n)`, where `n` is the inner dimension of `x`. And `d` should be 
#' in `[0, n-m]`.
#' 
#' **Note 1**: `x` can also encode a single value, in which case the 
#' operation works the same way as in the case of a vector of values.
#' 
#' **Note 2**: See the examples for a more concrete explanation regarding the 
#' use of `extractT()`.
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param x a `LazyTensor` or a `ComplexLazyTensor`.
#' @param m an `integer` corresponding to the starting index (starting from 0) 
#' where the elements from `x` will be inserted.
#' @param d an `integer` corresponding to the output inner dimension.
#' @return A `LazyTensor`.
#' @examples
#' \dontrun{
#' # I - Three very rudimentary examples
#' # -----------------------------------
#' 
#' # 1) Let's say that you have a matrix `g` looking like this:
#' #      [,1] [,2]
#' # [1,]    1    4
#' # [2,]    2    5
#' # [3,]    3    6
#' 
#' # Convert it to LazyTensor:
#' g_i <- LazyTensor(g, index = 'i') # indexed by 'i' (for example)
#' 
#' # Then insert it in a matrix of inner dimension equal to 5,
#' # starting at index 1:
#' extT_g <- extractT(g_i, 1, 5)
#' 
#' # In this case, `extT_g` is a LazyTensor encoding, symbolically,
#' # the following matrix:
#' #      [,1] [,2] [,3] [,4] [,5]
#' # [1,]    0    1    4    0    0
#' # [2,]    0    2    5    0    0
#' # [3,]    0    3    6    0    0
#' 
#' 
#' # 2) Same principle with a LazyTensor encoding a vector:
#' v <- c(1, 2, 3, 1, 5)
#' Pm_v <- LazyTensor(v)
#' 
#' extT_Pm_v <- extract(Pm_v, 2, 3)
#' 
#' # In this case, `extT_Pm_v` is a LazyTensor encoding, symbolically,
#' # the following vector:
#' #       [,1] [,2] [,3] [,4] [,5] [,6] [,7] [,8]
#' # [1,]    0    1    2    3    1    5    0    0
#' 
#' 
#' # 3) Same again with a scalar value:
#' scal <- 3.14
#' Pm_scal <- Pm(scal) # `Pm(x)` is an aliases for `LazyTensor(x, index = NA)`
#' 
#' extT_Pm_scal <- extractT(Pm_scal, 2, 4)
#' 
#' # In this case, `extT_Pm_scal` is a LazyTensor encoding, symbolically,
#' # the following vector:
#' #       [,1] [,2] [,3] [,4]
#' # [1,]    0    0 3.14    0
#' 
#' 
#' # II - A more general example
#' # --------------------------- 
#' 
#' x <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows, 3 columns
#' x_i <- LazyTensor(x, index = 'i')   # LazyTensor from matrix x, 'i' indexed
#' 
#' m <- 2
#' d <- 7
#' 
#' extractT_x <- extractT(x_i, m, d)   # symbolic matrix
#' }
#' @export
extractT <- function(x, m, d) {
    # type checking
    if(!is.int(m)) 
        stop("`m` input argument should be an integer.")
    if(!is.int(d)) 
        stop("`d` input argument should be an integer.")
    # dim checking
    D <- x$dimres
    if(d < D) {
        stop(
            paste("Input `d` is out of bounds. Should be at least equal",
                  " to `x` inner dimension, which is ",
                  D, ".", sep = "")
        )
    }
    if(m < 0 || m >= d) {
        stop(paste("Index `m` is out of bounds. Should be in [0, ",
                   "`d`).", sep = ""))
    }
    if((m + D - 1) > d) {
        stop(
            paste("Slice dimension is out of bounds: `d` - `m` should be", 
                  " strictly greater than `x` inner dimension, which is ",
                  D, ".", sep = "")
        )
    }
    return(unaryop.LazyTensor(x, "ExtractT", opt_arg = m, opt_arg2 = d))
}


# Concatenation ----------------------------------------------------------------

#' Concatenation
#' @description
#' Concatenation of two `LazyTensor` or `ComplexLazyTensor`.
#' @details If `x` and `y` are two `LazyTensor` or `ComplexLazyTensor`,
#' `concat(x, y)` encodes, symbolically, the concatenation of `x` and `y` along
#' their inner dimension.
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param x a `LazyTensor` or a `ComplexLazyTensor`.
#' @param y a `LazyTensor` or a `ComplexLazyTensor`.
#' @return a `LazyTensor` or a `ComplexLazyTensor` that encodes, symbolically,
#' the concatenation of `x` and `y` along their inner dimension.
#' @examples
#' \dontrun{
#' x <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows, 3 columns
#' y <- matrix(runif(250 * 3), 250, 3) # arbitrary R matrix, 250 rows, 3 columns
#' x_i <- LazyTensor(x, index = 'i')   # LazyTensor from matrix x, indexed by 'i'
#' y_j <- LazyTensor(y, index = 'j')   # LazyTensor from matrix x, indexed by 'j'
#' 
#' concat_xy <- concat(x_i, y_j)
#' }
#' @export
concat <- function(x, y) {
    dim_res <- get_inner_dim(x) + get_inner_dim(y)
    return(binaryop.LazyTensor(x, y, "Concat",
                               dim_check_type = NA,
                               dim_res = dim_res))
}


# One-hot ----------------------------------------------------------------------

#' One-hot encoding vector
#' @description
#' Create one-hot encoding vector as a `LazyTensor` object.
#' @details If `x` is a scalar value encoded as a `LazyTensor`,
#' `one_hot(x, D)` encodes, symbolically, a vector of length **D**
#' where round(`x`)-th coordinate is equal to 1, and the other ones to 0.
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param x A `LazyTensor` encoding a scalar value.
#' @param D an `integer` corresponding to the output dimension.
#' @return a `LazyTensor`.
#' @examples
#' \dontrun{
#' A <- 7
#' A_LT <- LazyTensor(A) # LazyTensor from scalar A
#' D <- 7
#' 
#' onehot_x <- one_hot(A_LT, D) # symbolic vector of length D
#' }
#' @export
one_hot <- function(x, D) {
    if((!is.LazyTensor(x)) || (is.ComplexLazyTensor(x))) {
        stop(paste("`one_hot` operation can only be applied to ",
                   "`LazyTensor`, not `ComplexLazyTensor`.",
                   sep = ""))
    }
    
    if(x$dimres != 1 || !is.LazyParameter(x)) {
        stop("One-hot encoding is only supported for scalar formulas.")
    }
    
    if(!is.int(D)) {
        stop("`D` input argument should be an integer.")
    }
    
    return(unaryop.LazyTensor(x, "OneHot", opt_arg = D, dim_res = D))
}



# ELEMENTARY DOT PRODUCT =======================================================


# MatVecMult -------------------------------------------------------------------

#' Matrix-vector product
#' @description
#' Matrix-vector product for `LazyTensor` objects.
#' @details `matvecmult(m, v)` encodes, symbolically, the matrix-vector product 
#' of matrix `m` and vector `v`.
#' 
#' **Note**: `m` and `v` should have the same inner dimension or `v` should be 
#' of dimension 1.
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param m a `LazyTensor` or a `ComplexLazyTensor` encoding a matrix.
#' @param v a `LazyTensor` or a `ComplexLazyTensor` encoding a parameter vector.
#' @return a `LazyTensor` or a `ComplexLazyTensor`. 
#' @examples
#' \dontrun{
#' m <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows, 3 columns
#' v <- c(1,2,3)                       # arbitrary R vector of length 3
#' m_i <- LazyTensor(m, index = 'i')   # LazyTensor from matrix m, indexed by 'i'
#' Pm_v <- LazyTensor(v)               # parameter vector LazyTensor from v
#' 
#' mv_mult <- matvecmult(m_i, Pm_v)    # symbolic matrix
#' }
#' @export
matvecmult <- function(m, v) {
    if(!is.LazyMatrix(m)) {
        stop(paste("`m` input argument should be a `LazyTensor` encoding",
                   " a matrix defined with `Vi()` or `Vj()`.", sep = ""))
    }
    if(!is.LazyVector(v)) {
        stop(paste("`v` input argument should be a `LazyTensor` encoding",
                   " a vector defined with `Pm()`.", sep = ""))
    }
    
    if((m$dimres != v$dimres) && (v$dimres != 1)) {
        stop(paste("`m` and `v` should have the same inner dimension or",
                   " `v` should be of dimension 1.", sep = ""))
    }
    
    dim_res <- 1
    
    if(v$dimres == 1) {
        dim_res <- m$dimres
    }
    
    return(binaryop.LazyTensor(m, v, "MatVecMult", dim_res = dim_res))
}


# VecMatMult -------------------------------------------------------------------

#' Vector-matrix product
#' @description
#' Vector-matrix product for `LazyTensor` objects.
#' @details `vecmatmult(v, m)` encodes, symbolically, the vector-matrix product 
#' of vector `v` and matrix `m`.
#' 
#' **Note**: `v` and `m` should have the same inner dimension or `v` should be 
#' of dimension 1.
#' 
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param v a `LazyTensor` or a `ComplexLazyTensor` encoding a parameter vector.
#' @param m a `LazyTensor` or a `ComplexLazyTensor` encoding a matrix.
#' @return a `LazyTensor` or a `ComplexLazyTensor`.
#' @examples
#' \dontrun{
#' v <- c(1,2,3)                        # arbitrary R vector of length 3
#' m <- matrix(runif(150 * 3), 150, 3)  # arbitrary R matrix, 150 rows, 3 columns
#' Pm_v <- LazyTensor(v)                # parameter vector LazyTensor from v
#' m_i <- LazyTensor(m, index = 'i')    # LazyTensor from matrix m, indexed by 'i'
#' 
#' vm_mult <- vecmatmult(Pm_v, m_i)     # symbolic matrix
#' }
#' @export
vecmatmult <- function(v, m) {
    if(!is.LazyVector(v)) {
        stop(paste("`v` input argument should be a `LazyTensor` encoding",
                   " a vector defined with `Pm()`.", sep = ""))
    }
    if(!is.LazyMatrix(m)) {
        stop(paste("`m` input argument should be a `LazyTensor` encoding",
                   " a matrix defined with `Vi()` or `Vj()`.", sep = ""))
    }
    
    if((m$dimres != v$dimres) && (v$dimres != 1)) {
        stop(paste("`v` and `m` should have the same inner dimension or",
                   " `v` should be of dimension 1.", sep = ""))
    }
    
    dim_res <- 1
    
    if(v$dimres == 1) {
        dim_res <- m$dimres
    }
    
    return(binaryop.LazyTensor(v, m, "VecMatMult", dim_res = dim_res))
}


# Tensorprod -------------------------------------------------------------------

#' Tensor product
#' @description
#' Tensor product for `LazyTensor` objects.
#' @details If `x` and `y` are `LazyTensor` objects encoding matrices,
#' respectively of length `nx*px` and `ny*py`, then `tensorprod(x, y)` encodes,
#' symbolically, the tensor product between matrix `x` and `y`, which is
#' a symbolic matrix of dimension (`nx*ny`, `px*py`).
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param x a `LazyTensor` or a `ComplexLazyTensor`.
#' @param y a `LazyTensor` or a `ComplexLazyTensor`. 
#' @return a `LazyTensor` or a `ComplexLazyTensor`.
#' @examples
#' \dontrun{
#' x <- matrix(c(1, 2, 3), 2, 3)       # arbitrary R matrix, 150 rows, 3 columns
#' x_i <- LazyTensor(x, index = 'i')   # LazyTensor from matrix x, indexed by 'i'
#' y <- matrix(c(1, 1, 1), 2, 3)       # arbitrary R matrix, 200 rows, 3 columns
#' y_i <- LazyTensor(y, index = 'i')   # LazyTensor from matrix y, indexed by 'i'
#' 
#' tp_xy <- tensorprod(x_i, y_i) # symbolic (4, 9) matrix. 
#' 
#' }
#' @export
tensorprod <- function(x, y) {
    dim_res <- x$dimres * y$dimres
    return(binaryop.LazyTensor(x, y, "TensorProd", dim_check_type = NA, 
                               dim_res = dim_res))
}


# REDUCTIONS ===================================================================


# Reduction --------------------------------------------------------------------

#' Reduction operation
#' @description
#' Applies a reduction to a `LazyTensor`.
#' @details `reduction.LazyTensor(x, opstr, index)` will :
#' - if `index = "i"`, return the `opstr` reduction of `x` over the "i" indexes;
#' - if `index = "j"`, return the `opstr` reduction of `x` over the "j" indexes.
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param x a `LazyTensor` or a `ComplexLazyTensor`.
#' @param opstr a `string` formula (like "Sum" or "Max").
#' @param index a `character` corresponding to the wanted reduction dimension, 
#' either `"i"` or `"j"`, to specify whether if the reduction is done along 
#' the index `i` or `j`.
#' @param opt_arg an optional argument: an `integer` (e.g. for `Kmin` 
#' reduction), a `character`, a `LazyTensor` or a `ComplexLazyTensor`. `NULL` 
#' if not used (default).
#' @return an array storing the result of the specified reduction.
#' @examples
#' \dontrun{
#' x <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows, 3 columns
#' x_i <- LazyTensor(x, index = 'i')   # creating LazyTensor from matrix x, 
#'                                     # indexed by 'i'
#' 
#' red_x <- reduction.LazyTensor(x_i, "Sum", "i")
#' }
#' @export
reduction.LazyTensor <- function(x, opstr, index, opt_arg = NULL) {
    
    res <- NULL
    
    if(!is.LazyTensor(x) && !is.ComplexLazyTensor(x))
        stop("`x` input should be a LazyTensor or a ComplexLazyTensor.")
    
    if(!is.character(opstr))
        stop("`opstr` input should be a string text.")
    
    if(check_index(index)) {
        op <- preprocess_reduction(x, opstr, index, opt_arg)
        if(!is.null(opt_arg) && is.LazyTensor(opt_arg))
            res <- op(c(x$data, opt_arg$data))
        else {
            res <- op(x$data)
        }
    } else {
        stop("`index` input argument should be a character, either 'i' or 'j'.")
    }
    
    return(res)
}


# sum function -----------------------------------------------------------------

#' Sum operation
#' @inherit base::sum description
#' @inherit base::sum details
#' @inherit base::sum params
#' @inherit base::sum return
#' @inherit base::sum examples
#' @seealso [base::sum()]
#' @author R core team and contributors
#' @export
sum.default <- function(...) {
    return(base::sum(...))
}

#' Sum operation or sum reduction
#' @author Chloe Serre-Combe, Amelie Vernay
#' @description
#' Sum operation for vector/matrix/array or `LazyTensor`, or sum reduction for 
#' `LazyTensor`.
#' @details
#' If `x` is a `LazyTensor`, see [rkeops::sum.LazyTensor()], else see
#' [rkeops::sum.default()].
#' @param x input for [rkeops::sum.default()] or [rkeops::sum.LazyTensor()].
#' @param ... optional arguments for [rkeops::sum.default()] or 
#' [rkeops::sum.LazyTensor()].
#' @return Sum of input.
#' @examples
#' \dontrun{
#' # R base operation
#' sum(1:10)
#' sum(c(NA, 1, 2), na.rm = TRUE)
#' # LazyTensor operation
#' x <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows, 3 columns
#' x_i <- LazyTensor(x, index = 'i')   # creating LazyTensor from matrix x, 
#'                                     # indexed by 'i'
#' 
#' sum_x <- sum(x_i)                   # LazyTensor object
#' sum_red_x <- sum(x_i, "i")          # sum reduction indexed by 'i'
#' }
#' @export
sum <- function(x, ...) {
    UseMethod("sum")
}

#' Sum operation or sum reduction
#' @author Chloe Serre-Combe, Amelie Vernay
#' @description
#' Sum operation or sum reduction for `LazyTensor` objects.
#' @details
#' If `x` is a `LazyTensor`, `sum(x, index)` will :
#' - if `index = "i"`, return the sum reduction of `x` over the `i` index.
#' - if `index = "j"`, return the sum reduction of `x` over the `j` index.
#' - if `index = NA` (default), return a new `LazyTensor` object representing 
#' the symbolic sum of the values along the inner dimension of `x`.
#' 
#' **Note**: Run `browseVignettes("rkeops")` to access the vignettes and find 
#' details about this function in the *"RKeOps LazyTensor"* vignette, at 
#' section *"Reductions"*.
#' @param x a `LazyTensor` or a `ComplexLazyTensor`.
#' @param index a `character` corresponding to the wanted reduction dimension, 
#' either `"i"` or `"j"`, to specify whether if the reduction is done along 
#' the index `i` or `j`.
#' It can be `NA` (default) when no reduction is desired.
#' @param ... not used, only present for method compatibility with the
#' corresponding generic function.
#' @return a `LazyTensor` if `index = NA` or an array storing the result of 
#' the specified sum reduction otherwise.
#' @seealso [rkeops::sum_reduction()]
#' @examples
#' \dontrun{
#' x <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows, 3 columns
#' x_i <- LazyTensor(x, index = 'i')   # creating LazyTensor from matrix x, 
#'                                     # indexed by 'i'
#' 
#' sum_x <- sum(x_i)                   # LazyTensor object
#' sum_red_x <- sum(x_i, "i")          # sum reduction indexed by 'i'
#' }
#' 
#' @export
sum.LazyTensor <- function(x, index = NA, ...) {
    
    if(!is.na(index) && !check_index(index)) {
        stop(paste0("`index` input argument should be a character,",
                    " either 'i' or 'j', or NA."))
    }
    
    if(is.na(index)) {
        if(is.ComplexLazyTensor(x)) {
            return(unaryop.LazyTensor(x, "ComplexSum", dim_res = 2))
        }
        else {
            return(unaryop.LazyTensor(x, "Sum", dim_res = 1))
        }
    }
    else
        return(reduction.LazyTensor(x, "Sum", index))
}


# sum reduction ----------------------------------------------------------------

#' Sum reduction
#' @author Chloe Serre-Combe, Amelie Vernay
#' @name sum.LazyTensor
#' @aliases sum_reduction
#' @details **Note**: `index` input argument cannot be `NA` for the 
#' `sum_reduction()` function.
#' @examples
#' \dontrun{
#' x <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows, 3 columns
#' x_i <- LazyTensor(x, index = 'i')   # creating LazyTensor from matrix x, 
#'                                     # indexed by 'i'
#' 
#' sum_reduction(x_i, "i")
#' }
#' @export
sum_reduction <- function(x, index) {
    if(check_index(index)) {
        return(reduction.LazyTensor(x, "Sum", index))
    }
    else {
        stop(paste("`index` input argument should be a character,",
                   " either 'i' or 'j', or NA.", sep = ""))
    }
}


# min function -----------------------------------------------------------------

#' Minimum operation
#' @inherit base::min description
#' @inherit base::min details
#' @inherit base::min params
#' @inherit base::min return
#' @inherit base::min examples
#' @seealso [base::min()]
#' @author R core team and contributors
#' @export
min.default <- function(...) {
    return(base::min(...))
}

#' Minimum operation or minimum reduction
#' @author Chloe Serre-Combe, Amelie Vernay
#' @description
#' Minimum operation for vector/matrix/array or `LazyTensor`, or minimum 
#' reduction for `LazyTensor`.
#' @details
#' If `x` is a `LazyTensor`, see [rkeops::min.LazyTensor()], else see
#' [rkeops::min.default()].
#' @param x input for [rkeops::min.default()] or [rkeops::min.LazyTensor()].
#' @param ... optional arguments for [rkeops::min.default()] or 
#' [rkeops::min.LazyTensor()].
#' @return Minimum of input.
#' @examples
#' \dontrun{
#' # R base operation
#' min(1:10)
#' min(c(NA, 1, 2), na.rm = TRUE)
#' # LazyTensor operation
#' x <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows, 3 columns
#' x_i <- LazyTensor(x, index = 'i')   # creating LazyTensor from matrix x, 
#'                                     # indexed by 'i'
#' 
#' min_x <- min(x_i)                   # LazyTensor object
#' min_red_x <- min(x_i, "i")          # min reduction indexed by 'i'
#' }
#' @export
min <- function(x, ...) {
    UseMethod("min")
}

#' Minimum operation or minimum reduction
#' @author Chloe Serre-Combe, Amelie Vernay
#' @description
#' Minimum operation or minimum reduction for `LazyTensor` objects.
#' @details
#' If `x` is a `LazyTensor`, `min(x, index)` will :
#' - if `index = "i"`, return the min reduction of `x` over the `i` index.
#' - if `index = "j"`, return the min reduction of `x` over the `j` index.
#' - if `index = NA` (default), return a new `LazyTensor` object 
#'   representing the symbolic min of the values along the inner dimension of 
#'   `x`.
#' 
#' **Note**: Run `browseVignettes("rkeops")` to access the vignettes and find 
#' details about this function in the *"RKeOps LazyTensor"* vignette, at 
#' section *"Reductions"*.
#' @param x a `LazyTensor` or a `ComplexLazyTensor`.
#' @param index a `character` corresponding to the wanted reduction dimension, 
#' either `"i"` or `"j"`, to specify whether if the min reduction is done along 
#' the index `i` or `j`.
#' It can be `NA` (default) when no reduction is desired.
#' @param ... not used, only present for method compatibility with the
#' corresponding generic function.
#' @return a `LazyTensor` if `index = NA` or an array storing the result of 
#' the specified min reduction otherwise.
#' @examples
#' \dontrun{
#' x <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows, 3 columns
#' x_i <- LazyTensor(x, index = 'i')   # creating LazyTensor from matrix x, 
#'                                     # indexed by 'i'
#' 
#' min_x <- min(x_i)                   # LazyTensor object
#' min_red_x <- min(x_i, "i")          # min reduction indexed by 'i'
#' }
#' @export
min.LazyTensor <- function(x, index = NA, ...) {
    if(is.na(index))
        return(unaryop.LazyTensor(x, "Min"))
    else if(check_index(index))
        return(reduction.LazyTensor(x, "Min", index))
    else 
        stop(paste("`index` input argument should be a character,",
                   " either 'i' or 'j', or NA.", sep = ""))
}


# min reduction ----------------------------------------------------------------

#' Minimum reduction
#' @author Chloe Serre-Combe, Amelie Vernay
#' @name min.LazyTensor
#' @aliases min_reduction
#' @details **Note**: `index` input argument cannot be `NA` for the 
#' `min_reduction()` function.
#' @examples
#' \dontrun{
#' x <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows, 3 columns
#' x_i <- LazyTensor(x, index = 'i')   # creating LazyTensor from matrix x, 
#'                                     # indexed by 'i'
#' 
#' min_reduction(x_i, "i")
#' }
#' @export
min_reduction <- function(x, index) {
    if(check_index(index)) {
        return(reduction.LazyTensor(x, "Min", index))
    }
    else {
        stop("`index` input argument should be a character, either 'i' or 'j'.")
    }
}


# argmin function --------------------------------------------------------------

#' ArgMin operation or ArgMin reduction
#' @author Chloe Serre-Combe, Amelie Vernay
#' @description
#' ArgMin operation or ArgMin reduction for `LazyTensor` objects.
#' @details
#' If `x` is a `LazyTensor`, `argmin(x, index)` will :
#' - if `index = "i"`, return the argmin reduction of `x` over the `i` 
#'   indexes.
#' - if `index = "j"`, return the argmin reduction of `x` over the `j` 
#'   indexes.
#' - if `index = NA` (default), return a new `LazyTensor` object 
#'   representing the symbolic argmin of the values along the inner dimension 
#'   of `x`.
#' 
#' **Note**: Run `browseVignettes("rkeops")` to access the vignettes and find 
#' details about this function in the *"RKeOps LazyTensor"* vignette, at 
#' section *"Reductions"*.
#' @param x a `LazyTensor` or a `ComplexLazyTensor`.
#' @param index a `character` corresponding to the wanted reduction dimension, 
#' either `"i"` or `"j"`, to specify whether if the argmin reduction is done 
#' along the index `i` or `j`.
#' It can be `NA` (default) when no reduction is desired.
#' @return a `LazyTensor` if `index = NA` or an array storing the result of 
#' the specified argmin reduction otherwise.
#' @seealso [rkeops::argmin_reduction()]
#' @examples
#' \dontrun{
#' x <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows, 3 columns
#' x_i <- LazyTensor(x, index = 'i')   # creating LazyTensor from matrix x, 
#'                                     # indexed by 'i'
#' 
#' argmin_xi <- argmin(x_i, "i")  # argmin reduction indexed by 'i'
#' argmin_x <- argmin(x_i)        # symbolic matrix
#' }
#' @export
argmin <- function(x, index = NA) {
    if(is.na(index))
        return(unaryop.LazyTensor(x, "ArgMin"))
    else if(check_index(index))
        return(reduction.LazyTensor(x, "ArgMin", index))
    else
        stop(paste("`index` input argument should be a character,",
                   " either 'i' or 'j', or NA.", sep = ""))
}


# argmin reduction -------------------------------------------------------------

#' ArgMin reduction
#' @author Chloe Serre-Combe, Amelie Vernay
#' @name argmin
#' @aliases argmin_reduction
#' @details **Note**: `index`  input argument cannot be `NA` for the 
#' `argmin_reduction()` function.
#' @examples
#' \dontrun{
#' x <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows, 3 columns
#' x_i <- LazyTensor(x, index = 'i')   # creating LazyTensor from matrix x, 
#'                                     # indexed by 'i'
#' 
#' argmin_red <- argmin(x_i, "i")      # argmin reduction indexed by 'i'
#' }
#' @export
argmin_reduction <- function(x, index) {
    if(check_index(index))
        return(reduction.LazyTensor(x, "ArgMin", index))
    else
        stop("`index` input argument should be a character, either 'i' or 'j'.")
}


# min_argmin -------------------------------------------------------------------

#' Min-ArgMin reduction
#' @author Chloe Serre-Combe, Amelie Vernay
#' @name min_argmin
#' @description
#' Min-ArgMin reduction for `LazyTensor` objects.
#' @details
#' If `x` is a `LazyTensor`, `min_argmin(x, index)` will :
#' - if `index = "i"`, return the Min-ArgMin reduction of `x`, i.e. the minimal 
#'   values and its position, over the `i` indexes.
#' - if `index = "j"`, return the Min-ArgMin reduction of `x`, i.e. the minimal 
#'   values and its position, over the `j` indexes.
#' 
#' **Note**: Run `browseVignettes("rkeops")` to access the vignettes and find 
#' details about this function in the *"RKeOps LazyTensor"* vignette, at 
#' section *"Reductions"*.
#' @param x a `LazyTensor` or a `ComplexLazyTensor`.
#' @param index a `character` corresponding to the wanted reduction dimension, 
#' either `"i"` or `"j"`, to specify whether if the Min-ArgMin reduction is done 
#' along the index `i` or `j`.
#' @return an array storing the result of the specified Min-ArgMin 
#' reduction otherwise.
#' @examples
#' \dontrun{
#' x <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows, 3 columns
#' x_i <- LazyTensor(x, index = 'i')   # creating LazyTensor from matrix x, 
#'                                     # indexed by 'i'
#' 
#' min_argmin_xi <- min_argmin(x_i, "i")  # min argmin reduction indexed by 'i'
#' }
#' @export
min_argmin <- function(x, index) {
    if(check_index(index))
        return(reduction.LazyTensor(x, "Min_ArgMin", index))
    else
        stop("`index` input argument should be a character, either 'i' or 'j'.")
}


# min_argmin reduction ---------------------------------------------------------

#' Min-ArgMin reduction
#' @author Chloe Serre-Combe, Amelie Vernay
#' @name min_argmin
#' @aliases min_argmin_reduction
#' @examples
#' \dontrun{
#' x <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows, 3 columns
#' x_i <- LazyTensor(x, index = 'i')   # creating LazyTensor from matrix x, 
#'                                     # indexed by 'i'
#' 
#' min_argmin_red <- min_argmin_reduction(x_i, "i") # min reduction indexed by 'i'
#' }
#' @export
min_argmin_reduction <- function(x, index) {
    if(check_index(index))
        return(min_argmin(x, index))
    else
        stop("`index` input argument should be a character, either 'i' or 'j'.")
}


# max function -----------------------------------------------------------------

#' Maximum operation
#' @inherit base::max description
#' @inherit base::max details
#' @inherit base::max params
#' @inherit base::max return
#' @inherit base::max examples
#' @seealso [base::max()]
#' @author R core team and contributors
#' @export
max.default <- function(...) {
    return(base::max(...))
}

#' Maximum operation or maximum reduction
#' @author Chloe Serre-Combe, Amelie Vernay
#' @description
#' Maximum operation for vector/matrix/array or `LazyTensor`, or maximum 
#'  reduction for `LazyTensor`.
#' @details
#' If `x` is a `LazyTensor`, see [rkeops::max.LazyTensor()], else see
#' [rkeops::max.default()].
#' @param x input for [rkeops::max.default()] or [rkeops::max.LazyTensor()].
#' @param ... optional arguments for [rkeops::max.default()] or 
#' [rkeops::max.LazyTensor()].
#' @return Maximum of input.
#' @examples
#' \dontrun{
#' # R base operation
#' max(1:10)
#' max(c(NA, 1, 2), na.rm = TRUE)
#' # LazyTensor operation
#' x <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows, 3 columns
#' x_i <- LazyTensor(x, index = 'i')   # creating LazyTensor from matrix x, 
#'                                     # indexed by 'i'
#' 
#' max_x <- max(x_i)                   # LazyTensor object
#' max_red_x <- max(x_i, "i")          # max reduction indexed by 'i'
#' }
#' @export
max <- function(x, ...) {
    UseMethod("max")
}

#' Maximum operation or maximum reduction
#' @author Chloe Serre-Combe, Amelie Vernay
#' @description
#' Maximum operation or maximum reduction for `LazyTensor` objects.
#' @details
#' If `x` is a `LazyTensor`, `max(x, index)` will :
#' - if `index = "i"`, return the max reduction of `x` over the `i` 
#'   indexes.
#' - if `index = "j"`, return the max reduction of `x` over the `j` 
#'   indexes.
#' - if `index = NA` (default), return a new `LazyTensor` object 
#'   representing the symbolic max of the values along the inner dimension of 
#'   `x`.
#' 
#' **Notes**: If `index = NA`, `x` input argument should be a `LazyTensor` 
#' encoding a parameter vector.
#' 
#' **Note**: Run `browseVignettes("rkeops")` to access the vignettes and find 
#' details about this function in the *"RKeOps LazyTensor"* vignette, at 
#' section *"Reductions"*.
#' @param x a `LazyTensor` or a `ComplexLazyTensor`.
#' @param index a `character` corresponding to the wanted reduction dimension, 
#' either `"i"` or `"j"`, to specify whether if the max reduction is done along 
#' the index `i` or `j`.
#' It can be `NA` (default) when no reduction is desired.
#' @param ... not used, only present for method compatibility with the
#' corresponding generic function.
#' @return a `LazyTensor` if `index = NA` or an array storing the result of 
#' the specified max reduction otherwise.
#' @seealso [rkeops::max_reduction()]
#' @examples
#' \dontrun{
#' x <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows, 3 columns
#' x_i <- LazyTensor(x, index = 'i')   # creating LazyTensor from matrix x, 
#'                                     # indexed by 'i'
#' 
#' max_x <- max(x_i)                   # LazyTensor object
#' max_red_x <- max(x_i, "i")          # max reduction indexed by 'i'
#' }
#' @export
max.LazyTensor <- function(x, index = NA, ...) {
    if(is.na(index))
        return(unaryop.LazyTensor(x, "Max"))
    else if(check_index(index))
        return(reduction.LazyTensor(x, "Max", index))
    else 
        stop(paste("`index` input argument should be a character,",
                   " either 'i' or 'j', or NA.", sep = ""))
}


# max reduction ----------------------------------------------------------------

#' Maximum reduction
#' @author Chloe Serre-Combe, Amelie Vernay
#' @name max.LazyTensor
#' @aliases max_reduction
#' @details **Note**: `index` input argument cannot be `NA` for the 
#' `max_reduction()` function.
#' @examples
#' \dontrun{
#' x <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows, 3 columns
#' x_i <- LazyTensor(x, index = 'i')   # creating LazyTensor from matrix x, 
#'                                     # indexed by 'i'
#' 
#' max_reduction(x_i, "i")
#' }
#' @export
max_reduction <- function(x, index) {
    if(check_index(index))
        return(reduction.LazyTensor(x, "Max", index))
    else 
        stop("`index` input argument should be a character, either 'i' or 'j'.")
}


# argmax function --------------------------------------------------------------

#' ArgMax operation or ArgMax reduction
#' @author Chloe Serre-Combe, Amelie Vernay
#' @description
#' ArgMax operation or ArgMax reduction for `LazyTensor` objects.
#' @details
#' If `x` is a `LazyTensor`, `argmax(x, index)` will :
#' - if `index = "i"`, return the argmax reduction of `x` over the `i` 
#'   indexes.
#' - if `index = "j"`, return the argmax reduction of `x` over the `j` 
#'   indexes.
#' - if `index = NA` (default), return a new `LazyTensor` object 
#'   representing the symbolic argmax of the values along the inner dimension 
#'   of `x`.
#' 
#' **Note**: Run `browseVignettes("rkeops")` to access the vignettes and find 
#' details about this function in the *"RKeOps LazyTensor"* vignette, at 
#' section *"Reductions"*.
#' @param x a `LazyTensor` or a `ComplexLazyTensor`.
#' @param index a `character` corresponding to the wanted reduction dimension, 
#' either `"i"` or `"j"`, to specify whether if the argmax reduction is done 
#' along the index `i` or `j`.
#' It can be `NA` (default) when no reduction is desired.
#' @return a `LazyTensor` if `index = NA` or an array storing the result of 
#' the specified argmax reduction otherwise.
#' @seealso [rkeops::argmax_reduction()]
#' @examples
#' \dontrun{
#' x <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows, 3 columns
#' x_i <- LazyTensor(x, index = 'i')   # creating LazyTensor from matrix x, 
#'                                     # indexed by 'i'
#' 
#' argmax_xi <- argmax(x_i, "i")  # argmax reduction indexed by 'i'
#' argmax_x <- argmax(x_i)        # symbolic matrix
#' }
#' @export
argmax <- function(x, index = NA) {
    if(is.na(index)) {
        return(unaryop.LazyTensor(x, "ArgMax"))
    }
    else if(check_index(index))
        return(reduction.LazyTensor(x, "ArgMax", index))
    else 
        stop(paste("`index` input argument should be a character,",
                   " either 'i' or 'j', or NA.", sep = ""))
}


# argmax reduction -------------------------------------------------------------

#' ArgMax reduction
#' @author Chloe Serre-Combe, Amelie Vernay
#' @name argmax
#' @aliases argmax_reduction
#' @details **Note**: `index` input argument cannot be `NA` for the 
#' `argmax_reduction()` function.
#' @examples
#' \dontrun{
#' x <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows, 3 columns
#' x_i <- LazyTensor(x, index = 'i')   # creating LazyTensor from matrix x, 
#'                                     # indexed by 'i'
#' 
#' argmax_red <- argmax_reduction(x_i, "i")  # argmax reduction indexed by 'i'
#' }
#' @export
argmax_reduction <- function(x, index) {
    if(check_index(index))
        return(reduction.LazyTensor(x, "ArgMax", index))
    else 
        stop("`index` input argument should be a character, either 'i' or 'j'.")
}


# max_argmax -------------------------------------------------------------------

#' Max-ArgMax reduction
#' @author Chloe Serre-Combe, Amelie Vernay
#' @name max_argmax
#' @description
#' Max-ArgMax reduction for `LazyTensor` objects.
#' @details
#' If `x` is a `LazyTensor`, `max_argmax(x, index)` will :
#' - if `index = "i"`, return the Max-ArgMax reduction of `x` over the `i` 
#'   indexes.
#' - if `index = "j"`, return the Max-ArgMax reduction of `x` over the `j` 
#'   indexes.
#' 
#' **Note**: Run `browseVignettes("rkeops")` to access the vignettes and find 
#' details about this function in the *"RKeOps LazyTensor"* vignette, at 
#' section *"Reductions"*.
#' @param x a `LazyTensor` or a `ComplexLazyTensor`.
#' @param index a `character` corresponding to the wanted reduction dimension, 
#' either `"i"` or `"j"`, to specify whether if the Max-ArgMax reduction is done 
#' along the index `i` or `j`.
#' @return an array storing the result of the specified Max-ArgMax 
#' reduction otherwise.
#' @examples
#' \dontrun{
#' x <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows, 3 columns
#' x_i <- LazyTensor(x, index = 'i')   # creating LazyTensor from matrix x, 
#'                                     # indexed by 'i'
#' 
#' max_argmax_x <- max_argmax(x_i, "i")  # max argmax reduction indexed by 'i'
#' }
#' @export
max_argmax <- function(x, index) {
    if(check_index(index))
        return(reduction.LazyTensor(x, "Max_ArgMax", index))
    else 
        stop("`index` input argument should be a character, either 'i' or 'j'.")
}


# max_argmax reduction ---------------------------------------------------------

#' Max-ArgMax reduction
#' @author Chloe Serre-Combe, Amelie Vernay
#' @name max_argmax
#' @aliases max_argmax_reduction
#' @examples
#' \dontrun{
#' x <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows, 3 columns
#' x_i <- LazyTensor(x, index = 'i')   # creating LazyTensor from matrix x, 
#'                                     # indexed by 'i'
#' 
#' max_argmax_red <- max_argmax_reduction(x_i, "i")  # max argmax reduction 
#'                                                   # indexed by 'i'
#' }
#' @export
max_argmax_reduction <- function(x, index) {
    if(check_index(index))
        return(max_argmax(x, index))
    else 
        stop("`index` input argument should be a character, either 'i' or 'j'.")
}


# Kmin -------------------------------------------------------------------------

#' K-Min reduction
#' @name Kmin
#' @author Chloe Serre-Combe, Amelie Vernay
#' @description
#' K-Min reduction for `LazyTensor` objects.
#' @details If `x` is a `LazyTensor` or a `ComplexLazyTensor`, 
#' `Kmin(x, K, index)` will:
#' - if `index = "i"`, return the `K` minimal values of `x` over the `i` 
#'   indexes.
#' - if `index = "j"`, return the `K` minimal values of `x` over the `j` 
#'   indexes.
#' 
#' **Note**: Run `browseVignettes("rkeops")` to access the vignettes and find 
#' details about this function in the *"RKeOps LazyTensor"* vignette, at 
#' section *"Reductions"*.
#' @param x a `LazyTensor` or a `ComplexLazyTensor`.
#' @param index a `character` corresponding to the reduction dimension that 
#' should be either `"i"` or `"j"` to specify whether if the reduction is 
#' indexed by `"i"` or `"j"`.
#' @param K An `integer` corresponding to the  number of minimal values 
#' required.
#' @return A matrix corresponding to the Kmin reduction.
#' @examples
#' \dontrun{
#' x <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows, 3 columns
#' x_i <- LazyTensor(x, index = 'i')   # LazyTensor from matrix x, indexed by 'i'
#' K <- 2
#' kmin_x <- Kmin(x_i, K, "i")         # Kmin reduction, over the 'i' indices
#' }
#' @export
Kmin <- function(x, K, index) {
    if(!is.int(K))
        stop("`K` input argument should be an integer.")
    if(!check_index(index))
        stop("`index` input argument should be a character, either 'i' or 'j'.")
    return(reduction.LazyTensor(x, "KMin", index, opt_arg = K))
}


# Kmin reduction ---------------------------------------------------------------

#' Kmin reduction
#' @author Chloe Serre-Combe, Amelie Vernay
#' @name Kmin
#' @aliases Kmin_reduction
#' @examples
#' \dontrun{
#' x <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows, 3 columns
#' x_i <- LazyTensor(x, index = 'i')   # LazyTensor from matrix x, indexed by 'i'
#' K <- 2
#' kmin_red_x <- Kmin_reduction(x_i, K, "i")   # Kmin reduction, indexed by 'i'
#' }
#' @export
Kmin_reduction <- function(x, K, index) {
    return(Kmin(x, K, index))
}


# argKmin ----------------------------------------------------------------------


#' Arg-K-min reduction
#' @author Chloe Serre-Combe, Amelie Vernay
#' @name argKmin
#' @description
#' Arg-K-min reduction for `LazyTensor` objects.
#' @details If `x` is a `LazyTensor` or a `ComplexLazyTensor`,
#' `argKmin(x, K, index)` will:
#' - if `index = "i"`, return the indices of the `K` minimal values of `x` 
#'   over the `i` indexes.
#' - if `index = "j"`, return the indices of the `K` minimal values of `x`
#'   over the `j` indexes.
#' 
#' **Note**: Run `browseVignettes("rkeops")` to access the vignettes and find 
#' details about this function in the *"RKeOps LazyTensor"* vignette, at 
#' section *"Reductions"*.
#' @param x a `LazyTensor` or a `ComplexLazyTensor`.
#' @param index a `character` corresponding to the reduction dimension that 
#' should be either `"i"` or `"j"` to specify whether if the reduction is 
#' indexed by `"i"` or `"j"`.
#' @param K An `integer` corresponding to the  number of minimal values 
#' required.
#' @return A matrix corresponding to the argKmin reduction.
#' @examples
#' \dontrun{
#' x <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows, 3 columns
#' x_i <- LazyTensor(x, index = 'i')   # creating LazyTensor from matrix x, 
#'                                     # indexed by 'i'
#' K <- 2
#' argkmin_x <- argKmin(x_i, K, "i")   # argKmin reduction 
#'                                     # indexed by 'i'
#' 
#' }
#' @export
argKmin <- function(x, K, index) {
    if(!is.int(K))
        stop("`K` input argument should be an integer.")
    if(!check_index(index))
        stop("`index` input argument should be a character, either 'i' or 'j'.")
    return(reduction.LazyTensor(x, "ArgKMin", index, opt_arg = K))
}


# argKmin reduction ------------------------------------------------------------

#' Arg-K-min reduction
#' @author Chloe Serre-Combe, Amelie Vernay
#' @name argKmin
#' @aliases argKmin_reduction
#' @examples
#' \dontrun{
#' x <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows, 3 columns
#' x_i <- LazyTensor(x, index = 'i')   # creating LazyTensor from matrix x, 
#'                                     # indexed by 'i'
#' K <- 2
#' argkmin_red_x <- argKmin_reduction(x_i, K, "i")  # argKmin reduction 
#'                                                  # indexed by 'i'
#' }
#' @export
argKmin_reduction <- function(x, K, index) {
    return(argKmin(x, K, index))
}


# Kmin-argKmin -----------------------------------------------------------------

#' K-Min-arg-K-min reduction
#' @description
#' K-Min-arg-K-min reduction for `LazyTensor` objects.
#' @details If `x` is a `LazyTensor` or a `ComplexLazyTensor`,
#' `Kmin_argKmin(x, K, index)` will:
#' - if `index = i`, return the `K` minimal values of `x` (i.e. Kmin) and 
#'   the corresponding indexes over  the `i` indices.
#' - if `index = j`, return the `K` minimal values of `x` and its indices 
#'   over the `j` indices (columns).
#' 
#' **Note**: Run `browseVignettes("rkeops")` to access the vignettes and find 
#' details about this function in the *"RKeOps LazyTensor"* vignette, at 
#' section *"Reductions"*.
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param x a `LazyTensor` or a `ComplexLazyTensor`.
#' @param index A `character` corresponding to the reduction dimension that 
#' should be either `i` or `j` to specify whether if the reduction is 
#' indexed by `i` (rows) or `j` (columns).
#' @param K An `integer` corresponding to the  number of minimal values
#' required.
#' @return A matrix corresponding to the Kmin-argKmin reduction.
#' @examples
#' \dontrun{
#' x <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows, 3 columns
#' x_i <- LazyTensor(x, index = 'i')   # creating LazyTensor from matrix x, 
#'                                     # indexed by 'i'
#' K <- 2
#' k_argk_x <- Kmin_argKmin(x_i, K, "i")  # Kmin-argKmin reduction 
#'                                        # indexed by 'i'
#' 
#' }
#' @export
Kmin_argKmin <- function(x, K, index) {
    if(!is.int(K))
        stop("`K` input argument should be an integer.")
    if(!check_index(index))
        stop("`index` input argument should be a character, either 'i' or 'j'.")
    return(reduction.LazyTensor(x, "KMin_ArgKMin", index, opt_arg = K))
}


# Kmin-argKmin reduction -------------------------------------------------------

#' Kmin-argKmin reduction.
#' @author Chloe Serre-Combe, Amelie Vernay
#' @name Kmin_argKmin
#' @aliases Kmin_argKmin_reduction
#' @examples
#' \dontrun{
#' x <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows, 3 columns
#' x_i <- LazyTensor(x, index = 'i')   # LazyTensor from matrix x, 
#'                                     # indexed by 'i'
#' 
#' K <- 2
#' k_argk_x <- Kmin_argKmin_reduction(x_i, K, "i")  # Kmin-argKmin reduction 
#'                                                  # over the "i" indices
#' }
#' @export
Kmin_argKmin_reduction <- function(x, K, index) {
    return(Kmin_argKmin(x, K, index))
}


# LogSumExp  -------------------------------------------------------------------

#' Log-Sum-Exp reduction.
#' @description
#' Log-Sum-Exp reduction for `LazyTensor` objects.
#' @details If `x` is a `LazyTensor` or a `ComplexLazyTensor`, 
#' `logsumexp(x, index, weight)` will:
#' - if `index = "i"`, return the Log-Sum-Exp reduction of `x` 
#'   over the `i` indexes;
#' - if `index = "j"`, return the Log-Sum-Exp reduction of `x` 
#'   over the `j` indexes.
#' 
#' **Note**: Run `browseVignettes("rkeops")` to access the vignettes and find 
#' details about this function in the *"RKeOps LazyTensor"* vignette, at 
#' section *"Reductions"*.
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param x a `LazyTensor` or a `ComplexLazyTensor`.
#' @param index a `character` corresponding to the reduction dimension that 
#' should be either `"i"` or `"j"` to specify whether if the reduction is 
#' indexed by `"i"` or `"j"`.
#' @param weight an optional object (`LazyTensor` or `ComplexLazyTensor`) that 
#' specifies scalar or vector-valued weights. `NULL` by default and not used.
#' @return a matrix corresponding to the Log-Sum-Exp reduction.
#' @examples
#' \dontrun{
#' x <- matrix(runif(150 * 3), 150, 3) 
#' x_i <- LazyTensor(x, index = 'i') 
#' y <- matrix(runif(100 * 3), 100, 3)
#' y_j <- LazyTensor(y, index = 'j')
#' w <- matrix(runif(100 * 3), 100, 3) # weight LazyTensor
#' w_j <- LazyTensor(w, index = 'j')
#' 
#' S_ij = sum((x_i - y_j)^2)                                           
#' logsumexp_xw <- logsumexp(S_ij, 'i', w_j) # logsumexp reduction 
#'                                           # over the 'i' indices
#'                                          
#' logsumexp_x <- logsumexp(S_ij, 'i')      # logsumexp reduction without
#'                                          # weight over the 'i' indices
#' }
#' @export
logsumexp <- function(x, index, weight = NULL) {
    if(check_index(index) && is.null(weight)) {
        #res <- reduction.LazyTensor(x, "Max_SumShiftExp", index)
        return(reduction.LazyTensor(x, "LogSumExp", index))
    }
    else if(check_index(index) && !is.null(weight)) {
        #res <- reduction.LazyTensor(x, "Max_SumShiftExpWeight", 
        #                           index, opt_arg = weight)
        #res <- reduction.LazyTensor(x, "LogSumExp", 
        #                           index, opt_arg = weight)
        stop(paste("`logsumexp` reduction is not yet supported with weights.",
                   "\nThis should be fixed in a future release.", sep = ""))
    } else {
        stop("`index` input argument should be a character, either 'i' or 'j'.")
    }
}


# LogSumExp reduction  ---------------------------------------------------------

#' Log-Sum-Exp reduction.
#' @author Chloe Serre-Combe, Amelie Vernay
#' @name logsumexp
#' @aliases logsumexp_reduction
#' @examples
#' \dontrun{
#' x <- matrix(runif(150 * 3), 150, 3) 
#' x_i <- LazyTensor(x, index = 'i') 
#' y <- matrix(runif(100 * 3), 100, 3)
#' y_j <- LazyTensor(y, index = 'j')
#' w <- matrix(runif(150 * 3), 150, 3) # weight LazyTensor
#' w_j <- LazyTensor(y, index = 'j')
#' 
#' S_ij = sum( (x_i - y_j)^2 )                                           
#' logsumexp_xw <- logsumexp_reduction(S_ij, 'i', w_j) # logsumexp reduction 
#'                                                     # over the 'i' indices
#'                                          
#' logsumexp_x <- logsumexp_reduction(S_ij, 'i')  # logsumexp reduction without
#'                                                # weight over the 'i' indices
#' }
#' @export
logsumexp_reduction <- function(x, index, weight = NULL) {
    return(logsumexp(x, index, weight))
}


# SumSoftMaxWeight -------------------------------------------------------------

#' Sum of weighted Soft-Max reduction.
#' @details If `x` is a `LazyTensor` or a `ComplexLazyTensor`, 
#' `sumsoftmaxweight(x, index, weight)` will:
#' - if `index = "i"`, return the Sum of weighted Soft-Max reduction of `x` 
#'   over the `i` indexes;
#' - if `index = "j"`, return the Sum of weighted Soft-Max reduction of `x` 
#'   over the `j` indexes.
#' 
#' **Note**: Run `browseVignettes("rkeops")` to access the vignettes and find 
#' details about this function in the *"RKeOps LazyTensor"* vignette, at 
#' section *"Reductions"*.
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param x a `LazyTensor` or a `ComplexLazyTensor`.
#' @param index a `character` corresponding to the reduction dimension that 
#' should be either `"i"` or `"j"` to specify whether if the reduction is 
#' indexed by `"i"` or `"j"`.
#' @param weight an optional object (`LazyTensor` or `ComplexLazyTensor`) that 
#' specifies scalar or vector-valued weights.
#' @return a matrix corresponding to the Sum of weighted Soft-Max reduction.
#' @examples
#' \dontrun{
#' x <- matrix(runif(150 * 3), 150, 3) 
#' x_i <- LazyTensor(x, index = 'i') 
#' y <- matrix(runif(100 * 3), 100, 3)
#' y_j <- LazyTensor(y, index = 'j')
#' 
#' V_ij <- x_i - y_j   # weight matrix
#' S_ij = sum(V_ij^2)     
#' 
#' ssmaxweight <- sumsoftmaxweight(S_ij, 'i', V_ij) # sumsoftmaxweight reduction
#'                                                  # over the 'i' indices
#' }
#' @export
sumsoftmaxweight <- function(x, index, weight) {
    tmp_weight <- weight #concat(1, weight)
    if(check_index(index)) {
        return(reduction.LazyTensor(x, "SumSoftMaxWeight", 
                                    index, opt_arg = tmp_weight))
    } else {
        stop("`index` input argument should be a character, either 'i' or 'j'.")
    }
}


# SumSoftMaxWeight Reduction ---------------------------------------------------

#' Sum of weighted Soft-Max reduction.
#' @author Chloe Serre-Combe, Amelie Vernay
#' @name sumsoftmaxweight
#' @aliases sumsoftmaxweight_reduction
#' @examples
#' \dontrun{
#' x <- matrix(runif(150 * 3), 150, 3) 
#' x_i <- LazyTensor(x, index = 'i') 
#' y <- matrix(runif(100 * 3), 100, 3)
#' y_j <- LazyTensor(y, index = 'j')
#' 
#' V_ij <- x_i - y_j   # weight matrix
#' S_ij = sum(V-ij^2)     
#' 
#' # sumsoftmaxweight reduction over the 'i' indices
#' ssmaxw_red <- sumsoftmaxweight_reduction(S_ij, 'i', V_ij) 
#' 
#' }
#' @export
sumsoftmaxweight_reduction <- function(x, index, weight) {
    return(sumsoftmaxweight(x, index, weight))
}


# GRADIENT =====================================================================

# Gradient ---------------------------------------------------------------------

#' Gradient operation.
#' @description
#' Gradient operation.
#' @details `grad(f, gradin, opstr, var, index)` returns a `matrix` which 
#' corresponds to the gradient (more precisely, the adjoint of the differential 
#' operator) of `f` with respect to the variable `var` and applied to `gradin` 
#' with compiling the corresponding reduction operator of `opstr`.
#' It has an additional integer input parameter `index` indicating if the inner 
#' dimension corresponds to columns, i.e. `index = 'j'` or rows, i.e. 
#' `index = 'i'`.
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param f A `LazyTensor` or a `ComplexLazyTensor`.
#' @param gradin a `LazyTensor`, a `ComplexLazyTensor` encoding a matrix of ones
#' with an inner dimension equal to 1 and indexed by the 
#' same index and with number of rows equal to 
#' the number of rows of the first `x` variable (in `x$data`).
#' @param opstr A `string` formula corresponding to a reduction 
#' (like "Sum" or "Max").
#' @param var An `integer` number indicating regarding to which 
#' variable/parameter (given by name or by position index starting at 0) the 
#' gradient of the formula should be computed or a one of the `LazyTensor` 
#' contained in `f`.
#' @param index A `character` that should be either **i** or **j** to specify 
#' whether if the reduction is indexed by **i** (rows), or **j** (columns). 
#' When the first `f` variable is indexed by **i** (resp. **j**), index cannot 
#' be **i** (resp. **j**). 
#' @return A `matrix`.
#' @examples
#' \dontrun{
#' nx <- 100
#' ny <- 150
#' x <- matrix(runif(nx*3), nrow=nx, ncol=3)     # matrix 100 x 3
#' y <- matrix(runif(ny*3), nrow=ny, ncol=3)     # matrix 150 x 3
#' eta <- matrix(runif(nx*1), nrow=nx, ncol=1)   # matrix 100 x 1 
#' # nrow(x) == nrow(eta)
#' 
#' x_i <- LazyTensor(x, index = 'i')  # LazyTensor from matrix x, indexed by 'i'
#' y_j <- LazyTensor(y, index = 'j')  # LazyTensor from matrix y, indexed by 'j'
#' eta_i <- LazyTensor(eta, index = 'i')   # LazyTensor from matrix eta, 
#'                                         # indexed by 'i' (like x_i)
#' 
#' # gradient with the formula from position
#' grad_xy <- grad(sqnorm2(x_i-y_j), eta_i, "Sum", var = y_j$formula, "j")  
#' 
#' # gradient with the formula from index
#' grad_xy <- grad(sqnorm2(x_i-y_j), eta_i, "Sum", var = 0, "j")     
#' }
#' @export
grad <- function(f, gradin, opstr, var, index) {
    if(is.LazyTensor(var)) {
        var <- fix_variables(var)$formula
    }
    
    if(!is.LazyTensor(f) || !is.LazyTensor(gradin)) {
        stop(paste0("`f` and `gradin` input arguments should be LazyTensor."))
    }
    
    if(!is.character(opstr)) {
        stop(
            paste0("`opstr` input should be a string text corresponding to", 
                   " a reduction formula.")
        )
    }
    
    first_f_index_i <- grep("A0x.*i", f$args[[1]])
    first_f_index_j <- grep("A0x.*j", f$args[[1]])
    gradin_index_i <- grep("A0x.*i", gradin$args[[1]])
    gradin_index_j <- grep("A0x.*j", gradin$args[[1]])
    
    # errors when `f` first argument and `gradin` are not indexed in 
    # the same way
    if(any(first_f_index_i) && any(gradin_index_j)) {
        stop(
            paste0("`gradin` input argument should be indexed by 'i'.")
        )
    }
    
    if(any(first_f_index_j) && any(gradin_index_i)) {
        stop(
            paste0("`gradin` input argument should be indexed by 'j'.")
        )
    }
    
    # errors to avoid "R aborting session" when the first argument of `f` is
    # indexed by `index`. To change in the future ?
    if(any(first_f_index_i) && index == "i") {
        stop(
            paste0("`index` input argument should be 'j'.")
        )
    }
    
    if(any(first_f_index_j) && index == "j") {
        stop(
            paste0("`index` input argument should be 'i'.")
        )
    }
    
    # Verification for gradin shape
    if(gradin$dimres != 1 || (nrow(gradin$data[[1]]) != nrow(f$data[[1]]))) {
        stop(paste0("`gradin` input argument should be a LazyTensor encoding", 
                    " a matrix of shape (", nrow(f$data[[1]]), ",1)."))
    }
    
    if(!check_index(index)) {
        stop(paste0("`index` input argument should be a character,",
                    " either 'i' or 'j'."))
    }
    
    op <- preprocess_reduction(f, opstr, index)
    
    grad_op <- keops_grad(op, var)
    res <- grad_op(c(f$data, gradin$data))
    return(res)
}
