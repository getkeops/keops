library(rkeops)
library(stringr)
library(data.table)

set_rkeops_option("tagCpuGpu", 0)
set_rkeops_option("precision", "double")



# OPERATIONS ===================================================================


# addition ---------------------------------------------------------------------
"+.default" <- .Primitive("+") # assign default as current definition

#' Addition.
#' @description
#' Symbolic binary operation for addition.
#' @usage x + y
#' @details If `x` or `y` is a `LazyTensor`, `x + y` returns a `LazyTensor`
#' that encodes, symbolically, the addition of `x` and `y`.
#' (In case one of the arguments is a vector or a scalar, it is first converted to `LazyTensor`).
#' If none of the arguments is a `LazyTensor`, is equivalent to the "+" R operator.
#' 
#' **Note**
#' 
#' `x` and `y` input arguments should have the same inner dimension or be of dimension 1.
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param x A `LazyTensor`, a `ComplexLazyTensor`, a vector of numeric values, or a scalar value.
#' @param y A `LazyTensor`, a `ComplexLazyTensor`, a vector of numeric values, or a scalar value.
#' @return An object of class "LazyTensor" if the function is called with a `LazyTensor`,
#' and an object of class "numeric", otherwise.
#' @examples
#' \dontrun{
#' x <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows and 3 columns
#' y <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows and 3 columns
#' x_i <- LazyTensor(x, index = 'i')   # creating LazyTensor from matrix x, indexed by 'i'
#' y_j <- LazyTensor(x, index = 'j')   # creating LazyTensor from matrix x, indexed by 'j'
#' Sum_xy <- x_i + y_j                 # symbolic matrix, 150 rows and 3 columns
#' }
#' @export
"+" <- function(x, y) { 
    if(!is.LazyTensor(x) && !is.ComplexLazyTensor(x))
        UseMethod("+", y)
    else
        UseMethod("+", x)
}

"+.LazyTensor" <- function(x, y) {
    res <- binaryop.LazyTensor(x, y, "+", is_operator = TRUE, dim_check_type = "sameor1")
    return(res)
}

"+.ComplexLazyTensor" <- function(x, y) {
    res <- binaryop.LazyTensor(x, y, "+", is_operator = TRUE, dim_check_type = "sameor1")
    return(res)
}


# subtraction  ----------------------------------------------------------------
"-.default" <- .Primitive("-") # assign default as current definition

#' Subtraction or minus sign.
#' @description
#' Symbolic binary operation for subtraction.
#' @usage x - y
#' @details Two possible use cases:
#' \itemize{
#'     \item{**Subtraction**:}{ If `x` or `y` is a `LazyTensor`, `x - y` returns a `LazyTensor`
#'     that encodes, symbolically, the subtraction of `x` and `y`.
#'     (In case one of the arguments is a vector or a scalar, it is first converted to `LazyTensor`).
#'     If none of the arguments is a `LazyTensor`, is equivalent to the "-" R operator.}
#'     \item{**Minus sign**:}{ If `x` is a `LazyTensor`, `-x` returns a `LazyTensor`
#'     that encodes, symbolically, the element-wise opposite of `x`.}
#' }
#' **Note**
#' 
#' For **subtraction operation**, `x` and `y` input arguments should have the same inner dimension or be of dimension 1.
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param x A `LazyTensor`, a `ComplexLazyTensor`, a vector of numeric values, or a scalar value (matrices can be used with the minus sign only).
#' @param y A `LazyTensor`, a `ComplexLazyTensor`, a vector of numeric values, or a scalar value (matrices can be used with the minus sign only).
#' @return An object of class "LazyTensor" if the function is called with a `LazyTensor`,
#' and an object of class "numeric", otherwise.
#' @examples
#' \dontrun{
#' x <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows and 3 columns
#' y <- matrix(runif(250 * 3), 250, 3) # arbitrary R matrix, 250 rows and 3 columns
#' x_i <- LazyTensor(x, index = 'i')   # creating LazyTensor from matrix x, indexed by 'i'
#' y_j <- LazyTensor(x, index = 'j')   # creating LazyTensor from matrix x, indexed by 'j'
#' Sub_xy <- x_i - y_j                 # symbolic matrix
#' Minus_x <- -x_i                     # symbolic matrix
#' }
#' @export
"-" <- function(x, y = NA) { 
    if(!is.LazyTensor(x) && !is.ComplexLazyTensor(x))
        UseMethod("-", y)
    else
        UseMethod("-", x)
}

"-.LazyTensor" <- function(x, y = NA) {
    if((length(y) == 1) && is.na(y))
        res <- unaryop.LazyTensor(x, "Minus")
    else
        res <- binaryop.LazyTensor(x, y, "-", is_operator = TRUE, dim_check_type = "sameor1")
    return(res)
}

"-.ComplexLazyTensor" <- function(x, y) {
    res <- binaryop.LazyTensor(x, y, "-", is_operator = TRUE, dim_check_type = "sameor1")
    return(res)
}


# multiplication  --------------------------------------------------------------
"*.default" <- .Primitive("*") # assign default as current definition

#' Multiplication.
#' @description
#' Symbolic binary operation for multiplication.
#' @usage x * y
#' @details If `x` or `y` is a `LazyTensor`, `x * y` returns a `LazyTensor`
#' that encodes, symbolically, the element-wise product of `x` and `y`.
#' (In case one of the arguments is a vector or a scalar, it is first converted to `LazyTensor`).
#' If none of the arguments is a `LazyTensor`, is equivalent to the "*" R operator.
#' 
#' **Note**
#' 
#' `x` and `y` input arguments should have the same inner dimension or be of dimension 1.
#' @author Chloé Serre-Combe, Amélie Vernay
#' @param x A `LazyTensor`, a `ComplexLazyTensor`, a vector of numeric values, or a scalar value.
#' @param y A `LazyTensor`, a `ComplexLazyTensor`, a vector of numeric values, or a scalar value.
#' @return An object of class "LazyTensor" if the function is called with a `LazyTensor`,
#' and an object of class "numeric", otherwise.
#' @examples
#' \dontrun{
#' x <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows and 3 columns
#' y <- matrix(runif(250 * 3), 250, 3) # arbitrary R matrix, 250 rows and 3 columns
#' x_i <- LazyTensor(x, index = 'i')   # creating LazyTensor from matrix x, indexed by 'i'
#' y_j <- LazyTensor(x, index = 'j')   # creating LazyTensor from matrix x, indexed by 'j'
#' x_times_y <- x_i * y_j              # symbolic matrix
#' }
#' @export
"*" <- function(x, y) { 
    if(!is.LazyTensor(x) && !is.ComplexLazyTensor(x))
        UseMethod("*", y)
    else
        UseMethod("*", x)
}

"*.LazyTensor" <- function(x, y) {
    res <- binaryop.LazyTensor(x, y, "*", is_operator = TRUE, dim_check_type = "sameor1")
    return(res)
}

"*.ComplexLazyTensor" <- function(x, y) {
    res <- binaryop.LazyTensor(x, y, "*", is_operator = TRUE, dim_check_type = "sameor1")
    return(res)
}


# division ---------------------------------------------------------------------
"/.default" <- .Primitive("/")

#' Division.
#' @description
#' Symbolic binary operation for division.
#' @usage x / y
#' @details If `x` or `y` is a `LazyTensor`, `x / y` returns a `LazyTensor`
#' that encodes, symbolically, the element-wise division of `x` by `y`.
#' (In case one of the arguments is a vector or a scalar, it is first converted to `LazyTensor`).
#' If none of the arguments is a `LazyTensor`, is equivalent to the "/" R operator.
#' 
#' **Note**
#' 
#' `x` and `y` input arguments should have the same inner dimension or be of dimension 1.
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param x A `LazyTensor`, a `ComplexLazyTensor`, a vector of numeric values, or a scalar value.
#' @param y A `LazyTensor`, a `ComplexLazyTensor`, a vector of numeric values, or a scalar value.
#' @return An object of class "LazyTensor" if the function is called with a `LazyTensor`,
#' and an object of class "numeric", otherwise.
#' @examples
#' \dontrun{
#' x <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows and 3 columns
#' y <- matrix(runif(250 * 3), 250, 3) # arbitrary R matrix, 250 rows and 3 columns
#' x_i <- LazyTensor(x, index = 'i')   # creating LazyTensor from matrix x, indexed by 'i'
#' y_j <- LazyTensor(x, index = 'j')   # creating LazyTensor from matrix x, indexed by 'j'
#' x_div_y <- x_i / y_j                # symbolic matrix
#' }
#' @export
"/" <- function(x, y) { 
    if(!is.LazyTensor(x) && !is.ComplexLazyTensor(x))
        UseMethod("/", y)
    else
        UseMethod("/", x)
}

"/.LazyTensor" <- function(x, y) {
    res <- binaryop.LazyTensor(x, y, "/", is_operator = TRUE, dim_check_type = "sameor1")
    return(res)
}

"/.ComplexLazyTensor" <- function(x, y) {
    res <- binaryop.LazyTensor(x, y, "/", is_operator = TRUE, dim_check_type = "sameor1")
    return(res)
}


# square -----------------------------------------------------------------------
square.default <- function(x) {
    res <- x^2
    return(res)
}

#' Element-wise square.
#' @description
#' Symbolic unary operation for element-wise square.
#' @details If `x` is a `LazyTensor`, `square(x)` returns a `LazyTensor` that encodes, symbolically,
#' the element-wise square of `x` ; else, is equivalent to the "^2" R operator.
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param x A `LazyTensor`, a `ComplexLazyTensor`, a vector or a matrix of numeric values, or a scalar value.
#' @return An object of class "LazyTensor" if the function is called with a `LazyTensor`,
#' and an object of class "numeric", "matrix", or "array" otherwise, same as the input class.
#' @examples
#' \dontrun{
#' x <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows and 3 columns
#' x_i <- LazyTensor(x, index = 'i')   # creating LazyTensor from matrix x, indexed by 'i'
#' Square_x <- square(x_i)             # symbolic matrix, 150 rows and 3 columns
#' }
#' @export
square <- function(x) {
    UseMethod("square", x)
}

square.LazyTensor <- function(x) {
    res <- unaryop.LazyTensor(x, "Square")
    return(res)
}

square.ComplexLazyTensor <- function(x) {
    res <- unaryop.LazyTensor(x, "Square", res_type = "ComplexLazyTensor")
    return(res)
}


# square root ------------------------------------------------------------------
sqrt.default <- .Primitive("sqrt") # assign default as current definition

#' Element-wise square root.
#' @description
#' Symbolic unary operation for element-wise square root.
#' @details If `x` is a `LazyTensor`, `sqrt(x)` returns a `LazyTensor` that encodes, symbolically,
#' the element-wise square root of `x` ; else, computes R default square root function.
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param x A `LazyTensor`, a `ComplexLazyTensor`, a vector or a matrix of numeric values, or a scalar value.
#' @return An object of class "LazyTensor" if the function is called with a `LazyTensor`,
#' and an object of class "numeric", "matrix", or "array" otherwise, depending on the input class
#' (see R default `sqrt()` function).
#' @examples
#' \dontrun{
#' x <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows and 3 columns
#' x_i <- LazyTensor(x, index = 'i')   # creating LazyTensor from matrix x, indexed by 'i'
#' Sqrt_x <- sqrt(x_i)                 # symbolic matrix, 150 rows and 3 columns
#' }
#' @export
sqrt <- function(x) { 
    UseMethod("sqrt", x)
}

sqrt.LazyTensor <- function(x) {
    res <- unaryop.LazyTensor(x, "Sqrt")
    return(res)
}


sqrt.ComplexLazyTensor <- function(x) {
    res <- unaryop.LazyTensor(x, "Sqrt", res_type = "ComplexLazyTensor")
    return(res)
}

# Rsqrt ------------------------------------------------------------------------
rsqrt.default <- function(x) {
    res <- 1 / sqrt(x)
    return(res)
}

#' Element-wise inverse square root.
#' @description
#' Symbolic unary operation for element-wise inverse square root.
#' @details If `x` is a `LazyTensor`, `sqrt(x)` returns a `LazyTensor` that encodes, symbolically,
#' the element-wise inverse square root of `x` ; else, computes the element-wise inverse
#' of R default square root function.
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param x A `LazyTensor`, a `ComplexLazyTensor`, a vector or a matrix of numeric values, or a scalar value.
#' @return An object of class "LazyTensor" if the function is called with a `LazyTensor`,
#' and an object of class "numeric", "matrix", or "array" otherwise, same as the input class.
#' @examples
#' \dontrun{
#' x <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows and 3 columns
#' x_i <- LazyTensor(x, index = 'i')   # creating LazyTensor from matrix x, indexed by 'i'
#' Rsqrt_x <- rsqrt(x_i)               # symbolic matrix, 150 rows and 3 columns
#' }
#' @export
rsqrt <- function(x) {
    UseMethod("rsqrt", x)
}

rsqrt.LazyTensor <- function(x) {
    res <- unaryop.LazyTensor(x, "Rsqrt")
    return(res)
}

rsqrt.ComplexLazyTensor <- function(x) {
    res <- unaryop.LazyTensor(x, "Rsqrt", res_type = "ComplexLazyTensor")
    return(res)
}


# power ------------------------------------------------------------------------
"^.default" <- .Primitive("^") # assign default as current definition

#' Power.
#' @description
#' Symbolic binary operation for element-wise power operator.
#' @usage x^y
#' @details If `x` or `y` is a `LazyTensor`, `x^y` returns a `LazyTensor`
#' that encodes, symbolically, the element-wise value of `x` to the power of `y`.
#' (In case one of the arguments is a vector or a scalar, it is first converted to `LazyTensor`).
#' If none of the arguments is a `LazyTensor`, is equivalent to the "^" R operator.
#' 
#' **Note**
#' 
#' \itemize{
#'     \item{if **y = 2**,}{ `x^y` relies on the `"Square"` KeOps operation;}
#'     \item{if **y = 0.5**,}{ `x^y` uses on the `"Sqrt"` KeOps operation;}
#'     \item{if **y = -0.5**,}{ `x^y` uses on the `"Rsqrt"` KeOps operation.}
#' }
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param x A `LazyTensor`, a `ComplexLazyTensor`, a vector of numeric values, or a scalar value.
#' @param y A `LazyTensor`, a `ComplexLazyTensor`, a vector of numeric values, or a scalar value.
#' @return An object of class "LazyTensor" if the function is called with a `LazyTensor`,
#' and an object of class "numeric", otherwise.
#' @examples
#' \dontrun{
#' x <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows and 3 columns
#' y <- matrix(runif(250 * 3), 250, 3) # arbitrary R matrix, 250 rows and 3 columns
#' x_i <- LazyTensor(x, index = 'i')   # creating LazyTensor from matrix x, indexed by 'i'
#' y_j <- LazyTensor(x, index = 'j')   # creating LazyTensor from matrix x, indexed by 'j'
#' x_pow_y <- x_i^y_j                  # symbolic matrix
#' }
#' @export
"^" <- function(x, y) { 
    if(!is.LazyTensor(x) && !is.ComplexLazyTensor(x))
        UseMethod("^", y)
    else
        UseMethod("^", x)
}

"^.LazyTensor" <- function(x, y) {   
    if(is.numeric(y) && length(y) == 1){
        if((as.integer(y) - y) == 0){
            if(y == 2)
                res <- unaryop.LazyTensor(x, "Square")
            else
                res <- unaryop.LazyTensor(x, "Pow", y)
        }
        else if(y == 0.5)
            res <- unaryop.LazyTensor(x, "Sqrt") # element-wise square root
        else if(y == (-0.5))
            res <- unaryop.LazyTensor(x, "Rsqrt") # element-wise inverse square root
        # check if Powf with y a float number has to be like Powf(var1,var2) or Powf(var,y) (Powf(var, 0.5))
        else {
            # TODO !!
            ## We convert before it goes into binaryop because `get_inner_dim()` requires LazyTensors
            #if(is.numeric(x))
            #    x <- LazyTensor(x)
            #if(is.numeric(y))
            #    y <- LazyTensor(y)
            #
            #x_inner_dim <- get_inner_dim(x)
            #y_inner_dim <- get_inner_dim(y)
            #if() {
            #    # x and y should have the same inner dimension or y should have its inner dimension equal to 1
            #    
            #}
            #res <- binaryop.LazyTensor(x, y, "Powf", dim_check_type = NA) # power operation
            res <- binaryop.LazyTensor(x, y, "Powf") # power operation
        }
    }
    else
        res <- binaryop.LazyTensor(x, y, "Powf") # power operation
    return(res)
}


"^.ComplexLazyTensor" <- function(x, y) {   
    if(is.numeric(y) && length(y) == 1){
        if((as.integer(y) - y) == 0){
            if(y == 2)
                res <- unaryop.LazyTensor(x, "Square", res_type = "ComplexLazyTensor")
            else
                res <- unaryop.LazyTensor(x, "Pow", y, res_type = "ComplexLazyTensor")
        }
        else if(y == 0.5)
            res <- unaryop.LazyTensor(x, "Sqrt", res_type = "ComplexLazyTensor") # element-wise square root
        else if(y == (-0.5))
            res <- unaryop.LazyTensor(x, "Rsqrt", res_type = "ComplexLazyTensor") # element-wise inverse square root
        # check if Powf with y a float number has to be like Powf(var1,var2) or Powf(var,y) (Powf(var, 0.5))
        else {
            # TODO !!
            ## We convert before it goes into binaryop because `get_inner_dim()` requires LazyTensors
            #if(is.numeric(x))
            #    x <- LazyTensor(x)
            #if(is.numeric(y))
            #    y <- LazyTensor(y)
            #
            #x_inner_dim <- get_inner_dim(x)
            #y_inner_dim <- get_inner_dim(y)
            #if() {
            #    # x and y should have the same inner dimension or y should have its inner dimension equal to 1
            #    
            #}
            #res <- binaryop.LazyTensor(x, y, "Powf", dim_check_type = NA, res_type = "ComplexLazyTensor") # power operation
            res <- binaryop.LazyTensor(x, y, "Powf") # power operation
        }
    }
    else
        res <- binaryop.LazyTensor(x, y, "Powf") # power operation
    return(res)
}


# Euclidean scalar product -----------------------------------------------------
"|.default" <- .Primitive("|")

# TODO finish the doc with dimensions
#' Euclidean scalar product.
#' @description
#' Symbolic binary operation for Euclidean scalar product.
#' @usage x | y  or  (x | y)
#' @details If `x` or `y` is a `LazyTensor`, `(x|y)` (or `x | y`) returns a `LazyTensor`
#' that encodes, symbolically, the Euclidean scalar product between `x` and `y`, which must have the same shape.
#' (In case one of the arguments is a vector or a scalar, it is first converted to `LazyTensor`).
#' If none of the arguments is a `LazyTensor`, is equivalent to the "|" R operator.
#'
#' **Note**
#'
#' `x` and `y` input arguments should have the same inner dimension.
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param x A `LazyTensor`, a `ComplexLazyTensor`, a vector of numeric values, or a scalar value.
#' @param y A `LazyTensor`, a `ComplexLazyTensor`, a vector of numeric values, or a scalar value.
#' @return An object of class "LazyTensor" if the function is called with a `LazyTensor`,
#' and an object of class "numeric", otherwise.
#' @examples
#' \dontrun{
#' x <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows and 3 columns
#' y <- matrix(runif(250 * 3), 250, 3) # arbitrary R matrix, 250 rows and 3 columns
#' x_i <- LazyTensor(x, index = 'i')   # creating LazyTensor from matrix x, indexed by 'i'
#' y_j <- LazyTensor(x, index = 'j')   # creating LazyTensor from matrix x, indexed by 'j'
#' x_div_y <- x_i / y_j                # symbolic matrix
#' }
#' @export
"|" <- function(x, y) { 
    if(!is.LazyTensor(x) && !is.ComplexLazyTensor(x))
        UseMethod("|", y)
    else
        UseMethod("|", x)
}

"|.LazyTensor" <- function(x, y) {
    res <- binaryop.LazyTensor(x, y, "|", is_operator = TRUE, dim_check_type = "same")
    res$formula <- paste("(", res$formula, ")", sep = "")
    return(res)
}

"|.ComplexLazyTensor" <- function(x, y) {
    res <- binaryop.LazyTensor(x, y, "|", is_operator = TRUE, dim_check_type = "same")
    res$formula <- paste("(", res$formula, ")", sep = "")
    return(res)
}




# Matrix product ---------------------------------------------------------------

"%*%.default" <- .Primitive("%*%") # assign default as current definition

#' Matrix multiplication.
#' @description
#' Symbolic binary operation for element-wise matrix multiplication operator.
#' @usage x %*% y
#' @details If `x` or `y` is a `LazyTensor`, `x %*% y` .... TODO (sum_reduction)
#' If none of the arguments is a `LazyTensor`, is equivalent to the "%*%" R operator.
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param x A `LazyTensor`, a `ComplexLazyTensor`, a vector or a matrix of numeric values, or a scalar value.
#' @param y A `LazyTensor`, a `ComplexLazyTensor`, a vector or a matrix of numeric values, or a scalar value.
#' @return A matrix
#' @examples
#' \dontrun{
#' x <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows and 3 columns
#' y <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows and 3 columns
#' x_i <- LazyTensor(x, index = 'i')   # creating LazyTensor from matrix x, indexed by 'i'
#' y_j <- LazyTensor(x, index = 'j')   # creating LazyTensor from matrix x, indexed by 'j'
#' x_mult_y <- x_i %*% y_j             
#' }
#' @export
"%*%" <- function(x, y) { 
    if(!is.LazyTensor(x) && !is.ComplexLazyTensor(x))
        UseMethod("%*%", y)
    else
        UseMethod("%*%", x)
}

"%*%.LazyTensor" <- function(x, y) {
    if(is.matrix(y))
        y <- LazyTensor(y, "j")
    sum(x * y, index = "j")
}

"%*%.ComplexLazyTensor" <- function(x, y) {
    if(is.matrix(y))
        y <- LazyTensor(y, "j")
    sum(x * y, index = "j")
}


# exponential ------------------------------------------------------------------
exp.default <- .Primitive("exp")

#' Element-wise exponential.
#' @description
#' Symbolic unary operation for element-wise exponential.
#' @details 
#' 
#' **Different use cases**:
#' 
#' \itemize{
#'     \item{`x` is a `LazyTensor`,}{ `exp(x)` returns a `LazyTensor` that encodes, symbolically,
#'     the element-wise exponential of `x`;}
#'     \item{`x` is a `ComplexLazyTensor`,}{ `exp(x)` returns a `ComplexLazyTensor` that encodes, symbolically,
#'     the element-wise complex exponential of `x`;}
#'     \item{else,}{ `exp(x)` applies R default exponential to `x`.}
#' }
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param x A `LazyTensor`, a `ComplexLazyTensor`, a vector of numeric values, or a scalar value.
#' @return An object of class "LazyTensor" if the function is called with a `LazyTensor`,
#' and an object of class "numeric", "matrix", or "array" otherwise, depending on the input class
#' (see R default `exp()` function).
#' @examples
#' \dontrun{
#' # basic example
#' x <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows and 3 columns
#' x_i <- LazyTensor(x, index = 'i')   # creating LazyTensor from matrix x, indexed by 'i'
#' Exp_x <- exp(x_i)                   # symbolic matrix, 150 rows and 3 columns
#' 
#' # basic example with complex exponential
#' z <- matrix(1i^ (-6:5), nrow = 4)                     # create a complex 4x3 matrix
#' z_i <- LazyTensor(z, index = 'i', is_complex = TRUE)  # create a ComplexLazyTensor
#' Exp_z_i <- exp(z_i)                                   # symbolic matrix
#' }
#' @export
exp <- function(x) {
    UseMethod("exp")
}

exp.LazyTensor <- function(x) {
    res <- unaryop.LazyTensor(x, "Exp")
    return(res)
}

exp.ComplexLazyTensor <- function(x) {
    res <- unaryop.LazyTensor(x, "ComplexExp", res_type = "ComplexLazyTensor")
}


# logarithm --------------------------------------------------------------------
log.default <- .Primitive("log")

#' Element-wise logarithm.
#' @description
#' Symbolic unary operation for element-wise logarithm.
#' @details If `x` is a `LazyTensor`, `exp(x)` returns a `LazyTensor` that encodes, symbolically,
#' the element-wise exponential of `x` ; else, computes R default logarithm.
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param x A `LazyTensor`, a `ComplexLazyTensor`, a vector of numeric values, or a scalar value.
#' @return An object of class "LazyTensor" if the function is called with a `LazyTensor`,
#' and an object of class "numeric", "matrix", or "array" otherwise, depending on the input class
#' (see R default `log()` function).
#' @examples
#' \dontrun{
#' x <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows and 3 columns
#' x_i <- LazyTensor(x, index = 'i')   # creating LazyTensor from matrix x, indexed by 'i'
#' Log_x <- log(x_i)                   # symbolic matrix, 150 rows and 3 columns
#' }
#' @export
log <- function(x) {
    UseMethod("log")
}

log.LazyTensor <- function(x) {
    res <- unaryop.LazyTensor(x, "Log")
    return(res)
}

log.ComplexLazyTensor <- function(x) {
    res <- unaryop.LazyTensor(x, "Log", res_type = "ComplexLazyTensor")
    return(res)
}


# inverse ----------------------------------------------------------------------
inv.default <- function(x) {
    res <- 1 / x
    return(res)
}

#' Element-wise 1/x inverse.
#' @description
#' Symbolic unary operation for element-wise inverse.
#' @details If `x` is a `LazyTensor`, `exp(x)` returns a `LazyTensor` that encodes, symbolically,
#' the element-wise inverse of `x` ; else, computes R default inverse.
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param x A `LazyTensor`, a `ComplexLazyTensor`, a vector of numeric values, or a scalar value.
#' @return An object of class "LazyTensor" if the function is called with a `LazyTensor`,
#' and an object of class "numeric", "matrix", or "array" otherwise, same as the input class.
#' @examples
#' \dontrun{
#' x <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows and 3 columns
#' x_i <- LazyTensor(x, index = 'i')   # creating LazyTensor from matrix x, indexed by 'i'
#' Inv_x <- inv(x_i)                   # symbolic matrix, 150 rows and 3 columns
#' }
#' @export
inv <- function(x) {
    UseMethod("inv")
}

inv.LazyTensor <- function(x) {
    res <- unaryop.LazyTensor(x, "Inv")
    return(res)
}

inv.ComplexLazyTensor <- function(x) {
    res <- unaryop.LazyTensor(x, "Inv", res_type = "ComplexLazyTensor")
    return(res)
}


# cosine -----------------------------------------------------------------------
cos.default <- .Primitive("cos")

#' Element-wise cosine.
#' @description
#' Symbolic unary operation for element-wise cosine.
#' @details If `x` is a `LazyTensor`, `exp(x)` returns a `LazyTensor` that encodes, symbolically,
#' the element-wise cosine of `x` ; else, computes R default cosine.
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param x A `LazyTensor`, a `ComplexLazyTensor`, a vector of numeric values, or a scalar value.
#' @return An object of class "LazyTensor" if the function is called with a `LazyTensor`,
#' and an object of class "numeric", "matrix", or "array" otherwise, depending on the input class
#' (see R default `cos()` function).
#' @examples
#' \dontrun{
#' x <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows and 3 columns
#' x_i <- LazyTensor(x, index = 'i')   # creating LazyTensor from matrix x, indexed by 'i'
#' Cos_x <- cos(x_i)                   # symbolic matrix, 150 rows and 3 columns
#' }
#' @export
cos <- function(x) {
    UseMethod("cos")
}

cos.LazyTensor <- function(x) {
    res <- unaryop.LazyTensor(x, "Cos")
    return(res)
}

cos.ComplexLazyTensor <- function(x) {
    res <- unaryop.LazyTensor(x, "Cos", res_type = "ComplexLazyTensor")
    return(res)
}


# sine -------------------------------------------------------------------------
sin.default <- .Primitive("sin")

#' Element-wise sine.
#' @description
#' Symbolic unary operation for element-wise sine.
#' @details If `x` is a `LazyTensor`, `sin(x)` returns a `LazyTensor` that encodes, symbolically,
#' the element-wise sine of `x`; else, computes R default sine function.
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param x A `LazyTensor`, a `ComplexLazyTensor`, a vector or a matrix of numeric values, or a scalar value.
#' @return An object of class "LazyTensor" if the function is called with a `LazyTensor`,
#' and an object of class "numeric", "matrix", or "array" otherwise, depending on the input class
#' (see R default `sin()` function).
#' @examples
#' \dontrun{
#' x <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows and 3 columns
#' x_i <- LazyTensor(x, index = 'i')   # creating LazyTensor from matrix x, indexed by 'i'
#' Sin_x <- sin(x_i)                   # symbolic matrix, 150 rows and 3 columns
#' }
#' @export
sin <- function(x) {
    UseMethod("sin")
}

sin.LazyTensor  <- function(x){
    res <- unaryop.LazyTensor(x, "Sin")
    return(res)
}

sin.ComplexLazyTensor  <- function(x){
    res <- unaryop.LazyTensor(x, "Sin", res_type = "ComplexLazyTensor")
    return(res)
}


# arccosine --------------------------------------------------------------------
acos.default <- .Primitive("acos")

#' Element-wise arccosine.
#' @description
#' Symbolic unary operation for element-wise arccosine.
#' @details If `x` is a `LazyTensor`, `acos(x)` returns a `LazyTensor` that encodes, symbolically,
#' the element-wise arccosine of `x` ; else, computes R default arccosine function.
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param x A `LazyTensor`, a `ComplexLazyTensor`, a vector or a matrix of numeric values, or a scalar value.
#' @return An object of class "LazyTensor" if the function is called with a `LazyTensor`,
#' and an object of class "numeric", "matrix", or "array" otherwise, depending on the input class
#' (see R default `acos()` function).
#' @examples
#' \dontrun{
#' x <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows and 3 columns
#' x_i <- LazyTensor(x, index = 'i')   # creating LazyTensor from matrix x, indexed by 'i'
#' Acos_x <- acos(x_i)                 # symbolic matrix, 150 rows and 3 columns
#' }
#' @export
acos <- function(x) {
    UseMethod("acos")
}

acos.LazyTensor <- function(x) {
    res <- unaryop.LazyTensor(x, "Acos")
    return(res)
}

acos.ComplexLazyTensor <- function(x) {
    res <- unaryop.LazyTensor(x, "Acos", res_type = "ComplexLazyTensor")
    return(res)
}


# arcsine ----------------------------------------------------------------------
asin.default <- .Primitive("asin")

#' Element-wise arcsine.
#' @description
#' Symbolic unary operation for element-wise arcsine.
#' @details If `x` is a `LazyTensor`, `asin(x)` returns a `LazyTensor` that encodes, symbolically,
#' the element-wise arcsine of `x` ; else, computes R default arcsine function.
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param x A `LazyTensor`, a `ComplexLazyTensor`, a vector or a matrix of numeric values, or a scalar value.
#' @return An object of class "LazyTensor" if the function is called with a `LazyTensor`,
#' and an object of class "numeric", "matrix", or "array" otherwise, depending on the input class
#' (see R default `asin()` function).
#' @examples
#' \dontrun{
#' x <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows and 3 columns
#' x_i <- LazyTensor(x, index = 'i')   # creating LazyTensor from matrix x, indexed by 'i'
#' Asin_x <- asin(x_i)                  # symbolic matrix, 150 rows and 3 columns
#' }
#' @export
asin <- function(x) {
    UseMethod("asin")
}

asin.LazyTensor <- function(x) {
    res <- unaryop.LazyTensor(x, "Asin")
    return(res)
}

asin.ComplexLazyTensor <- function(x) {
    res <- unaryop.LazyTensor(x, "Asin", res_type = "ComplexLazyTensor")
    return(res)
}


# arctangent -------------------------------------------------------------------
atan.default <- .Primitive("atan")

#' Element-wise arctangent.
#' @description
#' Symbolic unary operation for element-wise arctangent.
#' @details If `x` is a `LazyTensor`, `atan(x)` returns a `LazyTensor` that encodes, symbolically,
#' the element-wise arctangent of `x` ; else, computes R default arctangent function.
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param x A `LazyTensor`, a `ComplexLazyTensor`, a vector or a matrix of numeric values, or a scalar value.
#' @return An object of class "LazyTensor" if the function is called with a `LazyTensor`,
#' and an object of class "numeric", "matrix", or "array" otherwise, depending on the input class
#' (see R default `atan()` function).
#' @examples
#' \dontrun{
#' x <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows and 3 columns
#' x_i <- LazyTensor(x, index = 'i')   # creating LazyTensor from matrix x, indexed by 'i'
#' Atan_x <- atan(x_i)                 # symbolic matrix, 150 rows and 3 columns
#' }
#' @export
atan <- function(x) {
    UseMethod("atan")
}

atan.LazyTensor <- function(x) {
    res <- unaryop.LazyTensor(x, "Atan")
    return(res)
}

atan.ComplexLazyTensor <- function(x) {
    res <- unaryop.LazyTensor(x, "Atan", res_type = "ComplexLasyTensor")
    return(res)
}


# arctan2 ----------------------------------------------------------------------
atan2.default <- function(x, y) {
    .Internal(atan2(x, y))
}

#' Element-wise atan2.
#' @description
#' Symbolic binary operation for element-wise 2-argument arc-tangent function.
#' @details If `x` or `y` is a `LazyTensor`, `atan2(x, y)` returns a `LazyTensor` that encodes, symbolically,
#' the element-wise atan2 of `x` and `y`. (In case one of the arguments is a vector or a scalar,
#' it is first converted to LazyTensor). 
#' If none of the arguments is a LazyTensor, it computes R default atan2 function.
#' 
#' **Note**
#' 
#' `x` and `y` input arguments should have the same inner dimension.
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param x A `LazyTensor`, a `ComplexLazyTensor`, a vector of numeric values, or a scalar value.
#' @param y A `LazyTensor`, a `ComplexLazyTensor`, a vector of numeric values, or a scalar value.
#' @return An object of class "LazyTensor" if the function is called with a `LazyTensor`,
#' and an object of class "numeric", "matrix", or "array" otherwise, depending on the input class
#' (see R default `atan2()` function)
#' @examples
#' \dontrun{
#' x <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows and 3 columns
#' y <- matrix(runif(250 * 3), 250, 3) # arbitrary R matrix, 250 rows and 3 columns
#' x_i <- LazyTensor(x, index = 'i')   # creating LazyTensor from matrix x, indexed by 'i'
#' y_j <- LazyTensor(y, index = 'j')   # creating LazyTensor from matrix x, indexed by 'i'
#' Atan2_xy <- atan2(x_i, y_j)         # symbolic matrix
#' }
#' @export
atan2 <- function(x, y) {
    if(!is.LazyTensor(x) && !is.ComplexLazyTensor(x)) 
        UseMethod("atan2", y)
    else
        UseMethod("atan2", x)
}

atan2.LazyTensor <- function(x, y) {
    res <- binaryop.LazyTensor(x, y, "Atan2", dim_check_type = "same")
    return(res)
}

atan2.ComplexLazyTensor <- function(x, y) {
    res <- binaryop.LazyTensor(x, y, "Atan2", dim_check_type = "same")
    return(res)
}


# absolute value ---------------------------------------------------------------
abs.default <- .Primitive("abs")

#' Element-wise absolute value.
#' @description
#' Symbolic unary operation for element-wise absolute value.
#' @details If `x` is a `LazyTensor`, `abs(x)` returns a `LazyTensor` that encodes, symbolically,
#' the element-wise absolute value of `x` ; else, computes R default absolute value function.
#' If `x` is a `ComplexLazyTensor`, `abs(x)` returns a `LazyTensor` that encodes, symbolically,
#' the modulus of `x` ; else, computes R default absolute value function.
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param x A `LazyTensor`, a `ComplexLazyTensor`, a vector or a matrix of numeric values, or a scalar value.
#' @return An object of class "LazyTensor" if the function is called with a `LazyTensor`,
#' and an object of class "numeric", "matrix", or "array" otherwise, depending on the input class
#' (see R default `abs()` function).
#' @examples
#' \dontrun{
#' x <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows and 3 columns
#' x_i <- LazyTensor(x, index = 'i')   # creating LazyTensor from matrix x, indexed by 'i'
#' Abs_x <- abs(x_i)                   # symbolic matrix, 150 rows and 3 columns
#' }
#' @export
abs <- function(x) {
    UseMethod("abs")
}

abs.LazyTensor <- function(x) {
    res <- unaryop.LazyTensor(x, "Abs")
    return(res)
}

abs.ComplexLazyTensor <- function(x) {
    res <- unaryop.LazyTensor(x, "ComplexAbs")
    return(res)
}


# sign function ----------------------------------------------------------------
sign.default <- .Primitive("sign")

#' Element-wise sign.
#' @description
#' Symbolic unary operation for element-wise sign.
#' @details If `x` is a `LazyTensor`, `sign(x)` returns a `LazyTensor` that encodes, symbolically,
#' the element-wise sign of `x` in {-1, 0, +1} ; else, computes R default sign function.
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param x A `LazyTensor`, a `ComplexLazyTensor`, a vector or a matrix of numeric values, or a scalar value.
#' @return An object of class "LazyTensor" if the function is called with a `LazyTensor`,
#' and an object of class "numeric", "matrix", or "array" otherwise, depending on the input class
#' (see R default `sign()` function).
#' @examples
#' \dontrun{
#' x <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows and 3 columns
#' x_i <- LazyTensor(x, index = 'i')   # creating LazyTensor from matrix x, indexed by 'i'
#' Sign_x <- sign(x_i)                 # symbolic matrix, 150 rows and 3 columns
#' }
#' @export
sign <- function(x) {
    UseMethod("sign")
}

sign.LazyTensor <- function(x) {
    res <- unaryop.LazyTensor(x, "Sign")
    return(res)
}

sign.ComplexLazyTensor <- function(x) {
    res <- unaryop.LazyTensor(x, "Sign", res_type = "ComplexLazyTensor")
    return(res)
}


# round function ---------------------------------------------------------------
round.default <- .Primitive("round")

#' Element-wise rounding function.
#' @description
#' Symbolic binary operation for element-wise rounding function.
#' @details If `x` is a `LazyTensor`, `round(x, d)` returns a `LazyTensor` that encodes, symbolically,
#' the element-wise rounding of `x` to `d` decimal places ; else, computes R default rounding function.
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param x A `LazyTensor`, a `ComplexLazyTensor`, a vector or a matrix of numeric values, or a scalar value.
#' @param d A scalar value. (or a complex ?)
#' @return An object of class "LazyTensor" if the function is called with a `LazyTensor`,
#' and an object of class "numeric", "matrix", or "array" otherwise, depending on the input class
#' (see R default `round()` function).
#' @examples
#' \dontrun{
#' x <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows and 3 columns
#' x_i <- LazyTensor(x, index = 'i')   # creating LazyTensor from matrix x, indexed by 'i'
#' Round_x <- round(x_i, 2)            # symbolic matrix
#' }
#' @export
round <- function(x, ...) {
    UseMethod("round", x)
}

round.LazyTensor <- function(x, d) {
    if(is.numeric(d) && length(d) == 1)
        res <- unaryop.LazyTensor(x, "Round", d)
    else
        stop("`d` input argument should be a scalar.")
    return(res)
}

round.ComplexLazyTensor <- function(x, d) {
    if(is.numeric(d) && length(d) == 1)
        res <- unaryop.LazyTensor(x, "Round", d, res_type = "ComplexLazyTensor")
    else
        stop("`d` input argument should be a scalar.")
    return(res)
}

# xlogx function ---------------------------------------------------------------
xlogx.default <- function(x) {
    if(x == 0)
        res <- 0
    else
        res <- x * log(x)
    return(res)
}

#' Element-wise x*log(x) function.
#' @description
#' Symbolic unary operation for element-wise sign.
#' @details If `x` is a `LazyTensor`, `xlogx(x)` returns a `LazyTensor` that encodes, symbolically,
#' the element-wise `x` times logarithm of `x` (with value 0 at 0); else, computes `x * log(x)`.
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param x A `LazyTensor`, a `ComplexLazyTensor`, a vector or a matrix of numeric values, or a scalar value.
#' @return An object of class "LazyTensor" if the function is called with a `LazyTensor`,
#' and an object of class "numeric", "matrix", or "array" otherwise, depending on the input class.
#' @examples
#' \dontrun{
#' x <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows and 3 columns
#' x_i <- LazyTensor(x, index = 'i')   # creating LazyTensor from matrix x, indexed by 'i'
#' xlog_x <- xlogx(x_i)                # symbolic matrix, 150 rows and 3 columns
#' }
#' @export
xlogx <- function(x) {
    UseMethod("xlogx", x)
}

xlogx.LazyTensor <- function(x) {
    res <- unaryop.LazyTensor(x, "XLogX")
    return(res)
}

xlogx.ComplexLazyTensor <- function(x) {
    res <- unaryop.LazyTensor(x, "XLogX", res_type = "ComplexLazyTensor")
    return(res)
}


# sinxdivx function ------------------------------------------------------------
sinxdivx.default <- function(x) {
    if(x == 0)
        res <- 1
    else
        res <- sin(x) / x
    return(res)
}

#' Element-wise sin(x)/x function.
#' @description
#' Symbolic unary operation for element-wise sign.
#' @details If `x` is a `LazyTensor`, `xlogx(x)` returns a `LazyTensor` that encodes, symbolically,
#' the element-wise sin(x)/x function of `x` (with value 0 at 0); else, computes `sin(x) / x`.
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param x A `LazyTensor`, a `ComplexLazyTensor`, a vector or a matrix of numeric values, or a scalar value.
#' @return An object of class "LazyTensor" if the function is called with a `LazyTensor`,
#' and an object of class "numeric", "matrix", or "array" otherwise, depending on the input class.
#' @examples
#' \dontrun{
#' x <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows and 3 columns
#' x_i <- LazyTensor(x, index = 'i')   # creating LazyTensor from matrix x, indexed by 'i'
#' sindiv_x <- sinxdivx(x_i)           # symbolic matrix, 150 rows and 3 columns
#' }
#' @export
sinxdivx <- function(x) {
    UseMethod("sinxdivx", x)
}

sinxdivx.LazyTensor <- function(x) {
    res <- unaryop.LazyTensor(x, "SinXDivX")
    return(res)
}

sinxdivx.ComplexLazyTensor <- function(x) {
    res <- unaryop.LazyTensor(x, "SinXDivX", res_type = "ComplexLazyTensor")
    return(res)
}


# step function ----------------------------------------------------------------

#' Element-wise step function.
#' @description
#' Symbolic unary operation for element-wise step function.
#' @details If `x` is a `LazyTensor`, `step(x)` returns a `LazyTensor` that encodes, symbolically,
#' the element-wise step function of `x` ; else, computes R default step function with other specific arguments 
#' (see R default `step()` function).
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param x A `LazyTensor`, a `ComplexLazyTensor`, a vector or a matrix of numeric values, or a scalar value.
#' @return An object of class "LazyTensor" if the function is called with a `LazyTensor`,
#' and an object of class "numeric", "matrix", or "array" otherwise, depending on the input class
#' (see R default `step()` function).
#' @examples
#' \dontrun{
#' x <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows and 3 columns
#' x_i <- LazyTensor(x, index = 'i')   # creating LazyTensor from matrix x, indexed by 'i'
#' Step_x <- step.LazyTensor(x_i)      # symbolic matrix, 150 rows and 3 columns
#' }
#' @export
step.LazyTensor <- function(x) {
    if(is.LazyTensor(x) || is.ComplexLazyTensor(x))
        res <- unaryop.LazyTensor(x, "Step", res_type = class(x))
    
    else if(is.complex(x))
        res <- unaryop.LazyTensor(x, "Step")
    
    return(res)
}


# relu function ----------------------------------------------------------------

#' Element-wise ReLU function.
#' @description
#' Symbolic unary operation for element-wise ReLU function.
#' @details `relu(x)` returns a `LazyTensor` that encodes, symbolically,
#' the element-wise ReLU of `x`.
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param x A `LazyTensor`, a `ComplexLazyTensor`, a vector of numeric values, or a scalar value.
#' @return An object of class "LazyTensor".
#' @examples
#' \dontrun{
#' x <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows and 3 columns
#' x_i <- LazyTensor(x, index = 'i')   # creating LazyTensor from matrix x, indexed by 'i'
#' ReLU_x <- relu(x_i)                 # symbolic matrix, 150 rows and 3 columns
#' }
#' @export
relu <- function(x) {
    if(is.LazyTensor(x) || is.ComplexLazyTensor(x))
        res <- unaryop.LazyTensor(x, "ReLU", res_type = class(x))
    else
        res <- unaryop.LazyTensor(x, "ReLU")
    return(res)
}


# clamp function ---------------------------------------------------------------

#' Element-wise clamp function.
#' @description
#' Symbolic ternary operation for element-wise clamp function.
#' @details `clamp(x, a, b)` returns a `LazyTensor` that encodes, symbolically,
#' the element-wise clamping of ``x`` in ``(a, b)``. That is, `clamp(x, a, b)`
#' encodes symbolically `a` if `x < a`, `x` if `a <= x <= b`, and `b` if `b < x`.
#' Broadcasting rules apply.
#' 
#' **Note**
#' 
#' If `a` and `b` are not scalar values, these should have the same inner dimension as `x`.
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param x A `LazyTensor`, a `ComplexLazyTensor`, a vector of numeric values, or a scalar value.
#' @param a A `LazyTensor`, a `ComplexLazyTensor`, a vector of numeric values, or a scalar value.
#' @param b A `LazyTensor`, a `ComplexLazyTensor`, a vector of numeric values, or a scalar value.
#' @return An object of class "LazyTensor".
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
    if((is.numeric(a) && is.numeric(b)) && ((as.integer(a) - a) == 0 && (as.integer(b) - b) == 0))
        res <- unaryop.LazyTensor(x, "ClampInt", a, b)
    else
        res <- ternaryop.LazyTensor(x, a, b, "Clamp")
    return(res)
}


# clampint function ---------------------------------------------------------------

#' Element-wise clampint function.
#' @description
#' Symbolic ternary operation for element-wise clampint function.
#' @details `clampint(x, y, z)` returns a `LazyTensor` that encodes, symbolically,
#' the element-wise clamping of ``x`` in ``(y, z)`` which are integers. See `?clamp`
#' for more details.
#' Broadcasting rules apply.
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param x A `LazyTensor`, a `ComplexLazyTensor`, a vector of numeric values, or a scalar value.
#' @param y An `integer`.
#' @param z An `integer`.
#' @return An object of class "LazyTensor".
#' @examples
#' \dontrun{
#' }
#' @export
clampint <- function(x, y, z) {
    if((!is.numeric(y) || !is.numeric(z)) || ((as.integer(y) - y) != 0 || (as.integer(z) - z) != 0)) {
        stop("'clampint(x, y, z)' expects integer arguments for `y` and `z`. Use clamp(x, y, z) for different `y` and `z` types.")
    }
    res <- unaryop.LazyTensor(x, "ClampInt", y, z)
}


# ifelse function --------------------------------------------------------------
# Keep ".LazyTensor" because R ifelse function isn't an .Internal nor a .Primitive
# but is different from KeOps IfElse function.

#' Element-wise if-else function.
#' @description
#' Symbolic ternary operation for element-wise if-else function.
#' @details `ifelse.LazyTensor(x, a, b)` returns a `LazyTensor` that encodes, symbolically,
#' `a` where ``x >= 0`` and ``b`` where ``x < 0``.  Broadcasting rules apply. 
#' `a` and `b` may be fixed integers or floats, or other `LazyTensor`.
#' 
#' **Note**
#' 
#' If `a` and `b` are not scalar values, these should have the same inner dimension as `x`.
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param x A `LazyTensor`, a `ComplexLazyTensor`, a vector of numeric values, or a scalar value.
#' @param a A `LazyTensor`, a `ComplexLazyTensor`, a vector of numeric values, or a scalar value.
#' @param b A `LazyTensor`, a `ComplexLazyTensor`, a vector of numeric values, or a scalar value.
#' @return An object of class "LazyTensor".
#' @examples
#' \dontrun{
#' }
#' @export
ifelse.LazyTensor <- function(x, a, b) {
    res <- ternaryop.LazyTensor(x, a, b, "IfElse")
}


# mod function --------------------------------------------------------------

#' Element-wise modulo with offset function.
#' @description
#' Symbolic ternary operation for element-wise modulo with offset function.
#' @details `mod(x, a, b)` returns a `LazyTensor` that encodes, symbolically,
#' the element-wise modulo of `x` with modulus `a` and offset `b`. That is,
#' `mod(x, a, b)` encodes symbolically `x - a * floor((x - b)/a)`.
#' By default `b = 0`, so that `mod(x, a)` becomes equivalent to the R function `%%`.
#' `a` and `b` may be fixed integers or floats, or other `LazyTensor`.
#' Broadcasting rules apply.
#' 
#' **Note**
#' 
#' If `a` and `b` are not scalar values, these should have the same inner dimension as `x`.
#' 
#' **Warning**
#' 
#' Do not confuse with `Mod()`.
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param x A `LazyTensor`, a `ComplexLazyTensor`, a vector of numeric values, or a scalar value.
#' @param a A `LazyTensor`, a vector of numeric values, or a scalar value.
#' @param b A `LazyTensor`, a vector of numeric values, or a scalar value.
#' @return An object of class "LazyTensor".
#' @examples
#' \dontrun{
#' TODO add example
#' }
#' @export
mod <- function(x, ...) {
    UseMethod("mod", x)
}

mod.LazyTensor <- function(x, a, b = 0) {
    res <- ternaryop.LazyTensor(x, a, b, "Mod")
}



# Squared Euclidean norm -------------------------------------------------------

#' Squared Euclidean norm.
#' @description
#' Symbolic unary operation for squared Euclidean norm.
#' @details `sqnorm2(x)` returns a `LazyTensor` that encodes, symbolically,
#' the squared Euclidean norm of `x`.
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param x A `LazyTensor`, a `ComplexLazyTensor`, a vector of numeric values, or a scalar value.
#' @return An object of class "LazyTensor".
#' @examples
#' \dontrun{
#' x <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows and 3 columns
#' x_i <- LazyTensor(x, index = 'i')   # creating LazyTensor from matrix x, indexed by 'i'
#' SqN_x <- sqnorm2(x_i)               # symbolic matrix, 150 rows and 3 columns
#' }
#' @export
sqnorm2 <- function(x) {
    if(is.LazyTensor(x) || is.ComplexLazyTensor(x))
        res <- unaryop.LazyTensor(x, "SqNorm2", res_type = class(x))
    else
        res <- unaryop.LazyTensor(x, "SqNorm2")
    return(res)
}


# Euclidean norm ---------------------------------------------------------------

#' Euclidean norm.
#' @description
#' Symbolic unary operation for Euclidean norm.
#' @details `norm2(x)` returns a `LazyTensor` that encodes, symbolically,
#' the Euclidean norm of `x`.
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param x A `LazyTensor`, a vector of numeric values, or a scalar value.
#' @return An object of class "LazyTensor".
#' @examples
#' \dontrun{
#' x <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows and 3 columns
#' x_i <- LazyTensor(x, index = 'i')   # creating LazyTensor from matrix x, indexed by 'i'
#' N_x <- norm2(x_i)                   # symbolic matrix, 150 rows and 3 columns
#' }
#' @export
norm2 <- function(x) {
    if(is.LazyTensor(x) || is.ComplexLazyTensor(x))
        res <- unaryop.LazyTensor(x, "Norm2", res_type = class(x))
    else
        res <- unaryop.LazyTensor(x, "Norm2")
    return(res)
}


# Vector normalization ---------------------------------------------------------

#' Vector normalization.
#' @description
#' Symbolic unary operation for vector normalization.
#' @details `normalize(x)` returns a `LazyTensor` that encodes, symbolically,
#' the vector normalization of `x`.
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param x A `LazyTensor`, a `ComplexLazyTensor`, a vector of numeric values, or a scalar value.
#' @return An object of class "LazyTensor".
#' @examples
#' \dontrun{
#' x <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows and 3 columns
#' x_i <- LazyTensor(x, index = 'i')   # creating LazyTensor from matrix x, indexed by 'i'
#' N_x <- norm2(x_i)               # symbolic matrix, 150 rows and 3 columns
#' }
#' @export
normalize <- function(x) {
    if(is.LazyTensor(x) || is.ComplexLazyTensor(x))
        res <- unaryop.LazyTensor(x, "Normalize", res_type = "ComplexLazyTensor")
    else 
        res <- unaryop.LazyTensor(x, "Normalize")
    return(res)
}



# Squared distance -------------------------------------------------------------

#' Squared distance.
#' @description
#' Symbolic binary operation for vector normalization.
#' @details `sqdist(x)` returns a `LazyTensor` that encodes, symbolically,
#' the squared Euclidean distance between `x` and `y`.
#' 
#' **Note**
#' 
#' `x` and `y` input arguments should have the same inner dimension or be of dimension 1.
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param x A `LazyTensor`, a `ComplexLazyTensor`, a vector of numeric values, or a scalar value.
#' @param y A `LazyTensor`, a `ComplexLazyTensor`, a vector of numeric values, or a scalar value.
#' @return An object of class "LazyTensor".
#' @examples
#' \dontrun{
#' x <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows and 3 columns
#' x_i <- LazyTensor(x, index = 'i')   # creating LazyTensor from matrix x, indexed by 'i'
#' SqD_x <- sqdist(x_i)               # symbolic matrix, 150 rows and 3 columns
#' }
#' @export
sqdist <- function(x, y) {
    res <- binaryop.LazyTensor(x, y, "SqDist")
    return(res)
}


# Weighted squared norm --------------------------------------------------------

# TODO

#' Generic squared eucidian norm.
#' @description
#' Symbolic binary operation for weighted squared norm of a LazyTensor.
#' @details `weightedsqnorm(x)` returns a `LazyTensor` that encodes, symbolically,
#' the weighted squared norm of a vector `x` with weights stored in the LazyTensor `s`.
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param x A `vector` of numeric values or a scalar value.
#' @param s A `LazyTensor`, a `ComplexLazyTensor`, a vector of numeric values, or a scalar value.
#' @return An object of class "LazyTensor".
#' @examples
#' \dontrun{
#' x <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows and 3 columns
#' y <- matrix(runif(100 * 4), 100, 4) # arbitrary R matrix, 100 rows and 4 columns
#' x_i <- LazyTensor(x, index = 'i')   # creating LazyTensor from matrix x, indexed by 'i'
#' y_j <- LazyTensor(y, index = 'j')   # creating LazyTensor from matrix y, indexed by 'j'
#' 
#' wsqn_xy <- weightedsqnorm(x_i, y_j)
#' }
#' @export
weightedsqnorm <- function(x, s) {
    res <- binaryop.LazyTensor(x, s, "WeightedSqNorm")
    return(res)
}


# Weighted squared distance ----------------------------------------------------


#' Generic squared distance.
#' @description
#' Symbolic binary operation for weighted squared distance of a LazyTensor.
#' @details `weightedsqdist(x)` returns a `LazyTensor` that encodes, symbolically,
#' the weighted squared distance of a vector `x` with weights stored in the LazyTensor `s`.
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param x A `vector` of numeric values or a scalar value.
#' @param y A `LazyTensor`, a `ComplexLazyTensor`, a vector of numeric values, or a scalar value.
#' @param z A `LazyTensor`, a `ComplexLazyTensor`, a vector of numeric values, or a scalar value.
#' @return An object of class "LazyTensor".
#' @examples
#' \dontrun{

#' }
#' @export
weightedsqdist <- function(x, y, z) {
    res <- weightedsqnorm(x - y, z)
    return(res)
}



# COMPLEX FUNCTIONS ============================================================


# real -------------------------------------------------------------------------
Re.default <- .Primitive("Re")

#' Element-wise real part of complex.
#' @description
#' Symbolic unary operation for element-wise real part of complex.
#' @details If `z` is a `ComplexLazyTensor`, `Re(z)` returns a `ComplexLazyTensor` that encodes, symbolically,
#' the element-wise real part of complex `z` ; else, computes R default `Re()` function.
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param z A `ComplexLazyTensor` or any type of values accepted by R default `Re()` function.
#' @return An object of class "ComplexLazyTensor" if the function is called with a `ComplexLazyTensor`, else
#' see R default `Re()` function.
#' @examples
#' \dontrun{
#' z <- matrix(2 + 1i^ (-6:5), nrow = 4)
#' z_i <- LazyTensor(x, "i", is_complex = TRUE)
#' 
#' Re_z <- Re(z_i)
#' }
#' @export
Re <- function(z) {
    UseMethod("Re", z)
}

Re.LazyTensor <- function(z) {
    stop("`Re` cannot be applied to a LazyTensor. See `?Re` for compatible types.")
}

Re.ComplexLazyTensor <- function(z) {
    res <- unaryop.LazyTensor(z, "ComplexReal")
}


# imaginary --------------------------------------------------------------------
Im.default <- .Primitive("Im")

#' Element-wise imaginary part of complex.
#' @description
#' Symbolic unary operation for element-wise imaginary part of complex.
#' @details If `z` is a `ComplexLazyTensor`, `Im(z)` returns a `ComplexLazyTensor` that encodes, symbolically,
#' the element-wise imaginary part of complex `z` ; else, computes R default `Im()` function.
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param z A `ComplexLazyTensor` or any type of values accepted by R default `Im()` function.
#' @return An object of class "ComplexLazyTensor" if the function is called with a `ComplexLazyTensor`, else
#' see R default `Im()` function.
#' @examples
#' \dontrun{
#' z <- matrix(2 + 1i^ (-6:5), nrow = 4)
#' z_i <- LazyTensor(z, "i", is_complex = TRUE)
#' 
#' Im_z <- Im(z_i)
#' }
#' @export
Im <- function(z) {
    UseMethod("Im", z)
}

Im.LazyTensor <- function(z) {
    stop("`Im` cannot be applied to a LazyTensor. See `?Im` for compatible types.")
}

Im.ComplexLazyTensor <- function(z) {
    res <- unaryop.LazyTensor(z, "ComplexImag")
}


# angle ------------------------------------------------------------------------
Arg.default <- .Primitive("Arg")

#' Element-wise angle (or argument) of complex.
#' @description
#' Symbolic unary operation for element-wise angle (or argument) of complex.
#' @details If `z` is a `ComplexLazyTensor`, `Arg(z)` returns a `ComplexLazyTensor` that encodes, symbolically,
#' the element-wise angle (or argument) of complex `z` ; else, computes R default `Arg()` function.
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param z A `ComplexLazyTensor` or any type of values accepted by R default `Arg()` function.
#' @return An object of class "ComplexLazyTensor" if the function is called with a `ComplexLazyTensor`, else
#' see R default `Arg()` function.
#' @examples
#' \dontrun{
#' z <- matrix(2 + 1i^ (-6:5), nrow = 4)
#' z_i <- LazyTensor(z, "i", is_complex = TRUE)
#' 
#' Arg_z <- Arg(z_i)
#' }
#' @export
Arg <- function(z) {
    UseMethod("Arg", z)
}

Arg.LazyTensor <- function(z) {
    stop("`Arg` cannot be applied to a LazyTensor. See `?Arg` for compatible types.")
}

Arg.ComplexLazyTensor <- function(z) {
    res <- unaryop.LazyTensor(z, "ComplexAngle")
}


# real to complex ------------------------------------------------------------------------

#' Element-wise "real 2 complex" operation.
#' @description
#' Symbolic unary operation for element-wise "real 2 complex".
#' @details `real2complex(x)` returns a `ComplexLazyTensor` that encodes, symbolically,
#' the element-wise "real 2 complex" of `x`.
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param x A `LazyTensor`.
#' @return An object of class "ComplexLazyTensor".
#' @examples
#' \dontrun{
#' x <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows and 3 columns
#' x_i <- LazyTensor(x, "i")           # creating LazyTensor from matrix x, indexed by 'i'
#' 
#' z <- real2complex(x_i)              # ComplexLazyTensor object
#' }
#' @export
real2complex <- function(x) {
    UseMethod("real2complex", x)
}

real2complex.LazyTensor <- function(x) {
    res <- unaryop.LazyTensor(x, "Real2Complex", res_type = "ComplexLazyTensor")
}

real2complex.ComplexLazyTensor <- function(x) {
    stop("`real2complex` cannot be applied to a complex LazyTensor.")
}


# imaginary to complex ------------------------------------------------------------------------

#' Element-wise "imaginary 2 complex" operation.
#' @description
#' Symbolic unary operation for element-wise "imaginary 2 complex".
#' @details `imag2complex(x)` returns a `ComplexLazyTensor` that encodes,
#' symbolically, the element-wise "imaginary 2 complex" of `x`.
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param x A `LazyTensor`.
#' @return An object of class "ComplexLazyTensor".
#' @examples
#' \dontrun{
#' x <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows and 3 columns
#' x_i <- LazyTensor(x, "i")           # creating LazyTensor from matrix x,
#'                                     # indexed by 'i'
#' 
#' z <- imag2complex(x_i)              # ComplexLazyTensor object
#' }
#' @export
imag2complex <- function(x) {
    UseMethod("imag2complex", x)
}

imag2complex.LazyTensor <- function(x) {
    res <- unaryop.LazyTensor(x, "Imag2Complex", res_type = "ComplexLazyTensor")
}

imag2complex.ComplexLazyTensor <- function(x) {
    stop("`imag2complex` cannot be applied to a complex LazyTensor.")
}


# complex exponential of 1j x --------------------------------------------------

#' Element-wise "complex exponential of 1j x"" operation.
#' @description
#' Symbolic unary operation for element-wise "complex exponential of 1j x".
#' @details `exp1j(x)` returns a `ComplexLazyTensor` that encodes, symbolically,
#' the multiplication of `1j` with `x`.
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param x A `LazyTensor`.
#' @return An object of class "ComplexLazyTensor".
#' @examples
#' \dontrun{
#' x <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows and 3 columns
#' x_i <- LazyTensor(x, "i")           # creating LazyTensor from matrix x, indexed by 'i'
#' 
#' z <- exp1j(x_i)                     # ComplexLazyTensor object
#' }
#' @export
exp1j <- function(x) {
    UseMethod("exp1j", x)
}

exp1j.LazyTensor <- function(x) {
    res <- unaryop.LazyTensor(x, "ComplexExp1j", res_type = "ComplexLazyTensor")
}

exp1j.ComplexLazyTensor <- function(x) {
    stop("`exp1j` cannot be applied to a complex LazyTensor.")
}


# complex conjugate ------------------------------------------------------------
Conj.default <- .Primitive("Conj") # assign default as current definition

#' Element-wise complex conjugate.
#' @description
#' Symbolic unary operation for element-wise complex conjugate.
#' @details If `z` is a `ComplexLazyTensor`, `Conj(z)` returns a `ComplexLazyTensor` that encodes,
#' symbolically, the element-wise complex conjugate of `z` ; else, computes R default `Conj()` function.
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param z A `ComplexLazyTensor`, or any type of values accepted by R default `Conj()` function.
#' @return A `ComplexLazyTensor`.
#' @examples
#' \dontrun{
#' # basic example
#' z <- matrix(1i^ (-6:5), nrow = 4)                     # create a complex 4x3 matrix
#' z_i <- LazyTensor(z, index = 'i', is_complex = TRUE)  # create a ComplexLazyTensor
#' Conj_z_i <- Conj(z_i)                                 # symbolic matrix
#' }
#' @export
Conj <- function(z) { 
    UseMethod("Conj", z)
}

Conj.LazyTensor <- function(z) {
    stop("`Conj` cannot be applied to a LazyTensor. See `?Conj` for compatible types.")
}

Conj.ComplexLazyTensor <- function(z) {
    res <- unaryop.LazyTensor(z, "Conj", res_type = "ComplexLazyTensor")
}


# complex modulus --------------------------------------------------------------
Mod.default <- .Primitive("Mod") # assign default as current definition

#' Element-wise absolute value (or modulus).
#' @description
#' Symbolic unary operation for element-wise absolute value (or modulus).
#' @details If `z` is a `ComplexLazyTensor`, `Mod(z)` returns a `LazyTensor` that encodes,
#' symbolically, the element-wise absolute value (or modulus) of `z` ; else, computes R default `Mod()` function.
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param z A `ComplexLazyTensor`, or any type of values accepted by R default `Mod()` function.
#' @return A `LazyTensor`.
#' @examples
#' \dontrun{
#' # basic example
#' z <- matrix(1i^ (-6:5), nrow = 4)                     # create a complex 4x3 matrix
#' z_i <- LazyTensor(z, index = 'i', is_complex = TRUE)  # create a ComplexLazyTensor
#' Mod_z_i <- Mod(z_i)                                   # symbolic matrix
#' }
#' @export
Mod <- function(z) { 
    UseMethod("Mod", z)
}

Mod.ComplexLazyTensor <- function(z) {
    res <- unaryop.LazyTensor(z, "ComplexAbs")
}




# REDUCTIONS ===================================================================


# Reduction --------------------------------------------------------------------

#' Reduction operation.
#' @description
#' Applies a reduction to a `LazyTensor`.
#' @details `reduction.LazyTensor(x, opstr, index)` will :
#' \itemize{
#'   \item if **index = "i"**, return the **opstr** reduction of **x** over the "i" indexes;
#'   \item if **index = "j"**, return the **opstr** reduction of **x** over the "j" indexes.
#' }
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param x A `LazyTensor` or a `ComplexLazyTensor`.
#' @param opstr A `string` formula (like "Sum" or "Max").
#' @param index A `character` that should be either **i** or **j** to specify whether if 
#' the reduction is indexed by **i** (rows), or **j** (columns).
#' @return
#' @examples
#' \dontrun{
#' x <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows and 3 columns
#' x_i <- LazyTensor(x, index = 'i')   # creating LazyTensor from matrix x, indexed by 'i'
#' 
#' red_x <- reduction.LazyTensor(x_i, "Sum", "i")
#' }
#' @export
reduction.LazyTensor <- function(x, opstr, index) {
    if(!is.LazyTensor(x) && !is.ComplexLazyTensor(x))
        stop("`x` input should be a LazyTensor or a ComplexLazyTensor.")
    
    if(!is.character(opstr))
        stop("`opst` input should be a string text.")
    
    if(check_index(index)) {
        if(index == "i") 
            tag <- 1
        else 
            tag <- 0
        formula <- paste(opstr, "_Reduction(", x$formula, ",", tag, ")", sep = "")
        # ici mettre arg optionnel 
        args <- x$args
        op <- keops_kernel(formula, args)
        res <- op(x$vars)
    }
    
    else
        stop("`index` input argument should be a character `i`, `j`.")
    
    return(res)
}


# sum function -----------------------------------------------------------------
sum.default <- .Primitive("sum")

#' Summation operation or Sum reduction.
#' @description
#' Summation unary operation, or Sum reduction.
#' @details If `x` is a `LazyTensor`, `sum(x, index)` will :
#' \itemize{
#'   \item if **index = "i"**, return the min reduction of **x** over the "i" indexes.
#'   \item if **index = "j"**, return the min reduction of **x** over the "j" indexes.
#'   \item if **index = NA** (default), return a new `LazyTensor` object representing 
#'   the min of the values of the vector.
#' }
#' If `x` is not a `LazyTensor` it computes R default "sum" function with
#' other specific arguments (see R default `sum()` function).
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param x A `LazyTensor`, a `ComplexLazyTensor`, a vector or a matrix of numeric values, or a scalar value.
#' @param index A `character` corresponding to the reduction dimension that should 
#' be either **i** or **j** to specify whether if the summation is indexed by **i** (rows), or **j** (columns).
#' It can be NA (default) when no reduction is desired.
#' @return 
#' @examples
#' \dontrun{
#' x <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows and 3 columns
#' x_i <- LazyTensor(x, index = 'i')   # creating LazyTensor from matrix x, indexed by 'i'
#' 
#' sum_x <- sum(x_i) # LazyTensor object
#' sum_red_x <- sum(x_i, "i")  # sum reduction indexed by 'i'
#' }
#' @export
sum <- function(x, index) {
    UseMethod("sum")
}

sum.LazyTensor <- function(x, index = NA) {
    if(is.na(index))
        res <- unaryop.LazyTensor(x, "Sum")
    else if(check_index(index))
        res <- reduction.LazyTensor(x, "Sum", index)
    else
        stop("`index` input argument should be a character `i`, `j` or NA.")
    return(res)
}


sum.ComplexLazyTensor <- function(x, index = NA) {
    if(is.na(index))
        res <- unaryop.LazyTensor(x, "Sum", res_type = "ComplexLazyTensor")
    else if(check_index(index))
        res <- reduction.LazyTensor(x, "Sum", index)
    else
        stop("`index` input argument should be a character `i`, `j` or NA.")
    return(res)
}


# sum reduction ----------------------------------------------------------------

#' Summation operation or Sum reduction.
#' @description
#' Summation unary operation, or Sum reduction.
#' @details `sum_reduction(x, index)` will return the sum reduction of **x** indexed by **index**.
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param x A `LazyTensor` or a `ComplexLazyTensor`.
#' @param index A `character` corresponding to the reduction dimension that should 
#' be either **i** or **j** to specify whether if the summation is indexed by **i** (rows), by **j** (columns).
#' @return 
#' @examples
#' \dontrun{
#' x <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows and 3 columns
#' x_i <- LazyTensor(x, index = 'i')   # creating LazyTensor from matrix x, indexed by 'i'
#' 
#' sum_reduction(x_i, "i")
#' }
#' @export
sum_reduction <- function(x, index){
    if(check_index(index))
        res <- reduction.LazyTensor(x, "Sum", index)
    else 
        stop("`index` input argument should be a character `i`, `j` or NA.")
    return(res)
}


# min function -----------------------------------------------------------------
min.default <- .Primitive("min")

#' Minimum.
#' @description
#' Minimum unary operation or Minimum reduction.
#' @details If `x` is a `LazyTensor`, `min(x, index)` will :
#' \itemize{
#'   \item if **index = "i"**, return the min reduction of **x** over the "i" indexes.
#'   \item if **index = "j"**, return the min reduction of **x** over the "j" indexes.
#'   \item if **index = NA** (default), return a new `LazyTensor` object representing 
#'   the min of the values of the vector.
#' }
#' If `x` is not a `LazyTensor` it computes R default "min" function with
#' other specific arguments (see R default `min()` function).
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param x A `LazyTensor`, a `ComplexLazyTensor`, a vector or a matrix of numeric values, or a scalar value.
#' @param index A `character` corresponding to the reduction dimension that should 
#' be either **i** or **j** to specify whether if the summation is indexed by **i** (rows), or **j** (columns).
#' It can be NA (default) when no reduction is desired.
#' @return TODO otherwise, depending on the input class
#' (see R default `min()` function).
#' @examples
#' \dontrun{
#' x <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows and 3 columns
#' x_i <- LazyTensor(x, index = 'i')   # creating LazyTensor from matrix x, indexed by 'i'
#' 
#' min_xi <- min(x_i, "i")  # min reduction indexed by "i"
#' min_x <- min(x_i)        # symbolic matrix
#' }
#' @export
min <- function(x, ...) {
    UseMethod("min")
}

min.LazyTensor <- function(x, index = NA) {
    if(is.na(index))
        res <- unaryop.LazyTensor(x, "Min")
    else if(check_index(index))
        res <- reduction.LazyTensor(x, "Min", index)
    else 
        stop("`index` input argument should be a character `i`, `j` or NA.")
    return(res)
}

min.ComplexLazyTensor <- function(x, index = NA) {
    if(is.na(index))
        res <- unaryop.LazyTensor(x, "Min", res_type = "ComplexLazyTensor")
    else if(check_index(index))
        res <- reduction.LazyTensor(x, "Min", index)
    else 
        stop("`index` input argument should be a character `i`, `j` or NA.")
    return(res)
}



# min reduction ----------------------------------------------------------------

#' Min reduction.
#' @description
#' Minimum reduction.
#' @details `min_reduction(x, index)` will return the min reduction of **x** indexed by **index**.
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param x A `LazyTensor` or a `ComplexLazyTensor`.
#' @return 
#' @examples
#' \dontrun{
#' x <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows and 3 columns
#' x_i <- LazyTensor(x, index = 'i')   # creating LazyTensor from matrix x, indexed by 'i'
#' 
#' min_red_x <- min_reduction(x_i, "i")  # min reduction indexed by "i"
#' }
#' @export
min_reduction <- function(x, index) {
    if(check_index(index))
        res <- reduction.LazyTensor(x, "Min", index)
    else
        stop("`index` input argument should be a character `i`, `j`.")
    return(res)
}



# argmin function --------------------------------------------------------------

#' ArgMin.
#' @description
#' ArgMin unary operation, or ArgMin reduction.
#' @details `argmin(x, index)` will :
#' \itemize{
#'   \item if **index = "i"**, return the argmin reduction of **x** over the "i" indexes.
#'   \item if **index = "j"**, return the argmin reduction of **x** over the "j" indexes.
#'   \item if **index = NA** (default), return a new `LazyTensor` object representing 
#'   the argmin  of the values of the vector.
#' }
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param x A `LazyTensor`, a `ComplexLazyTensor`, a vector or a matrix of numeric values, or a scalar value.
#' @param index A `character` corresponding to the reduction dimension that should 
#' be either **i** or **j** to specify whether if the summation is indexed by **i** (rows), 
#' or **j** (columns).
#' It can be NA (default) when no reduction is desired.
#' @return 
#' @examples
#' \dontrun{
#' x <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows and 3 columns
#' x_i <- LazyTensor(x, index = 'i')   # creating LazyTensor from matrix x, indexed by 'i'
#' 
#' argmin_xi <- argmin(x_i, "i")  # argmin reduction indexed by "i"
#' argmin_x <- argmin(x_i)        # symbolic matrix
#' }
#' @export
argmin <- function(x, index = NA) {
    if(is.na(index)) {
        if(is.LazyTensor(x) || is.ComplexLazyTensor(x))
            res <- unaryop.LazyTensor(x, "ArgMin", res_type = class(x))
        else
            res <- unaryop.LazyTensor(x, "ArgMin")
    }
    
    else if(check_index(index))
        res <- reduction.LazyTensor(x, "ArgMin", index)
    else
        stop("`index` input argument should be a character `i`, `j` or NA.")
    return(res)
}


# argmin reduction -------------------------------------------------------------

#' Argmin reduction.
#' @description
#' Argmin reduction.
#' @details `argmin_reduction(x, index)` will return the argmin reduction of `x`.
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param x A `LazyTensor` or a `ComplexLazyTensor`.
#' @param index A `character` corresponding to the reduction dimension that should 
#' be either **i** or **j** to specify whether if the summation is indexed by **i** (rows), 
#' or **j** (columns).
#' @return 
#' @examples
#' \dontrun{
#' x <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows and 3 columns
#' x_i <- LazyTensor(x, index = 'i')   # creating LazyTensor from matrix x, indexed by 'i'
#' 
#' argmin_red <- argmin(x_i, "i")  # argmin reduction indexed by "i"
#' }
#' @export
argmin_reduction <- function(x, index) {
    if(check_index(index))
        res <- reduction.LazyTensor(x, "ArgMin", index)
    else
        stop("`index` input argument should be a character `i`, `j`.")
    return(res)
}


# min_argmin -------------------------------------------------------------------

#' Min-ArgMin.
#' @description
#' Min-ArgMin reduction.
#' @details `min_argmin(x, index)` will :
#' \itemize{
#'   \item if **index = "i"**, return the minimal values and its indices of **x** over the "i" indexes.
#'   \item if **index = "j"**, return the minimal values and its indices of **x** over the "j" indexes.
#' }
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param x A `LazyTensor` or a `ComplexLazyTensor`.
#' @param index A `character` corresponding to the reduction dimension that should 
#' be either **i** or **j** to specify whether if the summation is indexed by **i** (rows),
#' or **j** (columns).
#' @return 
#' @examples
#' \dontrun{
#' x <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows and 3 columns
#' x_i <- LazyTensor(x, index = 'i')   # creating LazyTensor from matrix x, indexed by 'i'
#' 
#' min_argmin_xi <- min_argmin(x_i, "i")  # min argmin reduction indexed by "i"
#' }
#' @export
min_argmin <- function(x, index) {
    if(check_index(index))
        res <- reduction.LazyTensor(x, "Min_ArgMin", index)
    else
        stop("`index` input argument should be a character `i`, `j`.")
    return(res)
}


# min_argmin reduction -------------------------------------------------------------

#' Min-ArgMin reduction.
#' @description
#' Min-ArgMin reduction.
#' @details `min_argmin_reduction(x, index)` will return the min reduction of `x`.
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param x A `LazyTensor` or a `ComplexLazyTensor`.
#' @param index A `character` corresponding to the reduction dimension that should 
#' be either **i** or **j** to specify whether if the summation is indexed by **i** (rows), 
#' or **j** (columns).
#' @return 
#' @examples
#' \dontrun{
#' x <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows and 3 columns
#' x_i <- LazyTensor(x, index = 'i')   # creating LazyTensor from matrix x, indexed by 'i'
#' 
#' min_argmin_red <- min_argmin_reduction(x_i, "i")  # min reduction indexed by "i"
#' }
#' @export
min_argmin_reduction <- function(x, index) {
    if(check_index(index))
        res <- min_argmin(x, index)
    else
        stop("`index` input argument should be a character `i`, `j`.")
    return(res)
}



# Préciser que si on a plusieurs scalaires, on peut faire e.g. max(3, 4, 11)
# qui renvoie 11 mais pour les LazyTensor c'est juste max(x_i) qui renvoie
# l'élément maximal de x_i 
# max function -----------------------------------------------------------------
max.default <- .Primitive("max")

#' Maximum.
#' @description
#' Maximum unary operation, or Max reduction.
#' @details If `x` is a `LazyTensor`, `max(x, index)` will :
#' \itemize{
#'   \item if **index = "i"**, return the max reduction of **x** over the "i" indexes.
#'   \item if **index = "j"**, return the max reduction of **x** over the "j" indexes.
#'   \item if **index = NA** (default), return a new `LazyTensor` object representing 
#'   the max of the values of the vector.
#' }
#' If `x` is not a `LazyTensor` it computes R default "max" function with
#' other specific arguments (see R default `max()` function).
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param x A `LazyTensor`, a `ComplexLazyTensor`, a vector or a matrix of numeric values, or a scalar value.
#' @param index A `character` corresponding to the reduction dimension that should 
#' be either **i** or **j** to specify whether if the summation is indexed by **i** (rows), or **j** (columns).
#' It can be NA (default) when no reduction is desired.
#' @return
#' @examples
#' \dontrun{
#' x <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows and 3 columns
#' x_i <- LazyTensor(x, index = 'i')   # creating LazyTensor from matrix x, indexed by 'i'
#' 
#' max_xi <- max(x_i, "i")  # max reduction indexed by "i"
#' max_x <- max(x_i)        # symbolic matrix
#' }
#' @export
max <- function(x, ...) {
    UseMethod("max", x)
}

max.LazyTensor <- function(x, index = NA) {
    if(is.na(index))
        res <- unaryop.LazyTensor(x, "Max")
    else if(check_index(index))
        res <- reduction.LazyTensor(x, "Max", index)
    else 
        stop("`index` input argument should be a character `i`, `j` or NA.")
    return(res)
}

max.ComplexLazyTensor <- function(x, index = NA) {
    if(is.na(index))
        res <- unaryop.LazyTensor(x, "Max", res_type = "ComplexLazyTensor")
    else if(check_index(index))
        res <- reduction.LazyTensor(x, "Max", index)
    else 
        stop("`index` input argument should be a character `i`, `j` or NA.")
    return(res)
}


# max reduction ----------------------------------------------------------------

#' Max reduction.
#' @description
#' Maximum reduction.
#' @details `max_reduction(x, index)` will return the max reduction of **x** indexed by **index**.
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param x A `LazyTensor` or a `ComplexLazyTensor`.
#' @param index A `character` corresponding to the reduction dimension that should 
#' be either **i** or **j** to specify whether if the summation is indexed by **i** (rows), 
#' or **j** (columns).
#' @return 
#' @examples
#' \dontrun{
#' x <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows and 3 columns
#' x_i <- LazyTensor(x, index = 'i')   # creating LazyTensor from matrix x, indexed by 'i'
#' 
#' max_red_x <- max_reduction(x_i, "i")  # max reduction indexed by "i"
#' }
#' @export
max_reduction <- function(x, index) {
    if(check_index(index))
        res <- reduction.LazyTensor(x, "Max", index)
    else 
        stop("`index` input argument should be a character `i`, `j`.")
    return(res)
}


# argmax function --------------------------------------------------------------

#' ArgMax.
#' @description
#' ArgMax unary operation, or ArgMax reduction.
#' @details 
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param x A `LazyTensor`, a `ComplexLazyTensor`, a vector or a matrix of numeric values, or a scalar value.
#' @param index A `character` corresponding to the reduction dimension that should 
#' be either **i** or **j** to specify whether if the summation is indexed by **i** (rows), 
#' or **j** (columns).
#' It can be NA (default) when no reduction is desired.
#' @return 
#' @examples
#' \dontrun{
#' x <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows and 3 columns
#' x_i <- LazyTensor(x, index = 'i')   # creating LazyTensor from matrix x, indexed by 'i'
#' 
#' argmax_xi <- argmax(x_i, "i")  # argmax reduction indexed by "i"
#' argmax_x <- argmax(x_i)        # symbolic matrix
#' }
#' @export
argmax <- function(x, index = NA) {
    if(is.na(index)) {
        if(is.LazyTensor(x) || is.ComplexLazyTensor(x))
            res <- unaryop.LazyTensor(x, "ArgMax", res_type = class(x))
        else
            res <- unaryop.LazyTensor(x, "ArgMax")
    }
    else if(check_index(index))
        res <- reduction.LazyTensor(x, "ArgMax", index)
    else 
        stop("`index` input argument should be a character `i`, `j` or NA.")
    return(res)
}


# argmax reduction -------------------------------------------------------------

#' ArgMax reduction.
#' @description
#' ArgMax reduction.
#' @details `argmax_reduction(x, index)` will return the argmax reduction of `x`.
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param x A `LazyTensor` or a `ComplexLazyTensor`.
#' @param index A `character` corresponding to the reduction dimension that should 
#' be either **i** or **j** to specify whether if the summation is indexed by **i** (rows), 
#' or **j** (columns).
#' @return 
#' @examples
#' \dontrun{
#' x <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows and 3 columns
#' x_i <- LazyTensor(x, index = 'i')   # creating LazyTensor from matrix x, indexed by 'i'
#' 
#' argmax_red <- argmax_reduction(x_i, "i")  # argmax reduction indexed by "i"
#' }
#' @export
argmax_reduction <- function(x, index) {
    if(check_index(index))
        res <- reduction.LazyTensor(x, "ArgMax", index)
    else 
        stop("`index` input argument should be a character `i`, `j`.")
    return(res)
}


# max_argmax reduction -------------------------------------------------------------

#' Max-ArgMax reduction.
#' @description
#' Max-ArgMax reduction.
#' @details `max_argmax(x, index)` will :
#' \itemize{
#'   \item if **index = "i"**, return the maximal values and its indices of **x** over the "i" indexes.
#'   \item if **index = "j"**, return the maximal values and its indices of **x** over the "j" indexes.
#' }
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param x A `LazyTensor` or a `ComplexLazyTensor`.
#' @param index A `character` corresponding to the reduction dimension that should 
#' be either **i** or **j** to specify whether if the summation is indexed by **i** (rows),
#' or **j** (columns).
#' @return 
#' @examples
#' \dontrun{
#' x <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows and 3 columns
#' x_i <- LazyTensor(x, index = 'i')   # creating LazyTensor from matrix x, indexed by 'i'
#' 
#' max_argmax_x <- max_argmax(x_i, "i")  # max argmax reduction indexed by "i"
#' }
#' @export
max_argmax <- function(x, index) {
    if(check_index(index))
        res <- reduction.LazyTensor(x, "Max_ArgMax", index)
    else 
        stop("`index` input argument should be a character `i`, `j`.")
    return(res)
}


# max_argmax reduction -------------------------------------------------------------

#' Max-ArgMax reduction.
#' @description
#' Max-ArgMax reduction.
#' @details `max_argmax_reduction(x, index)` 
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param x A `LazyTensor` or a `ComplexLazyTensor`.
#' @param index A `character` corresponding to the reduction dimension that should 
#' be either **i** or **j** to specify whether if the summation is indexed by **i** (rows), 
#' or **j** (columns).
#' @return 
#' @examples
#' \dontrun{
#' x <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows and 3 columns
#' x_i <- LazyTensor(x, index = 'i')   # creating LazyTensor from matrix x, indexed by 'i'
#' 
#' max_argmax_red <- max_argmax_reduction(x_i, "i")  # max argmax reduction indexed by "i"
#' }
#' @export
max_argmax_reduction <- function(x, index) {
    if(check_index(index))
        res <- max_argmax(x, index)
    else 
        stop("`index` input argument should be a character `i`, `j`.")
    return(res)
}



# Kmin -------------------------------------------------------------------------


# Kmin reduction ---------------------------------------------------------------


# argKmin ----------------------------------------------------------------------


# argKmin reduction ------------------------------------------------------------


# Kmin-argKmin -----------------------------------------------------------------


# Kmin-argKmin reduction -------------------------------------------------------








# Basic example

#D <- 3
#M <- 100
#N <- 150
#E <- 4
#x <- matrix(runif(M * D), M, D)
#y <- matrix(runif(N * D), N, D)
#z <- matrix(runif(N * E), N, E)
#b <- matrix(runif(N * E), N, E)
#
#vect <- rep(1, 10)
#s <- 0.25
##
### creating LazyTensor from matrices
#x_i <- LazyTensor(x, index = 'i')
#y_j <- LazyTensor(y, index = 'j')
#z_j <- LazyTensor(z, index = 'j')
#
#z <- matrix(1i^ (-6:5), nrow = 4) # complex 4x3 matrix
#z_i <- LazyTensor(z, index = 'i', is_complex = TRUE)
#conj_z_i <- Conj(z_i)
#b_j = b
#
## Symbolic matrix of squared distances:
#SqDist_ij = sum( (x_i - y_j)^2 )
#
## Symbolic Gaussian kernel matrix:
#K_ij = exp( - SqDist_ij / (2 * s^2) )
#
## Genuine matrix:
#v = K_ij %*% b_j
## equivalent
## v = "%*%.LazyTensor"(K_ij, b_j)
#
#s2 = (2 * s^2)
## equivalent
#op <- keops_kernel(
#    formula = "Sum_Reduction(Exp(Minus(Sum(Square(x-y)))/s)*b,0)",
#    args = c("x=Vi(3)", "y=Vj(3)", "s=Pm(1)", "b=Vj(4)")
#)
#
#v2 <- op(list(x, y, s2, b))
#
#sum((v2-v)^2)
#
#
#
## we compare to standard R computation
#SqDist = 0
#onesM = matrix(1, 1, 2)
#onesN = matrix(1, 1, 2)
#
#for(k in 1:D) {
#    print(SqDist)
#    SqDist = SqDist + (x[, k] %*% onesN - t(y[, k] %*% onesM))^2
#    print(SqDist)
#}
#    
#
#K = exp(-SqDist/(2*s^2))
#
#v2 = K %*% b
#
#print(mean(abs(v-v2)))
#
