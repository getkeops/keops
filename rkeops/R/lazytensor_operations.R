<<<<<<< Updated upstream
# library(rkeops)
# library(stringr)
# library(data.table)
# 
# set_rkeops_option("tagCpuGpu", 0)
# set_rkeops_option("precision", "double")
=======
#library(rkeops)
#library(stringr)
#library(data.table)
#
#set_rkeops_option("tagCpuGpu", 0)
#set_rkeops_option("precision", "double")
>>>>>>> Stashed changes


# OPERATIONS ===================================================================


# addition ---------------------------------------------------------------------
"+.default" <- .Primitive("+") # assign default as current definition

#' Addition.
#' @description
#' Symbolic binary operation for addition.
#' @usage x + y
#' @details If `x` or `y` is a `LazyTensor`, `x + y` returns a `LazyTensor`
#' that encodes, symbolically, the addition of `x` and `y`.
#' (In case one of the arguments is a vector or a scalar, it is first converted 
#' to `LazyTensor`). If none of the arguments is a `LazyTensor`, is equivalent 
#' to the "+" R operator.
#' 
#' **Note**
#' 
#' `x` and `y` input arguments should have the same inner dimension or be of 
#' dimension 1.
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param x A `LazyTensor`, a `ComplexLazyTensor`, a vector of numeric values, 
#' or a scalar value.
#' @param y A `LazyTensor`, a `ComplexLazyTensor`, a vector of numeric values, 
#' or a scalar value.
#' @return An object of class "LazyTensor" if the function is called with a `LazyTensor`,
#' and an object of class "numeric", otherwise.
#' @examples
#' \dontrun{
#' x <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows and 3 columns
#' y <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows and 3 columns
#' x_i <- LazyTensor(x, index = 'i')   # creating LazyTensor from matrix x, 
#'                                     # indexed by 'i'
#' y_j <- LazyTensor(y, index = 'j')   # creating LazyTensor from matrix y, 
#'                                     # indexed by 'j'
#' Sum_xy <- x_i + y_j                 # symbolic matrix, 150 rows and 3 columns
#' }
#' @export
"+" <- function(x, y) { 
    if(!is.ComplexLazyTensor(x) && is.LazyTensor(y))
        UseMethod("+", y)
    else
        UseMethod("+", x)
}

"+.LazyTensor" <- function(x, y) {
    res <- binaryop.LazyTensor(x, y, "+", is_operator = TRUE,
                               dim_check_type = "sameor1")
    return(res)
}

"+.ComplexLazyTensor" <- function(x, y) {
    if(!is.LazyTensor(y)) {
        y <- LasyTensor(y)
    }
    
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

# subtraction  ----------------------------------------------------------------
"-.default" <- .Primitive("-") # assign default as current definition

#' Subtraction or minus sign.
#' @description
#' Symbolic binary operation for subtraction.
#' @usage x - y
#' @details Two possible use cases:
#' \itemize{
#'     \item{**Subtraction**:}{ If `x` or `y` is a `LazyTensor`, `x - y` returns 
#'     a `LazyTensor` that encodes, symbolically, the subtraction of `x` and `y`.
#'     (In case one of the arguments is a vector or a scalar, it is first converted 
#'     to `LazyTensor`). If none of the arguments is a `LazyTensor`, it is equivalent 
#'     to the "-" R operator.}
#'     \item{**Minus sign**:}{ If `x` is a `LazyTensor`, `-x` returns a `LazyTensor`
#'     that encodes, symbolically, the element-wise opposite of `x`.}
#' }
#' **Note**
#' 
#' For **subtraction operation**, `x` and `y` input arguments should have the same
#' inner dimension or be of dimension 1.
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param x A `LazyTensor`, a `ComplexLazyTensor`, a vector of numeric values, 
#' or a scalar value (matrices can be used with the minus sign only).
#' @param y A `LazyTensor`, a `ComplexLazyTensor`, a vector of numeric values, 
#' or a scalar value (matrices can be used with the minus sign only).
#' @return An object of class "LazyTensor" if the function is called with a 
#' `LazyTensor`, and an object of class "numeric", otherwise.
#' @examples
#' \dontrun{
#' x <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows and 3 columns
#' y <- matrix(runif(250 * 3), 250, 3) # arbitrary R matrix, 250 rows and 3 columns
#' x_i <- LazyTensor(x, index = 'i')   # creating LazyTensor from matrix x, 
#'                                     # indexed by 'i'
#' y_j <- LazyTensor(y, index = 'j')   # creating LazyTensor from matrix y, 
#'                                     # indexed by 'j'
#'                                     
#' Sub_xy <- x_i - y_j                 # symbolic matrix
#' Minus_x <- -x_i                     # symbolic matrix
#' }
#' @export
"-" <- function(x, y = NA) { 
    if(!is.na(y) && (!is.ComplexLazyTensor(x) && is.LazyTensor(y)))
        UseMethod("-", y)
    else
        UseMethod("-", x)
}

"-.LazyTensor" <- function(x, y = NA) {
    if((length(y) == 1) && is.na(y))
        res <- unaryop.LazyTensor(x, "Minus")
    else
        res <- binaryop.LazyTensor(x, y, "-", is_operator = TRUE,
                                   dim_check_type = "sameor1")
    return(res)
}

"-.ComplexLazyTensor" <- function(x, y = NA) {
    if((length(y) == 1) && is.na(y)) {
        res <- unaryop.LazyTensor(x, "Minus")
        return(res)
    }
    
    if(!is.LazyTensor(y) && !is.matrix(y)) {
        y <- LasyTensor(y)
    }
    
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
"*.default" <- .Primitive("*") # assign default as current definition

#' Multiplication.
#' @description
#' Symbolic binary operation for multiplication.
#' @usage x * y
#' @details If `x` or `y` is a `LazyTensor`, `x * y` returns a `LazyTensor`
#' that encodes, symbolically, the element-wise product of `x` and `y`.
#' (In case one of the arguments is a vector or a scalar, it is first converted 
#' to `LazyTensor`). If none of the arguments is a `LazyTensor`, it is equivalent 
#' to the "*" R operator.
#' 
#' **Note**
#' 
#' `x` and `y` input arguments should have the same inner dimension or be of 
#' dimension 1.
#' @author Chloé Serre-Combe, Amélie Vernay
#' @param x A `LazyTensor`, a `ComplexLazyTensor`, a vector of numeric values, 
#' or a scalar value.
#' @param y A `LazyTensor`, a `ComplexLazyTensor`, a vector of numeric values, 
#' or a scalar value.
#' @return An object of class "LazyTensor" if the function is called with a 
#' `LazyTensor`, and an object of class "numeric", otherwise.
#' @examples
#' \dontrun{
#' x <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows and 3 columns
#' y <- matrix(runif(250 * 3), 250, 3) # arbitrary R matrix, 250 rows and 3 columns
#' x_i <- LazyTensor(x, index = 'i')   # creating LazyTensor from matrix x, 
#'                                     # indexed by 'i'
#' y_j <- LazyTensor(y, index = 'j')   # creating LazyTensor from matrix y, 
#'                                     # indexed by 'j'
#'                                     
#' x_times_y <- x_i * y_j              # symbolic matrix
#' }
#' @export
"*" <- function(x, y) { 
    if(!is.ComplexLazyTensor(x) && is.LazyTensor(y))
        UseMethod("*", y)
    else
        UseMethod("*", x)
}

"*.LazyTensor" <- function(x, y) {
    res <- binaryop.LazyTensor(x, y, "*", is_operator = TRUE,
                               dim_check_type = "sameor1")
    return(res)
}

"*.ComplexLazyTensor" <- function(x, y) {
    if(!is.LazyTensor(y) && !is.matrix(y)) {
        y <- LasyTensor(y)
    }
    
    if(is.LazyScalar(y)) {
        res <- binaryop.LazyTensor(x, y, "ComplexRealScal")
    }
    
    else if(!is.ComplexLazyTensor(y)) {
        res <- x * real2complex(y)
    }
    
    else if(!is.ComplexLazyTensor(x)) {
        res <- real2complex(x) * y
    }
    
    else if(is.ComplexLazyScalar(x) || is.ComplexLazyScalar(y)) {
        res <- binaryop.LazyTensor(x, y, "ComplexScal", dim_check_type = NA)
    }
    
    else {
        res <- binaryop.LazyTensor(x, y, "ComplexMult")
    }
    
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
#' (In case one of the arguments is a vector or a scalar, it is first converted 
#' to `LazyTensor`). If none of the arguments is a `LazyTensor`, it is equivalent 
#' to the "/" R operator.
#' 
#' **Note**
#' 
#' `x` and `y` input arguments should have the same inner dimension or be of 
#' dimension 1.
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param x A `LazyTensor`, a `ComplexLazyTensor`, a vector of numeric values, 
#' or a scalar value.
#' @param y A `LazyTensor`, a `ComplexLazyTensor`, a vector of numeric values, 
#' or a scalar value.
#' @return An object of class "LazyTensor" if the function is called with a 
#' `LazyTensor`, and an object of class "numeric", otherwise.
#' @examples
#' \dontrun{
#' x <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows and 3 columns
#' y <- matrix(runif(250 * 3), 250, 3) # arbitrary R matrix, 250 rows and 3 columns
#' x_i <- LazyTensor(x, index = 'i')   # creating LazyTensor from matrix x, 
#'                                     # indexed by 'i'
#' y_j <- LazyTensor(y, index = 'j')   # creating LazyTensor from matrix y,
#'                                     # indexed by 'j'
#'                                     
#' x_div_y <- x_i / y_j                # symbolic matrix
#' }
#' @export
"/" <- function(x, y) { 
    if(!is.ComplexLazyTensor(x) && is.LazyTensor(y))
        UseMethod("/", y)
    else
        UseMethod("/", x)
}

"/.LazyTensor" <- function(x, y) {
    res <- binaryop.LazyTensor(x, y, "/", is_operator = TRUE,
                               dim_check_type = "sameor1")
    return(res)
}

"/.ComplexLazyTensor" <- function(x, y) {
    if(!is.LazyTensor(y) && !is.matrix(y)) {
        y <- LasyTensor(y)
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
square.default <- function(x) {
    res <- x^2
    return(res)
}

#' Element-wise square.
#' @description
#' Symbolic unary operation for element-wise square.
#' @details If `x` is a `LazyTensor`, `square(x)` returns a `LazyTensor` that 
#' encodes, symbolically, the element-wise square of `x` ;
#' else, is equivalent to the "^2" R operator.
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param x A `LazyTensor`, a `ComplexLazyTensor`, a vector or a matrix of numeric 
#' values, or a scalar value.
#' @return An object of class "LazyTensor" if the function is called with a 
#' `LazyTensor`, and an object of class "numeric", "matrix", or "array" otherwise, 
#' same as the input class.
#' @examples
#' \dontrun{
#' x <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows and 3 columns
#' x_i <- LazyTensor(x, index = 'i')   # creating LazyTensor from matrix x, 
#'                                     # indexed by 'i'
#'                                     
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



# square root ------------------------------------------------------------------
sqrt.default <- .Primitive("sqrt") # assign default as current definition

#' Element-wise square root.
#' @description
#' Symbolic unary operation for element-wise square root.
#' @details If `x` is a `LazyTensor`, `sqrt(x)` returns a `LazyTensor` that 
#' encodes, symbolically, the element-wise square root of `x` ; 
#' else, computes R default square root function.
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param x A `LazyTensor`, a `ComplexLazyTensor`, a vector or a matrix of numeric 
#' values, or a scalar value.
#' @return An object of class "LazyTensor" if the function is called with a 
#' `LazyTensor`, and an object of class "numeric", "matrix", or "array" otherwise, 
#' depending on the input class (see R default `sqrt()` function).
#' @examples
#' \dontrun{
#' x <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows and 3 columns
#' x_i <- LazyTensor(x, index = 'i')   # creating LazyTensor from matrix x, 
#'                                     # indexed by 'i'
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


# Rsqrt ------------------------------------------------------------------------
rsqrt.default <- function(x) {
    res <- 1 / sqrt(x)
    return(res)
}

#' Element-wise inverse square root.
#' @description
#' Symbolic unary operation for element-wise inverse square root.
#' @details If `x` is a `LazyTensor`, `sqrt(x)` returns a `LazyTensor` that encodes, 
#' symbolically, the element-wise inverse square root of `x` ; else, computes 
#' the element-wise inverse of R default square root function.
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param x A `LazyTensor`, a `ComplexLazyTensor`, a vector or a matrix of numeric 
#' values, or a scalar value.
#' @return An object of class "LazyTensor" if the function is called with a 
#' `LazyTensor`, and an object of class "numeric", "matrix", or "array" otherwise, 
#' same as the input class.
#' @examples
#' \dontrun{
#' x <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows and 3 columns
#' x_i <- LazyTensor(x, index = 'i')   # creating LazyTensor from matrix x, 
#'                                     # indexed by 'i'
#'                                     
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


# power ------------------------------------------------------------------------
"^.default" <- .Primitive("^") # assign default as current definition

#' Power.
#' @description
#' Symbolic binary operation for element-wise power operator.
#' @usage x^y
#' @details If `x` or `y` is a `LazyTensor`, `x^y` returns a `LazyTensor`
#' that encodes, symbolically, the element-wise value of `x` to the power of `y`.
#' (In case one of the arguments is a vector or a scalar, it is first converted to 
#' `LazyTensor`).
#' If none of the arguments is a `LazyTensor`, it is equivalent to the "^" R 
#' operator.
#' 
#' **Note**
#' 
#' \itemize{
#'     \item{if **y = 2**,}{ `x^y` relies on the `"Square"` KeOps operation;}
#'     \item{if **y = 0.5**,}{ `x^y` uses on the `"Sqrt"` KeOps operation;}
#'     \item{if **y = -0.5**,}{ `x^y` uses on the `"Rsqrt"` KeOps operation.}
#' }
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param x A `LazyTensor`, a `ComplexLazyTensor`, a vector of numeric values, 
#' or a scalar value.
#' @param y A `LazyTensor`, a `ComplexLazyTensor`, a vector of numeric values, 
#' or a scalar value.
#' @return An object of class "LazyTensor" if the function is called with a 
#' `LazyTensor`, and an object of class "numeric", otherwise.
#' @examples
#' \dontrun{
#' x <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows and 3 columns
#' y <- matrix(runif(250 * 3), 250, 3) # arbitrary R matrix, 250 rows and 3 columns
#' x_i <- LazyTensor(x, index = 'i')   # creating LazyTensor from matrix x, 
#'                                     # indexed by 'i'
#' y_j <- LazyTensor(y, index = 'j')   # creating LazyTensor from matrix y, 
#'                                     # indexed by 'j'
#'                                     
#' x_pow_y <- x_i^y_j                  # symbolic matrix
#' }
#' @export
"^" <- function(x, y) { 
    if(!is.LazyTensor(x))
        UseMethod("^", y)
    else
        UseMethod("^", x)
}

"^.LazyTensor" <- function(x, y) {   
    if(is.numeric(y) && length(y) == 1){
        if(is.int(y)){
            if(y == 2)
                res <- unaryop.LazyTensor(x, "Square")
            else
                res <- unaryop.LazyTensor(x, "Pow", y)
        }
        else if(y == 0.5)
            res <- unaryop.LazyTensor(x, "Sqrt") # element-wise square root
        else if(y == (-0.5))
            res <- unaryop.LazyTensor(x, "Rsqrt") # element-wise inverse square root
        # check if Powf with y a float number has to be like Powf(var1,var2) 
        # or Powf(var,y) (Powf(var, 0.5))
        else {
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
#' @details If `x` or `y` is a `LazyTensor`, `(x|y)` (or `x | y`) returns a 
#' `LazyTensor` that encodes, symbolically, the Euclidean scalar product between 
#' `x` and `y`, which must have the same shape. (In case one of the arguments is 
#' a vector or a scalar, it is first converted to `LazyTensor`).
#' If none of the arguments is a `LazyTensor`, is equivalent to the "|" R operator.
#'
#' **Note**
#'
#' `x` and `y` input arguments should have the same inner dimension.
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param x A `LazyTensor`, a `ComplexLazyTensor`, a vector of numeric values, 
#' or a scalar value.
#' @param y A `LazyTensor`, a `ComplexLazyTensor`, a vector of numeric values, 
#' or a scalar value.
#' @return An object of class "LazyTensor" if the function is called with a 
#' `LazyTensor`, and an object of class "numeric", otherwise.
#' @examples
#' \dontrun{
#' x <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows and 3 columns
#' y <- matrix(runif(250 * 3), 250, 3) # arbitrary R matrix, 250 rows and 3 columns
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

"|.LazyTensor" <- function(x, y) {
    res <- binaryop.LazyTensor(x, y, "|", is_operator = TRUE,
                               dim_check_type = "same", dim_res = 1)
    res$formula <- paste("(", res$formula, ")", sep = "")
    return(res)
}


# Matrix product ---------------------------------------------------------------

"%*%.default" <- .Primitive("%*%") # assign default as current definition

#' Matrix multiplication.
#' @description
#' Symbolic binary operation for element-wise matrix multiplication operator.
#' @usage x %*% y
#' @details If `x` or `y` is a `LazyTensor`, `x %*% y` returns the sum reduction 
#' of the product `x * y`.
#' If none of the arguments is a `LazyTensor`, is equivalent to the "%*%" R operator.
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param x A `LazyTensor`, a `ComplexLazyTensor`, a vector or a matrix of numeric 
#' values, or a scalar value.
#' @param y A `LazyTensor`, a `ComplexLazyTensor`, a vector or a matrix of numeric 
#' values, or a scalar value.
#' @return A matrix.
#' @examples
#' \dontrun{
#' x <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows and 3 columns
#' y <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows and 3 columns
#' x_i <- LazyTensor(x, index = 'i')   # creating LazyTensor from matrix x, 
#'                                     # indexed by 'i'
#' y_j <- LazyTensor(y, index = 'j')   # creating LazyTensor from matrix y, 
#'                                     # indexed by 'j'
#' x_mult_y <- x_i %*% y_j             
#' }
#' @export
"%*%" <- function(x, y) { 
    if(!is.LazyTensor(x))
        UseMethod("%*%", y)
    else
        UseMethod("%*%", x)
}

"%*%.LazyTensor" <- function(x, y) {
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
#'     \item{`x` is a `LazyTensor`,}{ `exp(x)` returns a `LazyTensor` that encodes, 
#'     symbolically, the element-wise exponential of `x`;}
#'     \item{`x` is a `ComplexLazyTensor`,}{ `exp(x)` returns a `ComplexLazyTensor` 
#'     that encodes, symbolically, the element-wise complex exponential of `x`;}
#'     \item{else,}{ `exp(x)` applies R default exponential to `x`.}
#' }
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param x A `LazyTensor`, a `ComplexLazyTensor`, a vector of numeric values, 
#' or a scalar value.
#' @return An object of class "LazyTensor" if the function is called with a 
#' `LazyTensor`, and an object of class "numeric", "matrix", or "array" otherwise, 
#' depending on the input class (see R default `exp()` function).
#' @examples
#' \dontrun{
#' # basic example
#' x <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows and 3 columns
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
exp <- function(x) {
    UseMethod("exp")
}

exp.LazyTensor <- function(x) {
    res <- unaryop.LazyTensor(x, "Exp")
    return(res)
}

exp.ComplexLazyTensor <- function(x) {
    res <- unaryop.LazyTensor(x, "ComplexExp")
}


# logarithm --------------------------------------------------------------------
log.default <- .Primitive("log")

#' Element-wise natural logarithm.
#' @description
#' Symbolic unary operation for element-wise natural logarithm.
#' @details If `x` is a `LazyTensor`, `exp(x)` returns a `LazyTensor` that encodes, 
#' symbolically, the element-wise natural logarithm of `x` ; 
#' else, computes R default logarithm.
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param x A `LazyTensor`, a `ComplexLazyTensor`, a vector of numeric values, 
#' or a scalar value.
#' @return An object of class "LazyTensor" if the function is called with a 
#' `LazyTensor`, and an object of class "numeric", "matrix", or "array" otherwise, 
#' depending on the input class (see R default `log()` function).
#' @examples
#' \dontrun{
#' x <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows and 3 columns
#' x_i <- LazyTensor(x, index = 'i')   # creating LazyTensor from matrix x, 
#'                                     # indexed by 'i'
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


# inverse ----------------------------------------------------------------------
inv.default <- function(x) {
    res <- 1 / x
    return(res)
}

#' Element-wise 1/x inverse.
#' @description
#' Symbolic unary operation for element-wise inverse.
#' @details If `x` is a `LazyTensor`, `exp(x)` returns a `LazyTensor` that 
#' encodes, symbolically, the element-wise inverse of `x` ; else, computes R 
#' default inverse.
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param x A `LazyTensor`, a `ComplexLazyTensor`, a vector of numeric values,
#' or a scalar value.
#' @return An object of class "LazyTensor" if the function is called with a 
#' `LazyTensor`, and an object of class "numeric", "matrix", or "array" otherwise, 
#' same as the input class.
#' @examples
#' \dontrun{
#' x <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows and 3 columns
#' x_i <- LazyTensor(x, index = 'i')   # creating LazyTensor from matrix x, 
#'                                     # indexed by 'i'
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


# cosine -----------------------------------------------------------------------
cos.default <- .Primitive("cos")

#' Element-wise cosine.
#' @description
#' Symbolic unary operation for element-wise cosine.
#' @details If `x` is a `LazyTensor`, `exp(x)` returns a `LazyTensor` that encodes, 
#' symbolically, the element-wise cosine of `x` ; else, computes R default cosine.
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param x A `LazyTensor`, a `ComplexLazyTensor`, a vector of numeric values, 
#' or a scalar value.
#' @return An object of class "LazyTensor" if the function is called with a 
#' `LazyTensor`, and an object of class "numeric", "matrix", or "array" otherwise, 
#' depending on the input class (see R default `cos()` function).
#' @examples
#' \dontrun{
#' x <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows and 3 columns
#' x_i <- LazyTensor(x, index = 'i')   # creating LazyTensor from matrix x, 
#'                                     # indexed by 'i'
#'                                     
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


# sine -------------------------------------------------------------------------
sin.default <- .Primitive("sin")

#' Element-wise sine.
#' @description
#' Symbolic unary operation for element-wise sine.
#' @details If `x` is a `LazyTensor`, `sin(x)` returns a `LazyTensor` that encodes, 
#' symbolically, the element-wise sine of `x`; else, computes R default sine function.
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param x A `LazyTensor`, a `ComplexLazyTensor`, a vector or a matrix of numeric 
#' values, or a scalar value.
#' @return An object of class "LazyTensor" if the function is called with a 
#' `LazyTensor`, and an object of class "numeric", "matrix", or "array" otherwise, 
#' depending on the input class (see R default `sin()` function).
#' @examples
#' \dontrun{
#' x <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows and 3 columns
#' x_i <- LazyTensor(x, index = 'i')   # creating LazyTensor from matrix x, 
#'                                     # indexed by 'i'
#'                                     
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


# arc-cosine --------------------------------------------------------------------
acos.default <- .Primitive("acos")

#' Element-wise arc-cosine.
#' @description
#' Symbolic unary operation for element-wise arc-cosine.
#' @details If `x` is a `LazyTensor`, `acos(x)` returns a `LazyTensor` that encodes, 
#' symbolically, the element-wise arc-cosine of `x` ; else, computes R default 
#' arc-cosine function.
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param x A `LazyTensor`, a `ComplexLazyTensor`, a vector or a matrix of numeric 
#' values, or a scalar value.
#' @return An object of class "LazyTensor" if the function is called with a 
#' `LazyTensor`, and an object of class "numeric", "matrix", or "array" otherwise, 
#' depending on the input class (see R default `acos()` function).
#' @examples
#' \dontrun{
#' x <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows and 3 columns
#' x_i <- LazyTensor(x, index = 'i')   # creating LazyTensor from matrix x, 
#'                                     # indexed by 'i'
#'                                     
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


# arc-sine ----------------------------------------------------------------------
asin.default <- .Primitive("asin")

#' Element-wise arc-sine.
#' @description
#' Symbolic unary operation for element-wise arc-sine.
#' @details If `x` is a `LazyTensor`, `asin(x)` returns a `LazyTensor` that encodes, 
#' symbolically, the element-wise arc-sine of `x` ; else, computes R default 
#' arc-sine function.
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param x A `LazyTensor`, a `ComplexLazyTensor`, a vector or a matrix of numeric 
#' values, or a scalar value.
#' @return An object of class "LazyTensor" if the function is called with a 
#' `LazyTensor`, and an object of class "numeric", "matrix", or "array" otherwise, 
#' depending on the input class (see R default `asin()` function).
#' @examples
#' \dontrun{
#' x <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows and 3 columns
#' x_i <- LazyTensor(x, index = 'i')   # creating LazyTensor from matrix x, 
#'                                     # indexed by 'i'
#'                                     
#' Asin_x <- asin(x_i)                 # symbolic matrix, 150 rows and 3 columns
#' }
#' @export
asin <- function(x) {
    UseMethod("asin")
}

asin.LazyTensor <- function(x) {
    res <- unaryop.LazyTensor(x, "Asin")
    return(res)
}


# arc-tangent -------------------------------------------------------------------
atan.default <- .Primitive("atan")

#' Element-wise arc-tangent.
#' @description
#' Symbolic unary operation for element-wise arc-tangent.
#' @details If `x` is a `LazyTensor`, `atan(x)` returns a `LazyTensor` that encodes, 
#' symbolically, the element-wise arc-tangent of `x` ; else, computes R default 
#' arc-tangent function.
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param x A `LazyTensor`, a `ComplexLazyTensor`, a vector or a matrix of numeric
#' values, or a scalar value.
#' @return An object of class "LazyTensor" if the function is called with a 
#' `LazyTensor`, and an object of class "numeric", "matrix", or "array" otherwise, 
#' depending on the input class (see R default `atan()` function).
#' @examples
#' \dontrun{
#' x <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows and 3 columns
#' x_i <- LazyTensor(x, index = 'i')   # creating LazyTensor from matrix x, 
#'                                     # indexed by 'i'
#'                                     
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


# arc-tan2 ---------------------------------------------------------------------
atan2.default <- function(x, y) {
    .Internal(atan2(x, y))
}

#' Element-wise 2-argument arc-tangent.
#' @description
#' Symbolic binary operation for element-wise 2-argument arc-tangent function.
#' @details If `x` or `y` is a `LazyTensor`, `atan2(x, y)` returns a `LazyTensor` 
#' that encodes, symbolically, the element-wise atan2 of `x` and `y`. 
#' (In case one of the arguments is a vector or a scalar, it is first converted 
#' to LazyTensor). 
#' If none of the arguments is a LazyTensor, it computes R default atan2 function.
#' 
#' **Note**
#' 
#' `x` and `y` input arguments should have the same inner dimension.
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param x A `LazyTensor`, a `ComplexLazyTensor`, a vector of numeric values, 
#' or a scalar value.
#' @param y A `LazyTensor`, a `ComplexLazyTensor`, a vector of numeric values, 
#' or a scalar value.
#' @return An object of class "LazyTensor" if the function is called with a 
#' `LazyTensor`, and an object of class "numeric", "matrix", or "array" otherwise, 
#' depending on the input class (see R default `atan2()` function).
#' @examples
#' \dontrun{
#' x <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows and 3 columns
#' y <- matrix(runif(250 * 3), 250, 3) # arbitrary R matrix, 250 rows and 3 columns
#' x_i <- LazyTensor(x, index = 'i')   # creating LazyTensor from matrix x, 
#'                                     # indexed by 'i'
#' y_j <- LazyTensor(y, index = 'j')   # creating LazyTensor from matrix y, 
#'                                     # indexed by 'j'
#'                                     
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


# absolute value ---------------------------------------------------------------
abs.default <- .Primitive("abs")

#' Element-wise absolute value.
#' @description
#' Symbolic unary operation for element-wise absolute value.
#' @details If `x` is a `LazyTensor`, `abs(x)` returns a `LazyTensor` that encodes, 
#' symbolically, the element-wise absolute value of `x` ; else, computes R default 
#' absolute value function. If `x` is a `ComplexLazyTensor`, `abs(x)` returns a 
#' `LazyTensor` that encodes, symbolically, the modulus of `x` ; 
#' else, computes R default absolute value function.
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param x A `LazyTensor`, a `ComplexLazyTensor`, a vector or a matrix of numeric 
#' values, or a scalar value.
#' @return An object of class "LazyTensor" if the function is called with a 
#' `LazyTensor`, and an object of class "numeric", "matrix", or "array" otherwise, 
#' depending on the input class (see R default `abs()` function).
#' @examples
#' \dontrun{
#' x <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows and 3 columns
#' x_i <- LazyTensor(x, index = 'i')   # creating LazyTensor from matrix x, 
#'                                     # indexed by 'i'
#'                                     
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
    res <- unaryop.LazyTensor(x, "ComplexAbs", res_type = "LazyTensor")
    return(res)
}


# sign function ----------------------------------------------------------------
sign.default <- .Primitive("sign")

#' Element-wise sign.
#' @description
#' Symbolic unary operation for element-wise sign.
#' @details If `x` is a `LazyTensor`, `sign(x)` returns a `LazyTensor` that encodes, 
#' symbolically, the element-wise sign of `x` in {-1, 0, +1} ; else, computes R 
#' default sign function.
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param x A `LazyTensor`, a `ComplexLazyTensor`, a vector or a matrix of numeric
#' values, or a scalar value.
#' @return An object of class "LazyTensor" if the function is called with a 
#' `LazyTensor`, and an object of class "numeric", "matrix", or "array" otherwise, 
#' depending on the input class (see R default `sign()` function).
#' @examples
#' \dontrun{
#' x <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows and 3 columns
#' x_i <- LazyTensor(x, index = 'i')   # creating LazyTensor from matrix x, 
#'                                     # indexed by 'i'
#'                                     
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


# round function ---------------------------------------------------------------
round.default <- .Primitive("round")

#' Element-wise rounding function.
#' @description
#' Symbolic binary operation for element-wise rounding function.
#' @details If `x` is a `LazyTensor`, `round(x, d)` returns a `LazyTensor` that 
#' encodes, symbolically, the element-wise rounding of `x` to `d` decimal places ; 
#' else, computes R default rounding function.
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param x A `LazyTensor`, a `ComplexLazyTensor`, a vector or a matrix of numeric
#' values, or a scalar value.
#' @param d A scalar value. (or a complex ?)
#' @return An object of class "LazyTensor" if the function is called with a 
#' `LazyTensor`, and an object of class "numeric", "matrix", or "array" otherwise, 
#' depending on the input class (see R default `round()` function).
#' @examples
#' \dontrun{
#' x <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows and 3 columns
#' x_i <- LazyTensor(x, index = 'i')   # creating LazyTensor from matrix x, 
#'                                     # indexed by 'i'
#'                                     
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
#' @details If `x` is a `LazyTensor`, `xlogx(x)` returns a `LazyTensor` that 
#' encodes, symbolically, the element-wise `x` times logarithm of `x` 
#' (with value 0 at 0); else, computes `x * log(x)`.
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param x A `LazyTensor`, a `ComplexLazyTensor`, a vector or a matrix of numeric 
#' values, or a scalar value.
#' @return An object of class "LazyTensor" if the function is called with a 
#' `LazyTensor`, and an object of class "numeric", "matrix", or "array" otherwise, 
#' depending on the input class.
#' @examples
#' \dontrun{
#' x <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows and 3 columns
#' x_i <- LazyTensor(x, index = 'i')   # creating LazyTensor from matrix x, 
#'                                     # indexed by 'i'
#'                                     
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
#' @details If `x` is a `LazyTensor`, `xlogx(x)` returns a `LazyTensor` that 
#' encodes, symbolically, the element-wise sin(x)/x function of `x` (with value 
#' 0 at 0); else, computes `sin(x) / x`.
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param x A `LazyTensor`, a `ComplexLazyTensor`, a vector or a matrix of numeric 
#' values, or a scalar value.
#' @return An object of class "LazyTensor" if the function is called with a 
#' `LazyTensor`, and an object of class "numeric", "matrix", or "array" otherwise, 
#' depending on the input class.
#' @examples
#' \dontrun{
#' x <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows and 3 columns
#' x_i <- LazyTensor(x, index = 'i')   # creating LazyTensor from matrix x, 
#'                                     # indexed by 'i'
#'                                     
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


# step function ----------------------------------------------------------------
step.default <- function(x, ...){
    res <- stats::step(x, ...)
}

#' Element-wise step function.
#' @description
#' Symbolic unary operation for element-wise step function.
#' @details If `x` is a `LazyTensor`, `step(x)` returns a `LazyTensor` that
#' encodes, symbolically, the element-wise step function of `x` 
#' (`0` if `x < 0`, `1` if `x >= 0`); else, computes R default 
#' step function with other specific arguments (see R default `step()` function).
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param x A `LazyTensor`, a `ComplexLazyTensor`, a vector or a matrix
#' of numeric values, or a scalar value.
#' @return An object of class "LazyTensor" if the function is called with a 
#' `LazyTensor`, and an object of class "numeric", "matrix", or "array" otherwise, 
#' depending on the input class (see R default `stats::step()` function).
#' @examples
#' \dontrun{
#' x <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows and 3 columns
#' x_i <- LazyTensor(x, index = 'i')   # creating LazyTensor from matrix x, 
#'                                     # indexed by 'i'
#'                                     
#' Step_x <- step.LazyTensor(x_i)      # symbolic matrix, 150 rows and 3 columns
#' }
#' @export
step <- function(x, ...){
    UseMethod("step", x)
}

step.LazyTensor <- function(x) {
    res <- unaryop.LazyTensor(x, "Step")
    return(res)
}


# relu function ----------------------------------------------------------------

#' Element-wise ReLU function.
#' @description
#' Symbolic unary operation for element-wise ReLU function.
#' @details `relu(x)` returns a `LazyTensor` that encodes, symbolically,
#' the element-wise ReLU of `x` (`0` if `x < 0`, `x` if `x >= 0`).
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param x A `LazyTensor`, a `ComplexLazyTensor`, a vector of numeric values, 
#' or a scalar value.
#' @return An object of class "LazyTensor".
#' @examples
#' \dontrun{
#' x <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows and 3 columns
#' x_i <- LazyTensor(x, index = 'i')   # creating LazyTensor from matrix x, 
#'                                     # indexed by 'i'
#'                                     
#' ReLU_x <- relu(x_i)                 # symbolic matrix, 150 rows and 3 columns
#' }
#' @export
relu <- function(x) {
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
#' If `a` and `b` are not scalar values, these should have the same inner 
#' dimension as `x`.
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param x A `LazyTensor`, a `ComplexLazyTensor`, a vector of numeric values, 
#' or a scalar value.
#' @param a A `LazyTensor`, a `ComplexLazyTensor`, a vector of numeric values, 
#' or a scalar value.
#' @param b A `LazyTensor`, a `ComplexLazyTensor`, a vector of numeric values, 
#' or a scalar value.
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
    if(is.int(a) && is.int(b))
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
#' the element-wise clamping of `x` in `(y, z)` which are integers. See `?clamp`
#' for more details.
#' Broadcasting rules apply.
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param x A `LazyTensor`, a `ComplexLazyTensor`, a vector of numeric values, 
#' or a scalar value.
#' @param y An `integer`.
#' @param z An `integer`.
#' @return An object of class "LazyTensor".
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


# ifelse function --------------------------------------------------------------
ifelse.default <- function(x, a, b) {
    res <- base::ifelse(x, a, b)
    return(res)
}


#' Element-wise if-else function.
#' @description
#' Symbolic ternary operation for element-wise if-else function.
#' @details If `x` is a `LazyTensor`, `ifelse(x, a, b)` returns a `LazyTensor` 
#' that encodes, symbolically,
#' `a` if `x >= 0` and `b` if ``x < 0``.  Broadcasting rules apply. 
#' `a` and `b` may be fixed integers or floats, or other `LazyTensor`.
#' 
#' Else, computes R default if-else function.
#' 
#' **Note**
#' 
#' If `a` and `b` are not scalar values, these should have the same inner 
#' dimension as `x`.
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param x A `LazyTensor`, a `ComplexLazyTensor`, a vector of numeric values, 
#' or a scalar value.
#' @param a A `LazyTensor`, a `ComplexLazyTensor`, a vector of numeric values, 
#' or a scalar value.
#' @param b A `LazyTensor`, a `ComplexLazyTensor`, a vector of numeric values, 
#' or a scalar value.
#' @return An object of class "LazyTensor" if the function is called with a 
#' `LazyTensor`, and an object of class "numeric", "matrix", or "array" otherwise, 
#' depending on the input class (see R default `base::ifelse()` function).
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
ifelse <- function(x, a, b) {
    UseMethod("ifelse", x)
}

ifelse.LazyTensor <- function(x, a, b) {
    res <- ternaryop.LazyTensor(x, a, b, "IfElse")
}


# mod function -----------------------------------------------------------------

#' Element-wise modulo with offset function.
#' @description
#' Symbolic ternary operation for element-wise modulo with offset function.
#' @details `mod(x, a, b)` returns a `LazyTensor` that encodes, symbolically,
#' the element-wise modulo of `x` with modulus `a` and offset `b`. That is,
#' `mod(x, a, b)` encodes symbolically `x - a * floor((x - b)/a)`.
#' By default `b = 0`, so that `mod(x, a)` becomes equivalent to the R 
#' function `%%`.
#' `a` and `b` may be fixed integers or floats, or other `LazyTensor`.
#' Broadcasting rules apply.
#' 
#' **Note**
#' 
#' If `a` and `b` are not scalar values, these should have the same inner 
#' dimension as `x`.
#' 
#' **Warning**
#' 
#' Do not confuse with `Mod()`.
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param x A `LazyTensor`, a `ComplexLazyTensor`, a vector of numeric values, 
#' or a scalar value.
#' @param a A `LazyTensor`, a vector of numeric values, or a scalar value.
#' @param b A `LazyTensor`, a vector of numeric values, or a scalar value.
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

mod.LazyTensor <- function(x, a, b = 0) {
    res <- ternaryop.LazyTensor(x, a, b, "Mod")
}



# Squared Euclidean norm -------------------------------------------------------

#' Squared Euclidean norm.
#' @description
#' Symbolic unary operation for squared Euclidean norm.
#' @details `sqnorm2(x)` returns a `LazyTensor` that encodes, symbolically,
#' the squared Euclidean norm of `x`, same as `(x|x)`.
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param x A `LazyTensor`, a `ComplexLazyTensor`, a vector of numeric values, 
#' or a scalar value.
#' @return An object of class "LazyTensor".
#' @examples
#' \dontrun{
#' x <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows, 3 columns
#' x_i <- LazyTensor(x, index = 'i')   # creating LazyTensor from matrix x, 
#'                                     # indexed by 'i'
#'                                     
#' SqN_x <- sqnorm2(x_i)               # symbolic matrix, 150 rows and 3 columns
#' }
#' @export
sqnorm2 <- function(x) {
    res <- unaryop.LazyTensor(x, "SqNorm2",
                              res_type = "LazyTensor",
                              dim_res = 1)
    return(res)
}


# Euclidean norm ---------------------------------------------------------------

#' Euclidean norm.
#' @description
#' Symbolic unary operation for Euclidean norm.
#' @details `norm2(x)` returns a `LazyTensor` that encodes, symbolically,
#' the Euclidean norm of `x`, same as `sqrt(x|x)`.
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param x A `LazyTensor`, a `ComplexLazyTensor`, a vector of numeric values, 
#' or a scalar value.
#' @return An object of class "LazyTensor".
#' @examples
#' \dontrun{
#' x <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows and 3 columns
#' x_i <- LazyTensor(x, index = 'i')   # creating LazyTensor from matrix x, 
#'                                     # indexed by 'i'
#' N_x <- norm2(x_i)                   # symbolic matrix, 150 rows and 3 columns
#' }
#' @export
norm2 <- function(x) {
    res <- unaryop.LazyTensor(x, "Norm2",
                              res_type = "LazyTensor",
                              dim_res = 1)
    return(res)
}


# Vector normalization ---------------------------------------------------------

#' Vector normalization.
#' @description
#' Symbolic unary operation for vector normalization.
#' @details `normalize(x)` returns a `LazyTensor` that encodes, symbolically,
#' the vector normalization of `x` (with the Euclidean norm),
#' same as `rsqrt(sqnorm2(x)) * x`.
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param x A `LazyTensor`, a `ComplexLazyTensor`, a vector of numeric values, 
#' or a scalar value.
#' @return An object of class "LazyTensor".
#' @examples
#' \dontrun{
#' x <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows and 3 columns
#' x_i <- LazyTensor(x, index = 'i')   # creating LazyTensor from matrix x, 
#'                                     # indexed by 'i'
#'                                     
#' N_x <- norm2(x_i)                   # symbolic matrix, 150 rows and 3 columns
#' }
#' @export
normalize <- function(x) {
    res <- unaryop.LazyTensor(x, "Normalize")
    return(res)
}



# Squared distance -------------------------------------------------------------

#' Squared distance.
#' @description
#' Symbolic binary operation for vector normalization.
#' @details `sqdist(x)` returns a `LazyTensor` that encodes, symbolically,
#' the squared Euclidean distance between `x` and `y`, same as `sqnorm2(x - y)`.
#' 
#' **Note**
#' 
#' `x` and `y` input arguments should have the same inner dimension or be of 
#' dimension 1.
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param x A `LazyTensor`, a `ComplexLazyTensor`, a vector of numeric values, 
#' or a scalar value.
#' @param y A `LazyTensor`, a `ComplexLazyTensor`, a vector of numeric values, 
#' or a scalar value.
#' @return An object of class "LazyTensor".
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
    res <- binaryop.LazyTensor(x, y, "SqDist",
                               res_type = "LazyTensor",
                               dim_res = 1)
    return(res)
}


# Weighted squared norm --------------------------------------------------------

#' Generic squared Euclidean norm.
#' @description
#' Symbolic binary operation for weighted squared norm of a LazyTensor.
#' @details `weightedsqnorm(x)` returns a `LazyTensor` that encodes, symbolically,
#' the weighted squared norm of a vector `x` with weights stored in the LazyTensor 
#' `s`.
#' Run `browseVignettes("rkeops")` to access the vignettes and find details
#' about the function in "RKeOps LazyTensor".
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param x A `vector` of numeric values or a scalar value.
#' @param s A `LazyTensor`, a `ComplexLazyTensor`, a vector of numeric values, 
#' or a scalar value.
#' @return An object of class "LazyTensor".
#' @examples
#' \dontrun{
#' x <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows, 3 columns
#' y <- matrix(runif(100 * 3), 100, 3) # arbitrary R matrix, 100 rows, 3 columns
#' x_i <- LazyTensor(x, index = 'i')   # creating LazyTensor from matrix x, 
#'                                     # indexed by 'i'
#' y_j <- LazyTensor(y, index = 'j')   # creating LazyTensor from matrix y, 
#'                                     # indexed by 'j'
#' 
#' wsqn_xy <- weightedsqnorm(x_i, y_j)
#' }
#' @export
weightedsqnorm <- function(x, s) {
    res <- binaryop.LazyTensor(x, s, "WeightedSqNorm",
                               dim_check_type = NA,
                               res_type = "LazyTensor",
                               dim_res = 1)
    return(res)
}


# Weighted squared distance ----------------------------------------------------


#' Generic squared distance.
#' @description
#' Symbolic binary operation for weighted squared distance of a LazyTensor.
#' @details `weightedsqdist(x)` returns a `LazyTensor` that encodes, symbolically,
#' the weighted squared distance of a vector `x` with weights stored in the 
#' LazyTensor `s`, same as `weightedsqnorm(x - y, s)`.
#' Run `browseVignettes("rkeops")` to access the vignettes and find details
#' about the function in "RKeOps LazyTensor".
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param x A `vector` of numeric values or a scalar value.
#' @param y A `LazyTensor`, a `ComplexLazyTensor`, a vector of numeric values, 
#' or a scalar value.
#' @param z A `LazyTensor`, a `ComplexLazyTensor`, a vector of numeric values, 
#' or a scalar value.
#' @return An object of class "LazyTensor".
#' @examples
#' \dontrun{
#' x <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows, 3 columns
#' y <- matrix(runif(100 * 3), 100, 3) # arbitrary R matrix, 100 rows, 3 columns
#' z <- matrix(runif(200 * 3), 100, 3) # arbitrary R matrix, 200 rows, 3 columns
#' 
#' x_i <- LazyTensor(x, index = 'i')   # creating LazyTensor from matrix x, 
#'                                     # indexed by 'i'
#' y_j <- LazyTensor(y, index = 'j')   # creating LazyTensor from matrix y, 
#'                                     # indexed by 'j'
#' z_i <- LazyTensor(z, index = 'i')   # creating LazyTensor from matrix z, 
#'                                     # indexed by 'i'                                                                        
#' 
#' wsqd_xy <- weightedsqdist(x_i, y_j, z_i)    # symbolic matrix
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
#' @details If `z` is a `ComplexLazyTensor`, `Re(z)` returns a `ComplexLazyTensor` 
#' that encodes, symbolically, the element-wise real part of complex `z` ; 
#' else, computes R default `Re()` function.
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param z A `ComplexLazyTensor` or any type of values accepted by R default 
#' `Re()` function.
#' @return An object of class "ComplexLazyTensor" if the function is called with 
#' a `ComplexLazyTensor`, else see R default `Re()` function.
#' @examples
#' \dontrun{
#' z <- matrix(2 + 1i^ (-6:5), nrow = 4)        # complex matrix
#' z_i <- LazyTensor(x, "i", is_complex = TRUE) # creating ComplexLazyTensor
#' 
#' Re_z <- Re(z_i)                              # symbolic matrix
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
#' @details If `z` is a `ComplexLazyTensor`, `Im(z)` returns a `ComplexLazyTensor` 
#' that encodes, symbolically, the element-wise imaginary part of complex `z` ; 
#' else, computes R default `Im()` function.
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param z A `ComplexLazyTensor` or any type of values accepted by R default 
#' `Im()` function.
#' @return An object of class "ComplexLazyTensor" if the function is called with 
#' a `ComplexLazyTensor`, else see R default `Im()` function.
#' @examples
#' \dontrun{
#' z <- matrix(2 + 1i^ (-6:5), nrow = 4)         # complex matrix
#' z_i <- LazyTensor(z, "i", is_complex = TRUE)  # creating ComplexLazyTensor
#' 
#' Im_z <- Im(z_i)                               # symbolic matrix
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
Arg.default <- .Primitove("Arg")

#' Element-wise angle (or argument) of complex.
#' @description
#' Symbolic unary operation for element-wise angle (or argument) of complex.
#' @details If `z` is a `ComplexLazyTensor`, `Arg(z)` returns a `ComplexLazyTensor` 
#' that encodes, symbolically, the element-wise angle (or argument) of 
#' complex `z` ; else, computes R default `Arg()` function.
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param z A `ComplexLazyTensor` or any type of values accepted by R default 
#' `Arg()` function.
#' @return An object of class "ComplexLazyTensor" if the function is called with 
#' a `ComplexLazyTensor`, else see R default `Arg()` function.
#' @examples
#' \dontrun{
#' z <- matrix(2 + 1i^ (-6:5), nrow = 4)         # complex matrix
#' z_i <- LazyTensor(z, "i", is_complex = TRUE)  # creating ComplexLazyTensor
#' 
#' Arg_z <- Arg(z_i)                             # symbolic matrix
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


# real to complex --------------------------------------------------------------

#' Element-wise "real 2 complex" operation.
#' @description
#' Symbolic unary operation for element-wise "real 2 complex".
#' @details `real2complex(x)` returns a `ComplexLazyTensor` that encodes, 
#' symbolically, the element-wise "real 2 complex" of `x`.
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param x A `LazyTensor`.
#' @return An object of class "ComplexLazyTensor".
#' @examples
#' \dontrun{
#' x <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows, 3 columns
#' x_i <- LazyTensor(x, "i")           # creating LazyTensor from matrix x, 
#'                                     # indexed by 'i'
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


# imaginary to complex ---------------------------------------------------------

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
#' x <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows, 3 columns
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
    res <- unaryop.LazyTensor(x, "Imag2Complex", res_type = "ComplexLazyTensor",
                              dim_res = 2 * x$dimres)
}

imag2complex.ComplexLazyTensor <- function(x) {
    stop("`imag2complex` cannot be applied to a complex LazyTensor.")
}


# complex exponential of 1j x --------------------------------------------------

#' Element-wise "complex exponential of 1j x" operation.
#' @description
#' Symbolic unary operation for element-wise "complex exponential of 1j x".
#' @details `exp1j(x)` returns a `ComplexLazyTensor` that encodes, symbolically,
#' the multiplication of `1j` with `x`.
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param x A `LazyTensor`.
#' @return An object of class "ComplexLazyTensor".
#' @examples
#' \dontrun{
#' x <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows, 3 columns
#' x_i <- LazyTensor(x, "i")           # creating LazyTensor from matrix x, 
#'                                     # indexed by 'i'
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
#' @details If `z` is a `ComplexLazyTensor`, `Conj(z)` returns a `ComplexLazyTensor` 
#' that encodes symbolically, the element-wise complex conjugate of `z` ; 
#' else, computes R default `Conj()` function.
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param z A `ComplexLazyTensor`, or any type of values accepted by R default 
#' `Conj()` function.
#' @return A `ComplexLazyTensor`.
#' @examples
#' \dontrun{
#' z <- matrix(1i^ (-6:5), nrow = 4)                     # complex 4x3 matrix
#' z_i <- LazyTensor(z, index = 'i', is_complex = TRUE)  # ComplexLazyTensor
#' 
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
    res <- unaryop.LazyTensor(z, "Conj")
}


# complex modulus --------------------------------------------------------------
Mod.default <- .Primitive("Mod") # assign default as current definition

#' Element-wise absolute value (or modulus).
#' @description
#' Symbolic unary operation for element-wise absolute value (or modulus).
#' @details If `z` is a `ComplexLazyTensor`, `Mod(z)` returns a `LazyTensor` that 
#' encodes, symbolically, the element-wise absolute value (or modulus) of `z` ; 
#' else, computes R default `Mod()` function.
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param z A `ComplexLazyTensor`, or any type of values accepted by R default 
#' `Mod()` function.
#' @return A `LazyTensor`.
#' @examples
#' \dontrun{
#' z <- matrix(1i^ (-6:5), nrow = 4)                     # complex 4x3 matrix
#' z_i <- LazyTensor(z, index = 'i', is_complex = TRUE)  # ComplexLazyTensor
#' 
#' Mod_z_i <- Mod(z_i)                                   # symbolic matrix
#' }
#' @export
Mod <- function(z) { 
    UseMethod("Mod", z)
}

Mod.LazyTensor <- function(z) {
    stop("`Mod` cannot be applied to a LazyTensor. See `?Mod` for compatible types.")
}

Mod.ComplexLazyTensor <- function(z) {
    res <- unaryop.LazyTensor(z, "ComplexAbs", res_type = "LazyTensor")
}



# REDUCTIONS ===================================================================


# Reduction --------------------------------------------------------------------

#' Reduction operation.
#' @description
#' Applies a reduction to a `LazyTensor`.
#' @details `reduction.LazyTensor(x, opstr, index)` will :
#' \itemize{
#'   \item{ if **index = "i"**, return the **opstr** reduction of **x** over the 
#'   "i" indexes;}
#'   \item{ if **index = "j"**, return the **opstr** reduction of **x** over the 
#'   "j" indexes.}
#' }
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param x A `LazyTensor` or a `ComplexLazyTensor`.
#' @param opstr A `string` formula (like "Sum" or "Max").
#' @param index A `character` that should be either **i** or **j** to specify 
#' whether if the reduction is indexed by **i** (rows), or **j** (columns).
#' @param opt_arg An optional argument # TODO
#' @return A matrix corresponding to the reduction.
#' @examples
#' \dontrun{
#' x <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows, 3 columns
#' x_i <- LazyTensor(x, index = 'i')   # creating LazyTensor from matrix x, 
#'                                     # indexed by 'i'
#' 
#' red_x <- reduction.LazyTensor(x_i, "Sum", "i")
#' }
#' @export
reduction.LazyTensor <- function(x, opstr, index, opt_arg = NA) {
    if(!is.LazyTensor(x) && !is.ComplexLazyTensor(x))
        stop("`x` input should be a LazyTensor or a ComplexLazyTensor.")
    
    if(!is.character(opstr))
        stop("`opst` input should be a string text.")
    
    if(check_index(index)) {
        if(index == "i") 
            tag <- 1
        else 
            tag <- 0
        if(!any(is.na(opt_arg))) {
            if(is.LazyTensor(opt_arg)) {
                formula <- paste( opstr,  "_Reduction(",  x$formula, 
                                  ",",  tag, ",", opt_arg$formula, ")", sep = "")
            }
            
            else if(is.int(opt_arg)) {
                formula <- paste( opstr,  "_Reduction(",  x$formula, 
                                  ",",  opt_arg, ",", tag, ")", sep = "")
            }
            
            else if(is.character(opt_arg)) {
                formula <- paste( opstr,  "_Reduction(",  x$formula, 
                                  ",",  tag, ",", opt_arg, ")", sep = "")
            }
            
        }
        else {
            formula <- paste(opstr, "_Reduction(", x$formula, ",", 
                             tag, ")", sep = "")
        }
        
        args <- x$args
        op <- keops_kernel(formula, args)
        res <- op(x$vars)
    }
    
    else
        stop("`index` input argument should be a character, either 'i' or 'j'.")
    
    return(res)
}


# sum function -----------------------------------------------------------------
sum.default <- .Primitive("sum")

#' Summation operation or Sum reduction.
#' @description
#' Summation unary operation, or Sum reduction.
#' @details If `x` is a `LazyTensor`, `sum(x, index)` will :
#' \itemize{
#'   \item if **index = "i"**, return the sum reduction of **x** over the "i" 
#'   indexes.
#'   \item if **index = "j"**, return the sum reduction of **x** over the "j" 
#'   indexes.
#'   \item if **index = NA** (default), return a new `LazyTensor` object 
#'   representing the sum of the values of the vector.
#' }
#' If `x` is not a `LazyTensor` it computes R default "sum" function with
#' other specific arguments (see R default `sum()` function).
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param x A `LazyTensor`, a `ComplexLazyTensor`, a vector or a matrix of numeric 
#' values, or a scalar value.
#' @param index A `character` corresponding to the reduction dimension that should 
#' be either **i** or **j** to specify whether if the summation is indexed by 
#' **i** (rows), or **j** (columns).
#' It can be NA (default) when no reduction is desired.
#' @return 
#' @examples
#' \dontrun{
#' x <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows, 3 columns
#' x_i <- LazyTensor(x, index = 'i')   # creating LazyTensor from matrix x, 
#'                                     # indexed by 'i'
#' 
#' sum_x <- sum(x_i)                   # LazyTensor object
#' sum_red_x <- sum(x_i, "i")          # sum reduction indexed by 'i'
#' }
#' @export
sum <- function(x, index) {
    UseMethod("sum")
}

sum.LazyTensor <- function(x, index = NA) {
    if(is.na(index)) {
        if(is.ComplexLazyTensor(x)) {
            res <- unaryop.LazyTensor(x, "ComplexSum", dimres = 1) # dimres = 2 ??
        }
        else {
            res <- unaryop.LazyTensor(x, "Sum", dimres = 1)
        }
    }
    else if(check_index(index))
        res <- reduction.LazyTensor(x, "Sum", index)
    else
        stop(paste("`index` input argument should be a character,",
                   " either 'i' or 'j', or NA.", sep = ""))
    return(res)
}


# sum reduction ----------------------------------------------------------------

#' Summation operation or Sum reduction.
#' @description
#' Summation unary operation, or Sum reduction.
#' @details `sum_reduction(x, index)` will return the sum reduction of **x** 
#' indexed by **index**.
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param x A `LazyTensor` or a `ComplexLazyTensor`.
#' @param index A `character` corresponding to the reduction dimension that should 
#' be either **i** or **j** to specify whether if the summation is indexed by 
#' **i** (rows), by **j** (columns).
#' @return A matrix corresponding to the sum reduction.
#' @seealso [rkeops::sum()]
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
        res <- reduction.LazyTensor(x, "Sum", index)
    }
    else {
        stop(paste("`index` input argument should be a character,",
                   " either 'i' or 'j', or NA.", sep = ""))
    }
    return(res)
}


# min function -----------------------------------------------------------------
min.default <- .Primitive("min")

#' Minimum.
#' @description
#' Minimum unary operation or Minimum reduction.
#' @details If `x` is a `LazyTensor`, `min(x, index)` will :
#' \itemize{
#'   \item if **index = "i"**, return the min reduction of **x** over the "i" 
#'   indexes.
#'   \item if **index = "j"**, return the min reduction of **x** over the "j" 
#'   indexes.
#'   \item if **index = NA** (default), return a new `LazyTensor` object 
#'   representing the min of the values of the vector.
#' }
#' If `x` is not a `LazyTensor` it computes R default "min" function with
#' other specific arguments (see R default `min()` function).
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param x A `LazyTensor`, a `ComplexLazyTensor`, a vector or a matrix of numeric 
#' values, or a scalar value.
#' @param index A `character` corresponding to the reduction dimension that should 
#' be either **i** or **j** to specify whether if the reduction is indexed by 
#' **i** (rows), or **j** (columns).
#' It can be NA (default) when no reduction is desired.
#' @return A matrix corresponding to the min reduction or an object of class 
#' "LazyTensor" corresponding to a symbolic matrix, otherwise, depending on the 
#' input class (see R default `min()` function).
#' @examples
#' \dontrun{
#' x <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows, 3 columns
#' x_i <- LazyTensor(x, index = 'i')   # creating LazyTensor from matrix x, 
#'                                     # indexed by 'i'
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
        stop(paste("`index` input argument should be a character,",
                   " either 'i' or 'j', or NA.", sep = ""))
    return(res)
}


# min reduction ----------------------------------------------------------------

#' Min reduction.
#' @description
#' Minimum reduction.
#' @details `min_reduction(x, index)` will return the min reduction of **x** 
#' indexed by **index**.
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param x A `LazyTensor` or a `ComplexLazyTensor`.
#' @param index A `character` corresponding to the reduction dimension that should 
#' be either **i** or **j** to specify whether if the reduction is indexed by 
#' **i** (rows), or **j** (columns).
#' @return A matrix corresponding to the min reduction.
#' @seealso [rkeops::min()]
#' @examples
#' \dontrun{
#' x <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows, 3 columns
#' x_i <- LazyTensor(x, index = 'i')   # creating LazyTensor from matrix x, 
#'                                     # indexed by 'i'
#' 
#' min_red_x <- min_reduction(x_i, "i")  # min reduction indexed by "i"
#' }
#' @export
min_reduction <- function(x, index) {
    if(check_index(index)) {
        res <- reduction.LazyTensor(x, "Min", index)
    }
    else {
        stop("`index` input argument should be a character, either 'i' or 'j'.")
    }
    return(res)
}


# argmin function --------------------------------------------------------------

#' ArgMin.
#' @description
#' ArgMin unary operation, or ArgMin reduction.
#' @details `argmin(x, index)` will :
#' \itemize{
#'   \item if **index = "i"**, return the argmin reduction of **x** over the "i" 
#'   indexes.
#'   \item if **index = "j"**, return the argmin reduction of **x** over the "j" 
#'   indexes.
#'   \item if **index = NA** (default), return a new `LazyTensor` object 
#'   representing the argmin  of the values of the vector.
#' }
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param x A `LazyTensor`, a `ComplexLazyTensor`, a vector or a matrix of numeric 
#' values, or a scalar value.
#' @param index A `character` corresponding to the reduction dimension that should 
#' be either **i** or **j** to specify whether if the reduction is indexed by 
#' **i** (rows) or **j** (columns).
#' It can be NA (default) when no reduction is desired.
#' @return A matrix corresponding to the argmin reduction or an object of class 
#' "LazyTensor" corresponding to a symbolic matrix.
#' @examples
#' \dontrun{
#' x <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows, 3 columns
#' x_i <- LazyTensor(x, index = 'i')   # creating LazyTensor from matrix x, 
#'                                     # indexed by 'i'
#' 
#' argmin_xi <- argmin(x_i, "i")  # argmin reduction indexed by "i"
#' argmin_x <- argmin(x_i)        # symbolic matrix
#' }
#' @export
argmin <- function(x, index = NA) {
    if(is.na(index))
            res <- unaryop.LazyTensor(x, "ArgMin")
    else if(check_index(index))
        res <- reduction.LazyTensor(x, "ArgMin", index)
    else
        stop(paste("`index` input argument should be a character,",
                   " either 'i' or 'j', or NA.", sep = ""))
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
#' be either **i** or **j** to specify whether if the reduction is indexed by 
#' **i** (rows) or **j** (columns).
#' @return A matrix corresponding to the argmin reduction.
#' @seealso [rkeops::argmin()]
#' @examples
#' \dontrun{
#' x <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows and 3 columns
#' x_i <- LazyTensor(x, index = 'i')   # creating LazyTensor from matrix x, 
#'                                     # indexed by 'i'
#' 
#' argmin_red <- argmin(x_i, "i")  # argmin reduction indexed by "i"
#' }
#' @export
argmin_reduction <- function(x, index) {
    if(check_index(index))
        res <- reduction.LazyTensor(x, "ArgMin", index)
    else
        stop("`index` input argument should be a character, either 'i' or 'j'.")
    return(res)
}


# min_argmin -------------------------------------------------------------------

#' Min-ArgMin.
#' @description
#' Min-ArgMin reduction.
#' @details `min_argmin(x, index)` will :
#' \itemize{
#'   \item{ if **index = "i"**, return the minimal values and its indices
#'   of **x** over the "i" indexes.}
#'   \item{ if **index = "j"**, return the minimal values and its indices
#'   of **x** over the "j" indexes.}
#' }
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param x A `LazyTensor` or a `ComplexLazyTensor`.
#' @param index A `character` corresponding to the reduction dimension that should 
#' be either **i** or **j** to specify whether if the reduction is indexed by 
#' **i** (rows) or **j** (columns).
#' @return A matrix corresponding to the min-argmin reduction.
#' @examples
#' \dontrun{
#' x <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows and 3 columns
#' x_i <- LazyTensor(x, index = 'i')   # creating LazyTensor from matrix x, 
#'                                     # indexed by 'i'
#' 
#' min_argmin_xi <- min_argmin(x_i, "i")  # min argmin reduction indexed by "i"
#' }
#' @export
min_argmin <- function(x, index) {
    if(check_index(index))
        res <- reduction.LazyTensor(x, "Min_ArgMin", index)
    else
        stop("`index` input argument should be a character, either 'i' or 'j'.")
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
#' be either **i** or **j** to specify whether if the reduction is indexed by **i** 
#' (rows) or **j** (columns).
#' @return A matrix corresponding to the min-argmin reduction.
#' @seealso [rkeops::min_argmin()]
#' @examples
#' \dontrun{
#' x <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows and 3 columns
#' x_i <- LazyTensor(x, index = 'i')   # creating LazyTensor from matrix x, 
#'                                     # indexed by 'i'
#' 
#' min_argmin_red <- min_argmin_reduction(x_i, "i")  # min reduction indexed by "i"
#' }
#' @export
min_argmin_reduction <- function(x, index) {
    if(check_index(index))
        res <- min_argmin(x, index)
    else
        stop("`index` input argument should be a character, either 'i' or 'j'.")
    return(res)
}


# max function -----------------------------------------------------------------
max.default <- .Primitive("max")

#' Maximum.
#' @description
#' Maximum unary operation, or Max reduction.
#' @details If `x` is a `LazyTensor`, `max(x, index)` will :
#' \itemize{
#'   \item if **index = "i"**, return the max reduction of **x** over the "i" 
#'   indexes.
#'   \item if **index = "j"**, return the max reduction of **x** over the "j" 
#'   indexes.
#'   \item if **index = NA** (default), return a new `LazyTensor` object
#'   representing the max of the values of the vector.
#' }
#' If `x` is not a `LazyTensor` it computes R default "max" function with
#' other specific arguments (see R default `max()` function).
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param x A `LazyTensor`, a `ComplexLazyTensor`, a vector or a matrix of numeric
#' values, or a scalar value.
#' @param index A `character` corresponding to the reduction dimension that should 
#' be either **i** or **j** to specify whether if the reduction is indexed by 
#' **i** (rows), or **j** (columns).
#' It can be NA (default) when no reduction is desired.
#' @return A matrix corresponding to the max reduction or an object of class 
#' "LazyTensor" corresponding to a symbolic matrix, otherwise, depending on the 
#' input class (see R default `max()` function).
#' @examples
#' \dontrun{
#' x <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows and 3 columns
#' x_i <- LazyTensor(x, index = 'i')   # creating LazyTensor from matrix x, 
#'                                     # indexed by 'i'
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
        stop(paste("`index` input argument should be a character,",
                   " either 'i' or 'j', or NA.", sep = ""))
    return(res)
}


# max reduction ----------------------------------------------------------------

#' Max reduction.
#' @description
#' Maximum reduction.
#' @details `max_reduction(x, index)` will return the max reduction of **x** 
#' indexed by **index**.
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param x A `LazyTensor` or a `ComplexLazyTensor`.
#' @param index A `character` corresponding to the reduction dimension that should 
#' be either **i** or **j** to specify whether if the reduction is indexed by 
#' **i** (rows) or **j** (columns).
#' @return A matrix corresponding to the max reduction or an object of class 
#' "LazyTensor" corresponding to a symbolic matrix, otherwise, depending on the 
#' input class (see R default `min()` function).
#' @seealso [rkeops::max()]
#' @examples
#' \dontrun{
#' x <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows, 3 columns
#' x_i <- LazyTensor(x, index = 'i')   # creating LazyTensor from matrix x, 
#'                                     # indexed by 'i'
#' 
#' max_red_x <- max_reduction(x_i, "i")  # max reduction indexed by "i"
#' }
#' @export
max_reduction <- function(x, index) {
    if(check_index(index))
        res <- reduction.LazyTensor(x, "Max", index)
    else 
        stop("`index` input argument should be a character, either 'i' or 'j'.")
    return(res)
}


# argmax function --------------------------------------------------------------

#' ArgMax.
#' @description
#' ArgMax unary operation, or ArgMax reduction.
#' @details If `x` is a `LazyTensor` or a `ComplexLazyTensor`,
#' `argmax(x, index)` will:
#' \itemize{
#'     \item{if **index = NA** (default),}{ return a new `LazyTensor` object
#'     representing the argmax of the values of **x**;}
#'     \item{if **index = i**,}{ return the argmax reduction of **x** over the
#'     **i** indices (rows);}
#'     \item{if **index = j**,}{ return the argmax reduction of **x** over the
#'     **i** indices (columns).}
#' }
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param x A `LazyTensor`, a `ComplexLazyTensor`, a vector or a matrix of numeric 
#' values, or a scalar value.
#' @param index A `character` corresponding to the reduction dimension that should 
#' be either **i** or **j** to specify whether if the reduction is indexed by 
#' **i** (rows) or **j** (columns).
#' It can be NA (default) when no reduction is desired.
#' @return A matrix corresponding to the argmax reduction or an object of class 
#' "LazyTensor" corresponding to a symbolic matrix.
#' @examples
#' \dontrun{
#' x <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows and 3 columns
#' x_i <- LazyTensor(x, index = 'i')   # creating LazyTensor from matrix x, 
#'                                     # indexed by 'i'
#' 
#' argmax_xi <- argmax(x_i, "i")  # argmax reduction indexed by "i"
#' argmax_x <- argmax(x_i)        # symbolic matrix
#' }
#' @export
argmax <- function(x, index = NA) {
    if(is.na(index)) {
        res <- unaryop.LazyTensor(x, "ArgMax")
    }
    else if(check_index(index))
        res <- reduction.LazyTensor(x, "ArgMax", index)
    else 
        stop(paste("`index` input argument should be a character,",
                   " either 'i' or 'j', or NA.", sep = ""))
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
#' be either **i** or **j** to specify whether if the reduction is indexed by 
#' **i** (rows) or **j** (columns).
#' @return A matrix corresponding to the argmax reduction.
#' @seealso [rkeops::argmax()]
#' @examples
#' \dontrun{
#' x <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows and 3 columns
#' x_i <- LazyTensor(x, index = 'i')   # creating LazyTensor from matrix x, 
#'                                     # indexed by 'i'
#' 
#' argmax_red <- argmax_reduction(x_i, "i")  # argmax reduction indexed by "i"
#' }
#' @export
argmax_reduction <- function(x, index) {
    if(check_index(index))
        res <- reduction.LazyTensor(x, "ArgMax", index)
    else 
        stop("`index` input argument should be a character, either 'i' or 'j'.")
    return(res)
}


# max_argmax -------------------------------------------------------------------

#' Max-ArgMax.
#' @description
#' Max-ArgMax reduction.
#' @details `max_argmax(x, index)` will :
#' \itemize{
#'   \item if **index = "i"**, return the maximal values of **x** and its
#'   indices over the **i** indices;
#'   \item if **index = "j"**, return the maximal values of **x** and its
#'   indices over the **j** indices.
#' }
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param x A `LazyTensor` or a `ComplexLazyTensor`.
#' @param index A `character` corresponding to the reduction dimension that should 
#' be either **i** or **j** to specify whether if the reduction is indexed by 
#' **i** (rows) or **j** (columns).
#' @return A matrix corresponding to the max-argmax reduction.
#' @examples
#' \dontrun{
#' x <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows and 3 columns
#' x_i <- LazyTensor(x, index = 'i')   # creating LazyTensor from matrix x, 
#'                                     # indexed by 'i'
#' 
#' max_argmax_x <- max_argmax(x_i, "i")  # max argmax reduction indexed by "i"
#' }
#' @export
max_argmax <- function(x, index) {
    if(check_index(index))
        res <- reduction.LazyTensor(x, "Max_ArgMax", index)
    else 
        stop("`index` input argument should be a character, either 'i' or 'j'.")
    return(res)
}


# max_argmax reduction -------------------------------------------------------------

#' Max-ArgMax reduction.
#' @description
#' Max-ArgMax reduction. Redirects to `max_argmax` function.
#' @details `max_argmax(x, index)` will :
#' \itemize{
#'   \item if **index = "i"**, return the maximal values of **x** and its
#'   indices over the **i** indices;
#'   \item if **index = "j"**, return the maximal values of **x** and its
#'   indices over the **j** indices.
#' }
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param x A `LazyTensor` or a `ComplexLazyTensor`.
#' @param index A `character` corresponding to the reduction dimension that should 
#' be either **i** or **j** to specify whether if the reduction is indexed by 
#' **i** (rows) or **j** (columns).
#' @return A matrix corresponding to the max-argmax reduction.
#' @seealso [rkeops::max_argmax()]
#' @examples
#' \dontrun{
#' x <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows and 3 columns
#' x_i <- LazyTensor(x, index = 'i')   # creating LazyTensor from matrix x, 
#'                                     # indexed by 'i'
#' 
#' max_argmax_red <- max_argmax_reduction(x_i, "i")  # max argmax reduction 
#'                                                   # indexed by "i"
#' }
#' @export
max_argmax_reduction <- function(x, index) {
    if(check_index(index))
        res <- max_argmax(x, index)
    else 
        stop("`index` input argument should be a character, either 'i' or 'j'.")
    return(res)
}


# Kmin -------------------------------------------------------------------------

#' Kmin.
#' @description
#' K-Min reduction.
#' @details If `x` is a `LazyTensor` or a `ComplexLazyTensor`, `Kmin(x, K, index)`
#' will:
#' \itemize{
#'     \item{if **index = i**,}{ return the **K** minimal values of **x**
#'     over the **i** indices (rows);}
#'     \item{if **index = j**,}{ return the **K** minimal values of **x**
#'     over the **j** indices (columns).}
#' }
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param x A `LazyTensor` or a `ComplexLazyTensor`.
#' @param index A `character` corresponding to the reduction dimension that should 
#' be either **i** or **j** to specify whether if the reduction is indexed by
#' **i** (rows) or **j** (columns).
#' @param K An `integer` corresponding to the  number of minimal values required.
#' @return A matrix corresponding to the Kmin reduction.
#' @examples
#' \dontrun{
#' x <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows, 3 columns
#' x_i <- LazyTensor(x, index = 'i')   # LazyTensor from matrix x, indexed by 'i'
#' K <- 2
#' kmin_x <- Kmin(x_i, K, "i")         # Kmin reduction, over the "i" indices
#' }
#' @export
Kmin <- function(x, K, index) {
    if(!is.int(K))
        stop("`K` input argument should be an integer.")
    if(!check_index(index))
        stop("`index` input argument should be a character, either 'i' or 'j'.")
    res <- reduction.LazyTensor(x, "KMin", index, opt_arg = K)
    return(res)
}


# Kmin reduction ---------------------------------------------------------------

#' Kmin reduction.
#' @description
#' K-Min reduction. Redirects to `Kmin` function.
#' @details If `x` is a `LazyTensor` or a `ComplexLazyTensor`,
#' `Kmin(x, K, index)` will:
#' \itemize{
#'     \item{if **index = i**,}{ return the **K** minimal values of **x**
#'     over the **i** indices (rows);}
#'     \item{if **index = j**,}{ return the **K** minimal values of **x**
#'     over the **j** indices (columns).}
#' }
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param x A `LazyTensor` or a `ComplexLazyTensor`.
#' @param index A `character` corresponding to the reduction dimension that should 
#' be either **i** or **j** to specify whether if the reduction is indexed by
#' **i** (rows) or **j** (columns).
#' @param K An `integer` corresponding to the  number of minimal values required.
#' @return A matrix corresponding to the Kmin reduction.
#' @seealso [rkeops::Kmin()]
#' @examples
#' \dontrun{
#' x <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows, 3 columns
#' x_i <- LazyTensor(x, index = 'i')   # LazyTensor from matrix x, indexed by 'i'
#' K <- 2
#' kmin_red_x <- Kmin_reduction(x_i, K, "i")   # Kmin reduction, indexed by "i"
#' }
#' @export
Kmin_reduction <- function(x, K, index) {
    res <- Kmin(x, K, index)
    return(res)
}


# argKmin ----------------------------------------------------------------------


#' argKmin.
#' @description
#' argKmin reduction.
#' @details If `x` is a `LazyTensor` or a `ComplexLazyTensor`,
#' `argKmin(x, K, index)` will:
#' \itemize{
#'     \item{if **index = i**,}{ return the indices of the **K** minimal
#'     values of **x**
#'     over the **i** indexes (rows);}
#'     \item{if **index = j**,}{ return the indices of the **K** minimal
#'     values of **x**
#'     over the **j** indexes (columns).}
#' }
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param x A `LazyTensor` or a `ComplexLazyTensor`.
#' @param index A `character` corresponding to the reduction dimension that should 
#' be either **i** or **j** to specify whether if the reduction is indexed by
#' **i** (rows) or **j** (columns).
#' @param K An `integer` corresponding to the  number of minimal values required.
#' @return A matrix corresponding to the argKmin reduction.
#' @examples
#' \dontrun{
#' x <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows and 3 columns
#' x_i <- LazyTensor(x, index = 'i')   # creating LazyTensor from matrix x, 
#'                                     # indexed by 'i'
#' K <- 2
#' argkmin_x <- argKmin(x_i, K, "i")   # argKmin reduction 
#'                                     # indexed by "i"
#' 
#' }
#' @export
argKmin <- function(x, K, index) {
    if(!is.int(K))
        stop("`K` input argument should be an integer.")
    if(!check_index(index))
        stop("`index` input argument should be a character, either 'i' or 'j'.")
    res <- reduction.LazyTensor(x, "ArgKMin", index, opt_arg = K)
    return(res)
}


# argKmin reduction ------------------------------------------------------------

#' argKmin reduction.
#' @description
#' argKmin reduction. Redirects to `argKmin` function.
#' @details If `x` is a `LazyTensor` or a `ComplexLazyTensor`,
#' `argKmin(x, K, index)` will:
#' \itemize{
#'     \item{if **index = i**,}{ return the indices of the **K** minimal
#'     values of **x**
#'     over the **i** indexes (rows);}
#'     \item{if **index = j**,}{ return the indices of the **K** minimal
#'     values of **x**
#'     over the **j** indexes (columns).}
#' }
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param x A `LazyTensor` or a `ComplexLazyTensor`.
#' @param index A `character` corresponding to the reduction dimension that should 
#' be either **i** or **j** to specify whether if the reduction is indexed by
#' **i** (rows) or **j** (columns).
#' @param K An `integer` corresponding to the  number of minimal values required.
#' @return A matrix corresponding to the argKmin reduction.
#' @seealso [rkeops::argKmin()]
#' @examples
#' \dontrun{
#' x <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows and 3 columns
#' x_i <- LazyTensor(x, index = 'i')   # creating LazyTensor from matrix x, 
#'                                     # indexed by 'i'
#' K <- 2
#' argkmin_red_x <- argKmin_reduction(x_i, K, "i")  # argKmin reduction 
#'                                                  # indexed by "i"
#' }
#' @export
argKmin_reduction <- function(x, K, index) {
    res <- argKmin(x, K, index)
    return(res)
}


# Kmin-argKmin -----------------------------------------------------------------

#' Kmin-argKmin.
#' @description
#' K-Min-argK-min reduction.
#' @details If `x` is a `LazyTensor` or a `ComplexLazyTensor`,
#' `Kmin_argKmin(x, K, index)` will:
#' \itemize{
#'     \item{if **index = i**,}{ return the **K** minimal values of **x**
#'     and its indices over the **i** indices (rows);}
#'     \item{if **index = j**,}{ return the **K** minimal values of **x**
#'     and its indices over the **j** indices (columns).}
#' }
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param x A `LazyTensor` or a `ComplexLazyTensor`.
#' @param index A `character` corresponding to the reduction dimension that should 
#' be either **i** or **j** to specify whether if the reduction is indexed by
#' **i** (rows) or **j** (columns).
#' @param K An `integer` corresponding to the  number of minimal values required.
#' @return A matrix corresponding to the Kmin-argKmin reduction.
#' @examples
#' \dontrun{
#' x <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows and 3 columns
#' x_i <- LazyTensor(x, index = 'i')   # creating LazyTensor from matrix x, 
#'                                     # indexed by 'i'
#' K <- 2
#' k_argk_x <- Kmin_argKmin(x_i, K, "i")  # Kmin-argKmin reduction 
#'                                        # indexed by "i"
#' 
#' }
#' @export
Kmin_argKmin <- function(x, K, index) {
    if(!is.int(K))
        stop("`K` input argument should be an integer.")
    if(!check_index(index))
        stop("`index` input argument should be a character, either 'i' or 'j'.")
    res <- reduction.LazyTensor(x, "KMin_ArgKMin", index, opt_arg = K)
    return(res)
}


# Kmin-argKmin reduction -------------------------------------------------------

#' Kmin-argKmin reduction.
#' @description
#' K-Min-argK-min reduction. Redirects to `Kmin_argKmin` function.
#' @details If `x` is a `LazyTensor` or a `ComplexLazyTensor`,
#' `Kmin_argKmin(x, K, index)` will:
#' \itemize{
#'     \item{if **index = i**,}{ return the **K** minimal values of **x**
#'     and its indices over the **i** indices (rows);}
#'     \item{if **index = j**,}{ return the **K** minimal values of **x**
#'     and its indices over the **j** indices (columns).}
#' } 
#' @details  
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param x A `LazyTensor` or a `ComplexLazyTensor`.
#' @param index A `character` corresponding to the reduction dimension that should 
#' be either **i** or **j** to specify whether if the reduction is indexed by 
#' **i** (rows) or **j** (columns).
#' @param K An `integer` corresponding to the  number of minimal values required.
#' @return A matrix corresponding to the Kmin-argKmin reduction.
#' @seealso [rkeops::Kmin_argKmin()]
#' @examples
#' \dontrun{
#' x <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows and 3 columns
#' x_i <- LazyTensor(x, index = 'i')   # LazyTensor from matrix x, 
#'                                     # indexed by 'i'
#' 
#' K <- 2
#' k_argk_x <- Kmin_argKmin_reduction(x_i, K, "i")  # Kmin-argKmin reduction 
#'                                                  # over the "i" indices
#' }
#' @export
Kmin_argKmin_reduction <- function(x, K, index) {
    res <- Kmin_argKmin(x, K, index)
    return(res)
}


# LogSumExp  -------------------------------------------------------------------

#' Log-Sum-Exp reduction.
#' @description
#' Log-Sum-Exp reduction.
#' @details `logsumexp(x, index, weight)` will:
#' \itemize{
#'     \item{if **index = i**,}{ return the Log-Sum-Exp reduction of **x** 
#'     over the **i** indices (rows);}
#'     \item{if **index = j**,}{ return the Log-Sum-Exp reduction of **x** 
#'     over the **j** indices (columns).}
#' } 
#' @details  
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param x A `LazyTensor` or a `ComplexLazyTensor` indexed by 'i' and 'j'.
#' @param index A `character` corresponding to the reduction dimension that
#' should be either **i** or **j** to specify whether if the reduction
#' is indexed by **i** (rows) or **j** (columns).
#' @param weight An optional object (`LazyTensor` or `ComplexLazyTensor`) that 
#' specifies scalar or vector-valued weights.
#' @return A matrix corresponding to the Log-Sum-Exp reduction.
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
#' logsumexp_xw <- logsumexp(S_ij, 'i', w_j) # logsumexp reduction 
#'                                           # over the 'i' indices
#'                                          
#' logsumexp_x <- logsumexp(S_ij, 'i')      # logsumexp reduction without
#'                                          # weight over the 'i' indices
#' }
#' @export
logsumexp <- function(x, index, weight = NA) {
    if(check_index(index) && is.na(weight))
        res <- reduction.LazyTensor(x, "Max_SumShiftExp", index)
    else if(check_index(index) && !is.na(weight))
        res <- reduction.LazyTensor(x, "Max_SumShiftExpWeight", 
                                    index, opt_arg = weight)
    else
        stop("`index` input argument should be a character, either 'i' or 'j'.")
    return(res)
}


# LogSumExp reduction  ---------------------------------------------------------

#' Log-Sum-Exp reduction.
#' @description
#' Log-Sum-Exp reduction.
#' @details `logsumexp_reduction(x, index, weight)` will:
#' \itemize{
#'     \item{if **index = i**,}{ return the Log-Sum-Exp reduction of **x** 
#'     over the **i** indices (rows);}
#'     \item{if **index = j**,}{ return the Log-Sum-Exp reduction of **x** 
#'     over the **j** indices (columns).}
#' } 
#' @details  
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param x A `LazyTensor` or a `ComplexLazyTensor`.
#' @param index A `character` corresponding to the reduction dimension that should 
#' be either **i** or **j** to specify whether if the reduction is indexed by 
#' **i** (rows) or **j** (columns).
#' @param weight An optional object (`LazyTensor` or `ComplexLazyTensor`) that 
#' specifies scalar or vector-valued weights.
#' @return A matrix corresponding to the Log-Sum-Exp reduction.
#' @seealso [rkeops::logsumexp()]
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
logsumexp_reduction <- function(x, index, weight = NA) {
    res <- logsumexp(x, index)
    return(res)
}


# SumSoftMaxWeight --------------------------------------------------------------

#' Sum of weighted Soft-Max reduction.
#' @description
#' Sum of weighted Soft-Max reduction.
#' @details `sumsoftmaxweight(x, index, weight)` will:
#' \itemize{
#'     \item{if **index = i**,}{ return the Log-Sum-Exp reduction of **x** 
#'     over the **i** indices (rows);}
#'     \item{if **index = j**,}{ return the Log-Sum-Exp reduction of **x** 
#'     over the **j** indices (columns).}
#' } 
#' @details  
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param x A `LazyTensor` or a `ComplexLazyTensor` indexed by 'i' and 'j'.
#' @param index A `character` corresponding to the reduction dimension that should 
#' be either **i** or **j** to specify whether if the reduction is indexed by 
#' **i** (rows) or **j** (columns).
#' @param weight An optional object (`LazyTensor` or `ComplexLazyTensor`) that 
#' specifies scalar or vector-valued weights.
#' @return A matrix corresponding to the Sum of weighted Soft-Max reduction.
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
#' ssmaxweight <- sumsoftmaxweight(S_ij, 'i', V_ij) # sumsoftmaxweight reduction 
#'                                                    # over the 'i' indices
#' }
#' @export
sumsoftmaxweight <- function(x, index, weight) {
    formula2 = paste("Concat(IntCst(1),", weight$formula, ")", sep = "")
    if(check_index(index))
        res <- reduction.LazyTensor(x, "Max_SumShiftExpWeight", 
                                    index, opt_arg = formula2)
    else
        stop("`index` input argument should be a character, either 'i' or 'j'.")
    
    return(res)
}


# SumSoftMaxWeight Reduction ---------------------------------------------------

#' Sum of weighted Soft-Max reduction.
#' @description
#' Sum of weighted Soft-Max reduction.
#' @details `sumsoftmaxweight_reduction(x, index, weight)` will:
#' \itemize{
#'     \item{if **index = i**,}{ return the Log-Sum-Exp reduction of **x** 
#'     over the **i** indices (rows);}
#'     \item{if **index = j**,}{ return the Log-Sum-Exp reduction of **x** 
#'     over the **j** indices (columns).}
#' } 
#' @details  
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param x A `LazyTensor` or a `ComplexLazyTensor` indexed by 'i' and 'j'.
#' @param index A `character` corresponding to the reduction dimension that should 
#' be either **i** or **j** to specify whether if the reduction is indexed by 
#' **i** (rows) or **j** (columns).
#' @param weight An optional object (`LazyTensor` or `ComplexLazyTensor`) that 
#' specifies scalar or vector-valued weights.
#' @return A matrix corresponding to the Sum of weighted Soft-Max reduction.
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
    res <- sumsoftmaxweight(x, index, weight)
    return(res)
}


# CONSTANT AND PADDING/CONCATENATION OPERATIONS ================================

# Elem -------------------------------------------------------------------------

# TODO doc
#' Elem.
#' @description
#' Symbolic element extraction. A unary operation.
#' @details  
#' Extracts the m-th element of a given vector `x`.
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param x A `LazyTensor` or a `ComplexLazyTensor`.
#' @param m An `integer` corresponding to the index of the element of `x` we
#' want to extract (`x[m]`).
#' @return A `LazyTensor` that encodes, symbolically, the m-th element `x[m]`
#' of the vector `x`.
#' @examples
#' \dontrun{
#' x <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows, 3 columns
#' x_i <- LazyTensor(x, index = 'i')   # LazyTensor from matrix x, indexed by 'i'
#' m <- 2
#' 
#' elem_x <- elem(x_i, m)  # symbolic matrix
#' }
#' @export
elem <- function(x, m) {
    if(!is.int(m)) {
        stop("`m` input argument should be an integer.")
    }
    D <- get_inner_dim(x) # TODO check that this is, indeed, inner_dim that we want
    if(m < 0 || m > D) {
        stop(paste("Index `m` is out of bounds. Should be in [1, ",
                   D, "].", sep = ""))
    }
    res <- unaryop.LazyTensor(x, "Elem", m, dim_res = 1)
    return(res)
}


# Elem -------------------------------------------------------------------------

#' ElemT.
#' @description
#' ElemT.
#' @details 
#' `elemT(x, m, n)` insert ??? TODO
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param x A `LazyTensor` or a `ComplexLazyTensor`.
#' @param m An `integer` corresponding at position M of a vector of zeros.
#' @param n An `integer` corresponding to the length of the vector of zeros.
#' @return 
#' @examples
#' \dontrun{
#' x <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows, 3 columns
#' x_i <- LazyTensor(x, index = 'i')   # creating LazyTensor from matrix x, 
#'                                     # indexed by 'i'
#' 
#' m <- 2
#' n <- 3
#' 
#' elemT_x <- elemT(x_i, m, n)  # symbolic matrix
#' }
#' @export
elemT <- function(x, m, n) {
    if(!is.int(m)) 
        stop("`m` input argument should be an integer.")
    if(!is.int(n)) 
        stop("`n` input argument should be an integer.")
    res <- unaryop.LazyTensor(x, "ElemT", m, n)
    return(res)
}


# Extract ----------------------------------------------------------------------

#' Extract.
#' @description
#' Symbolic sub-vector extraction. A unary operation.
#' @details `extract(x_i, m, d)` extracts a sub-vector from vector `x`
#' (`m` is the starting index, and `d` is dimension of the extracted sub-vector).
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param x A `LazyTensor` or a `ComplexLazyTensor`.
#' @param m An `integer` corresponding to the starting index.
#' @param d An `integer` corresponding to the output dimension.
# TODO ci-dessous écrire autre chose que "vector" ??
#' @return A `LazyTensor` that encodes, symbolically, the sub-vector `x[m:m+d]`
#' of the vector `x`.
#' @examples
#' \dontrun{
#' x <- matrix(runif(150 * 5), 150, 5) # arbitrary R matrix, 150 rows, 3 columns
#' x_i <- LazyTensor(x, index = 'i')   # LazyTensor from matrix x, indexed by 'i'
#' m <- 2
#' d <- 2
#' 
#' extract_x <- extract(x_i, m, d)      # symbolic matrix
#' }
#' @export
extract <- function(x, m, d) {
    # type check
    if(!is.int(m))
        stop("`m` input argument should be an integer.")
    if(!is.int(d)) 
        stop("`d` input argument should be an integer.")
    # dim check
    D <- get_inner_dim(x) # TODO check that this is, indeed, inner_dim that we want
    if(m < 0 || m > D) {
        stop(paste("Index `m` is out of bounds. Should be in [1, ",
                   D, "].", sep = ""))
    }
    if(d < 1 || (D < (m + d))) {
        stop(
            paste("Slice dimension is out of bounds. Input `d` should be in [1, ",
                   D, "-m] where `m` is the starting index.", sep = "")
            )
    }
    res <- unaryop.LazyTensor(x, "Extract",
                              opt_arg = m, opt_arg2 = d,
                              dim_res = d)
    return(res)
}


# ExtractT ---------------------------------------------------------------------

#' ExtractT.
#' @description
#' ExtractT.
#' @details insert vector f in a larger vector of zeros (M is starting index,
#' D is dimension of output)
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param x A `LazyTensor` or a `ComplexLazyTensor`.
#' @param m An `integer` corresponding to the starting index.
#' @param d An `integer` corresponding to the output dimension.
#' @return 
#' @examples
#' \dontrun{
#' x <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows, 3 columns
#' x_i <- LazyTensor(x, index = 'i')   # creating LazyTensor from matrix x, 
#'                                     # indexed by 'i'
#' 
#' m <- 2
#' d <- 2
#' 
#' extractT_x <- extractT(x_i, m, d)      # symbolic matrix
#' }
#' @export
extractT <- function(d) {
    if(!is.int(m)) 
        stop("`m` input argument should be an integer.")
    if(!is.int(d)) 
        stop("`d` input argument should be an integer.")
    res <- unaryop.LazyTensor(x, "ExtractT", opt_arg = m, opt_arg2 = d)
    return(res)
}


# Concatenation ----------------------------------------------------------------

#' Concatenation.
#' @description
#' Concatenation of two `LazyTensor` or `ComplexLazyTensor`. A binary operation.
#' @details If `x` and `y` are two `LazyTensor` or `ComplexLazyTensor`,
#' `concat(x, y)` encodes, symbolically, the concatenation of `x` and `y` along
#' their inner dimension. TODO check if this is, indeed, along the inner dimension !
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param x A `LazyTensor` or a `ComplexLazyTensor`.
#' @param x A `LazyTensor` or a `ComplexLazyTensor`.
#' @return A `LazyTensor` or a `ComplexLazyTensor` that encodes, symbolically,
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
    res <- binaryop.LazyTensor(x, y, "Concat",
                               dim_check_type = NA,
                               dim_res = dim_res)
    return(res)
}


# One-hot ----------------------------------------------------------------------

#' One-hot.
#' @description
#' One-hot 
#' @details If `x` is a scalar value encoded as a `LazyTensor`,
#' `one_hot(x, D)` encodes, symbolically, a vector of length **D**
#' whose round(`x`)-th coordinate is equal to 1, and the other ones to 0.
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param x A `LazyTensor` encoding a scalar value.
#' @param D An `integer` corresponding to the output dimension.
#' @return A `LazyTensor`.
#' @examples
#' \dontrun{
#' A <- 7
#' A_LT <- LazyTensor(A) # LazyTensor from scalar A
#' D <- 7
#' 
#' onehot_x <- one_hot(A, D) # symbolic vector of length D
#' }
#' @export
one_hot <- function(x, D) {
    if((!is.LazyTensor(x)) || (is.ComplexLazyTensor(x))) {
        stop(paste("`one_hot` operation can only be applied to ",
                   "`LazyTensor`, not `ComplexLazyTensor`",
                   sep = ""))
    }
    
    if(x$dimres != 1 || !is.LazyScalar(x)) {
        stop("One-hot encoding is only supported for scalar formulas.")
    }
    
    if(!is.int(D)) {
        stop("`D` input argument should be an integer.")
    }
    
    res <- unaryop.LazyTensor(x, "OneHot", opt_arg = D, dim_res = D)
    return(res)
}


# ELEMENTARY DOT PRODUCT =======================================================


# MatVecMult -------------------------------------------------------------------

#' Matrix-vector product.
#' @description
#' Matrix-vector product - a binary operation.
#' @details `matvecmult(x, y)` encodes, symbolically,
#' the matrix-vector product of `x` and `y` along their last dimension.
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param x A `LazyTensor` or a `ComplexLazyTensor`.
#' @param D An `integer` corresponding to the output dimension.
#' @return 
#' @examples
#' \dontrun{
#' x <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows, 3 columns
#' x_i <- LazyTensor(x, index = 'i')   # LazyTensor from matrix x, indexed by 'i'
#' 
#' mv_mult_xy <- matvecmult(x_i, y_j) # symbolic matrix
#' }
#' @export
matvecmult <- function(x, y) {
    # TODO dim_res ??
    #dim_res <- get_inner_dim(x) / get_inner_dim(y)
    res <- binaryop.LazyTensor(x, y, "MatVecMult",
                               dim_check_type = NA,
                               dim_res = dim_res)
    return(res)
}


# VecMatMult -------------------------------------------------------------------

#' Vector-matrix product.
#' @description
#' Vector-matrix product - a binary operation.
#' @details vector-matrix product `x` x `y`: `x` is vector, `y` is vector 
#' interpreted as matrix (column-major)
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param x A `LazyTensor` or a `ComplexLazyTensor`.
#' @param D An `integer` corresponding to the output dimension.
#' @return 
#' @examples
#' \dontrun{
#' x <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows, 3 columns
#' x_i <- LazyTensor(x, index = 'i')   # LazyTensor from matrix x, indexed by 'i'
#' 
#' 
#' mv_mult_xy <- matvecmultt(x_i, y_j) # symbolic matrix
#' }
#' @export
vecmatmult <- function(x, y) {
    if(is.LazyVector(x))
        res <- binaryop.LazyTensor(x, y, "VecMatMult", dim_check_type = NA)
    else
        stop("`x` input should be a LazyTensor")
    return(res)
}


# Tensorprod -------------------------------------------------------------------



#    If ``x._shape[-1] == A`` and ``y._shape[-1] == B``,
# ``z = x.tensorprod(y)`` returns a :class:`GenericLazyTensor`
# such that ``z._shape[-1] == A*B`` which encodes, symbolically,
# the tensor product of ``x`` and ``y`` along their last dimension.
# For details, please check the documentation of the KeOps operation ``"TensorProd"`` in
# the :doc:`main reference page <../../../api/math-operations>`.
#' Tensor product.
#' @description
#' Tensor product of vectors - a binary operation.
#' @details 
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param x 
#' @param y 
#' @return 
#' @examples
#' \dontrun{

#' }
#' @export
tensorprod <- function(x, y) {
    if(is.LazyVector(x) && is.LazyVector(y)) {
        dim_res <- x$dimres * y$dimres
        res <- binaryop.LazyTensor(x, y, "TensorProd", dim_check_type = NA, 
                                   dimres = dim_res)
    }
    else
        stop("`x` and `y` arguments should be vectors.")
    return(res)
}

# SYMBOLIC GRADIENT ============================================================

# Gradient ---------------------------------------------------------------------

#' Symbolic gradient operation.
#' @description
#' Symbolic gradient operation.
#' @details `grad(x, v, e)` returns a `LazyTensor` which encodes, 
#' symbolically, the gradient (more precisely, the adjoint of the differential 
#' operator) of ``x``, with respect to variable `v`, and applied to `e`.
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param x A `LazyTensor` or a `ComplexLazyTensor`.
#' @param v A `LazyTensor`, a `ComplexLazyTensor`, a vector of numeric values, 
#' or a scalar value.
#' @param gradin A `LazyTensor`, a `ComplexLazyTensor`, a vector of numeric values, 
#' or a scalar value.
#' @return A `LazyTensor` or a `ComplexLazyTensor`.
#' @examples
#' \dontrun{
#' x <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows and 3 columns
#' g <- matrix(runif(100 * 3), 100, 3) # arbitrary R matrix, 100 rows and 3 columns
#' x_i <- LazyTensor(x, index = 'i')   # creating LazyTensor from matrix x, 
#'                                     # indexed by 'i'
#' g_j <- LazyTensor(y, index = 'j')   # creating LazyTensor from matrix g, 
#'                                     # indexed by 'j'
#' v_i <- LazyTensor(c(3,2))           # parameter LazyTensor
#' 
#' grad_xy <- weightedsqnorm(x_i, v_i, g_j) # symbolic matrix
#' }
#' @export
grad <- function(x, v, gradin) {
    if(!is.LazyTensor(v))
        v <- LazyTensor(v)
    res <- binaryop.LazyTensor(x, gradin, "Grad", opt_arg = v, 
                               dim_check_type = "same")
    return(res)
}



# Basic example
<<<<<<< Updated upstream
# 
# D <- 3
# M <- 100
# N <- 150
# E <- 4
# x <- matrix(runif(M * D), M, D)
# y <- matrix(runif(N * D), N, D)
# z <- matrix(runif(N * E), N, E)
# b <- matrix(runif(N * E), N, E)
# 
# vect <- rep(1, 10)
# s <- 0.25
# #
# ## creating LazyTensor from matrices
# x_i <- LazyTensor(x, index = 'i')
# y_j <- LazyTensor(y, index = 'j')
# z_j <- LazyTensor(z, index = 'j')
# 
# z <- matrix(1i^ (-6:5), nrow = 4) # complex 4x3 matrix
# z_i <- LazyTensor(z, index = 'i', is_complex = TRUE)
# conj_z_i <- Conj(z_i)
# b_j = b
# 
# # Symbolic matrix of squared distances:
# SqDist_ij = sum( (x_i - y_j)^2 )
# 
# # Symbolic Gaussian kernel matrix:
# K_ij = exp( - SqDist_ij / (2 * s^2) )
=======

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

# Symbolic Gaussian kernel matrix:
#K_ij = exp( - SqDist_ij / (2 * s^2) )
>>>>>>> Stashed changes
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
