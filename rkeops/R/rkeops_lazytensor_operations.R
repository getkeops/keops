#library(rkeops)
#library(stringr)
#library(data.table)
#
#set_rkeops_option("tagCpuGpu", 0)
#set_rkeops_option("precision", "double")

#' Build and return a LazyTensor object
#' @description
#' LazyTensors objects are wrappers around R matrices or vectors that are used to create
#' symbolic formulas for the KeOps reduction operations.
#' @details
#' The use of the function `LazyTensor` is detailed in the vignettes. 
#' Run `browseVignettes("rkeops")` to access the vignettes.
#' @author Ghislain Durif
#' @param x A matrix or a vector of numeric values, or a scalar value
#' @param index A text string that should be either **i** or **j**, or an **NA** value (the default),
#' to specify whether if the **x** variable is indexed by **i**, by **j**, or is a fixed parameter across indices.
#' If **x** is a matrix, **index** must be **i** or **j**.
#' @return An object of class "LazyTensor", which is a list with the following elements:
#' @return
#' \itemize{
#'     \item{**formula**:}{ A string defining the mathematical operation to be computed by the KeOps routine}
#'     \item{**args**:}{ A vector of arguments containing a unique identifier associated to type of the argument :
#'     \itemize{
#'         \item{**Vi(n)**:}{ vector indexed by **i** of dim **n**}
#'         \item{**Vj(n)**:}{ vector indexed by **j** of dim **n**}
#'         \item{**Pm(n)**:}{ fixed parameter of dim **n**}
#'     }}
#'     \item{**vars**:}{ A list of R matrices which will be the inputs of the KeOps routine}
#' }
#' 
#' **Alternatives**
#' \itemize{
#'    \item LazyTensor(x, "i") is equivalent to Vi(x) (see **Vi()** function)
#'    \item LazyTensor(x, "j") is equivalent to Vi(x) (see **Vj()** function)
#'    \item LazyTensor(x) is equivalent to Pm(x) (see **Pm()** function)
#' }
#'
#' @examples
#' \dontrun{
#' # Data
#' nx <- 100
#' ny <- 150
#' x <- matrix(runif(nx * 3), nrow = nx, ncol = 3) # arbitrary R matrix representing 100 data points in R^3
#' y <- matrix(runif(ny * 3), nrow = ny, ncol = 3) # arbitrary R matrix representing 150 data points in R^3
#' s <- 0.1                                        # scale parameter
#' 
#' # Turn our Tensors into KeOps symbolic variables:
#' x_i <- LazyTensor(x, "i")   # symbolic object representing an arbitrary row of x, indexed by the letter "i"
#' y_j <- LazyTensor(y, "j")   # symbolic object representing an arbitrary row of y, indexed by the letter "j"
#' 
#' # Perform large-scale computations, without memory overflows:
#' D_ij <- sum((x_i - y_j)^2)    # symbolic matrix of pairwise squared distances, with 100 rows and 150 columns
#' K_ij <- exp(- D_ij / s^2)     # symbolic matrix, 100 rows and 150 columns
#' res <- sum(K_ij, index = "i") # actual R matrix (in fact a row vector of length 150 here)
#'                               # containing the column sums of K_ij
#'                               # (i.e. the sums over the "i" index, for each "j" index)
#'
#' }
#' @export
LazyTensor <- function(x, index = NA) {
    # init
    d <- NULL
    cat <- NULL
    
    if(is.character(x))
        stop("`x` input argument should be a matrix, a vector or a scalar.")
    if(is.matrix(x) && is.na(index))
        stop("missing `index` argument.")
    if(!is.matrix(x) && !is.na(index))
        stop("`index` must be NA with a vector or a scalar value.")
    
    

    # 1) input is a matrix, treated as indexed variable, so index must be "i" or "j"
    if(is.matrix(x)) {
        d <- ncol(x)
        if(index == "i")
            cat = "Vi"
        else
            cat = "Vj"
    }
    # 2) else we assume x is a numeric vector, treated as parameter, then converted to matrix
    else {
        d <- length(x)
        cat <- "Pm"
    }

    # Now we define "formula", a string specifying the variable for KeOps C++ codes.
    var_name <- paste("A", address(x), index, sep = "") 
    formula <- var_name
    vars <- list(x)  # vars lists all actual matrices necessary to evaluate the current formula, here only one.
    args <- str_c(var_name, "=", cat, "(", d, ")")
    # finally we build and return the LazyTensor object
    res <- list(formula = formula, args = args, vars = vars)
    class(res) <- "LazyTensor"
    return(res)
}


#' Wrapper LazyTensor indexed by "i"
#' @description
#' Simple wrapper that return an instantiation of `LazyTensor` indexed by "i".
#' Equivalent to `LazyTensor(x, index = "i")`.
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param x A matrix of numeric values, or a scalar value.
#' @return An object of class "LazyTensor" indexed by "i".
#' @examples
#' \dontrun{
#' x <- matrix(runif(150 * 3), 150, 3)
#' Vi_x <- Vi(x) # symbolic object representing an arbitrary row of x, indexed by the letter "i"
#' }
#' @export
Vi <- function(x){
    if(class(x)[1] != "matrix")
        stop("`x` must be a matrix.")
    
    res <- LazyTensor(x, index = "i")
    return(res)
}


#' Wrapper LazyTensor indexed by "j"
#' @description
#' Simple wrapper that return an instantiation of `LazyTensor` indexed by "j".
#' Equivalent to `LazyTensor(x, index = "j")`.
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param x A matrix of numeric values.
#' @return An object of class "LazyTensor" indexed by "j".
#' @examples
#' \dontrun{
#' x <- matrix(runif(150 * 3), 150, 3)
#' Vj_x <- Vj(x) # symbolic object representing an arbitrary row of x, indexed by the letter "j"
#' }
#' @export
Vj <- function(x){
    if(class(x)[1] != "matrix")
        stop("`x` must be a matrix.")
    
    res <- LazyTensor(x, index = "j")
    return(res)
}


#' Wrapper LazyTensor parameter
#' @description
#' Simple wrapper that return an instantiation of `LazyTensor` indexed by "i".
#' Equivalent to `LazyTensor(x)`.
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param x A vector or a scalar value.
#' @return An object of class "LazyTensor" in parameter category.
#' @examples
#' \dontrun{
#' x <- 4
#' Pm_x <- Pm(x)
#' }
#' @export
Pm <- function(x){
    if(!is.numeric(x) || class(x)[1] == "matrix")
        stop("`x` input must be a scalar or a vector.")
    
    res <- LazyTensor(x)
    return(res)
}


# unary ------------------------------------------------------------------------

#' Build a unary operation
#' @description
#' Symbolically applies **opstr** operation to **x**.
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param x A LazyTensor, a vector of numeric values, or a scalar value.
#' @param opstr A text string corresponding to an operation.
#' @return An object of class "LazyTensor".
#' @examples
#' \dontrun{
#' x <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows and 3 columns
#' x_i <- LazyTensor(x, index = 'i')   # creating LazyTensor from matrix x, indexed by 'i'
#' una_xy <- unaryop.LazyTensor(x_i, "Minus")   # symbolic matrix
#' 
#' una2_xy <- unaryop.LazyTensor(x_i, "Pow", opt_arg = 3)  # symbolic matrix
#' }
#' @export
unaryop.LazyTensor <- function(x, opstr, opt_arg = NA, opt_arg2 = NA) {
    if(is.matrix(x)){
        stop(paste("`x` input argument should be a LazyTensor, a vector or a scalar.",
                   "\nIf you want to use a matrix, convert it to LazyTensor first.", sep = ""))
    }
    
    if(is.numeric(x))
        x <- LazyTensor(x)
    
    if(!is.na(opt_arg2))
        formula <- paste(opstr, "(", x$formula, ",", opt_arg, ",", opt_arg2, ")", sep="")
    else if(!is.na(opt_arg))
        formula <- paste(opstr, "(", x$formula, ",", opt_arg, ")", sep="")
    else 
        formula <- paste(opstr, "(", x$formula, ")", sep="")

    res <- list(formula = formula, args = x$args, vars = x$vars)
    class(res) <- "LazyTensor"
    return(res)
}


# binary -----------------------------------------------------------------------

#' Build a binary operation
#' @description
#' Symbolically applies **opstr** operation to **x** and **y**.
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param x A LazyTensor, a vector of numeric values, or a scalar value.
#' @param y A LazyTensor, a vector of numeric values, or a scalar value.
#' @param opstr A text string corresponding to an operation.
#' @param is_operator A boolean used to specify if **opstr** is an operator like ``+``
#' , ``-`` or a "genuine" function.
#' @param dim_check_type A string to specify if, and how, we should check input dimensions.
#' Supported values are:
#' \itemize{
#'    \item {**"same"**:}{ **x** and **y** should have the same inner dimension;}
#'    \item {**"sameor1"** (default):}{ **x** and **y** should have the same inner dimension or
#'    at least one of them should be of dimension 1;}
#'    \item {**NA**:}{ no dimension restriction.}
#' }
#' @return An object of class "LazyTensor".
#' @examples
#' \dontrun{
#' x <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows and 3 columns
#' y <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows and 3 columns
#' x_i <- LazyTensor(x, index = 'i')   # creating LazyTensor from matrix x, indexed by 'i'
#' y_j <- LazyTensor(x, index = 'j')   # creating LazyTensor from matrix x, indexed by 'j'
#' bin_xy <- binaryop.LazyTensor(x_i, y_j, "+", is_operator = TRUE)   # symbolic matrix
#' }
#' @export
binaryop.LazyTensor <- function(x, y, opstr, is_operator = FALSE, dim_check_type = "sameor1") {
    if(is.matrix(x))
        stop(paste("`x` input argument should be a LazyTensor, a vector or a scalar.",
                   "\nIf you want to use a matrix, convert it to LazyTensor first.", sep = ""))
    
    if(is.matrix(y))
        stop(paste("`y` input argument should be a LazyTensor, a vector or a scalar.",
                   "\nIf you want to use a matrix, convert it to LazyTensor first.", sep = ""))
    
    if(is.numeric(x))
        x <- LazyTensor(x)
    
    if(is.numeric(y))
        y <- LazyTensor(y)
    
    # check dimensions
    if(dim_check_type == "sameor1") {
        if (!check_inner_dim(x, y, check_type = dim_check_type)) {
            stop(paste("Operation `", opstr, "` expects inputs of the same dimension or dimension 1. Received ",
                       get_inner_dim(x), " and ", get_inner_dim(y), ".", sep = ""))
        }
    }
    if(dim_check_type == "same") {
        if (!check_inner_dim(x, y, check_type = dim_check_type)) {
            stop(paste("Operation `", opstr, "` expects inputs of the same dimension. Received ",
                       get_inner_dim(x), " and ", get_inner_dim(y), ".", sep = ""))
        }
    }
    
    # special formula for operator
    if(is_operator)
        formula <- paste(x$formula, opstr, y$formula, sep = "")
    else
        formula <- paste(opstr, "(", x$formula, ",", y$formula, ")", sep = "")
    vars <- c(x$vars, y$vars)
    vars[!duplicated(names(vars))]
    args <- unique(c(x$args, y$args))
    
    res <- list(formula = formula, args = args, vars = vars)
    class(res) <- "LazyTensor"
    return(res)
}


# ternary ----------------------------------------------------------------------

#' Build a ternary operation
#' @description
#' Symbolically applies **opstr** operation to **x**, **y** and **z**.
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param x A LazyTensor, a vector of numeric values, or a scalar value.
#' @param y A LazyTensor, a vector of numeric values, or a scalar value.
#' @param opstr A text string corresponding to an operation.
#' @return An object of class "LazyTensor".
#' @examples
#' \dontrun{
#' }
#' @export
ternaryop.LazyTensor <- function(x, y, z, opstr) {
    # check that there are no matrix
    # and convert numeric values to LazyTensor
    names <- c("x", "y", "z")
    args <- list(x, y, z)
    for (i in 1:3) {
        if(is.matrix(args[[i]])) {
            stop(paste("`", names[i], "` input argument should be a LazyTensor, a vector or a scalar.",
                       "\nIf you want to use a matrix, convert it to LazyTensor first.", sep = ""))
        }
        if(is.numeric(args[[i]])) {
            args[[i]] <- LazyTensor(args[[i]])
        }
    }
    x <- args[[1]]
    y <- args[[2]]
    z <- args[[3]]
    # format formula
    formula <- paste(opstr, "(", x$formula, ",", y$formula, ",", z$formula, ")", sep = "")
    vars <- c(x$vars, y$vars, z$vars)
    vars[!duplicated(names(vars))]
    args <- unique(c(x$args, y$args, z$args))
    res <- list(formula = formula, args = args, vars = vars)
    class(res) <- "LazyTensor"
    return(res)
}


# ====== tests for keops_kernel modif =========
# a <- c(1, 2)
# A <- matrix(a, 2, 2)
# ALT <- LazyTensor(A, index = 'i')
# ALTj <- LazyTensor(A, index = 'j')
# Sum_A <- sum(ALT, index = 'i')
# SumSum_Aj <- sum(ALT, index = 'j')
# Sum_Ajbis <- sum(ALTj, index = 'i')
# Sum_Ajter <- sum(ALTj, index = 'j')
# ==============================================


get_inner_dim <- function(x) {
    # Grab x inner dimension.
    # x must be a LazyTensor.
    if(class(x)[1] != "LazyTensor"){
        stop("`x` input argument should be a LazyTensor.")
    }
    end_x_inner_dim <- sub(".*\\(", "", x$args)
    x_inner_dim <- substr(end_x_inner_dim, 1, nchar(end_x_inner_dim) - 1)
    x_inner_dim <- as.integer(x_inner_dim)
    return(x_inner_dim)
}

# TODO finish when question about "or dimension 1" for ternary answered
#check_inner_dim <- function(x, y, z = NA, check_type = "sameor1") {
#    # x and y must be LazyTensors.
#    if((class(x)[1] != "LazyTensor") || (class(y)[1] != "LazyTensor")) {
#        stop("`x` and `y` input arguments should be of class LazyTensor.")
#    }
#    
#    x_inner_dim <- get_inner_dim(x)
#    y_inner_dim <- get_inner_dim(y)
#    
#    if(is.na(z)) {
#        # Check whether if x and y inner dimensions are the same or if at least one of these equals 1.
#        if(check_type == "sameor1") {
#            res <- ((x_inner_dim == y_inner_dim) || ((x_inner_dim == 1) || (y_inner_dim == 1)))
#        }
#        if(check_type == "same") {
#            res <- ((x_inner_dim == y_inner_dim))
#        }
#    }
#    else {
#        
#    }
#    
#    return(res)
#}


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
#' @param x A LazyTensor, a vector of numeric values, or a scalar value.
#' @param y A LazyTensor, a vector of numeric values, or a scalar value.
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
    if(class(x)[1] != "LazyTensor")
        UseMethod("+", y)
    else
        UseMethod("+", x)
}

"+.LazyTensor" <- function(x, y) {
    #if (!check_inner_dim(x, y, check_type = "sameor1")) {
    #    stop(paste("Operation `+` expects inputs of the same dimension or dimension 1. Received ",
    #    get_inner_dim(x), " and ", get_inner_dim(y), ".", sep = ""))
    #}
    res <- binaryop.LazyTensor(x, y, "+", is_operator = TRUE, dim_check_type = "sameor1")
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
#' 
#' **Note**
#' 
#' For **subtraction operation**, `x` and `y` input arguments should have the same inner dimension or be of dimension 1.
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param x A LazyTensor, a vector of numeric values, or a scalar value (matrices can be used with the minus sign only).
#' @param y A LazyTensor, a vector of numeric values, or a scalar value (matrices can be used with the minus sign only).
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
    if(class(x)[1] != "LazyTensor")
        UseMethod("-", y)
    else
        UseMethod("-", x)
}

"-.LazyTensor" <- function(x, y = NA) {
    if((length(y) == 1) && is.na(y))
        res <- unaryop.LazyTensor(x, "Minus")
    else
        res <- binaryop.LazyTensor(x, y, "-", is_operator = TRUE, dim_check_type = "sameor1")
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
#' @param x A LazyTensor, a vector of numeric values, or a scalar value.
#' @param y A LazyTensor, a vector of numeric values, or a scalar value.
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
    if(class(x)[1] != "LazyTensor")
        UseMethod("*", y)
    else
        UseMethod("*", x)
}

"*.LazyTensor" <- function(x, y) {
    res <- binaryop.LazyTensor(x, y, "*", is_operator = TRUE, dim_check_type = "sameor1")
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
#' @param x A LazyTensor, a vector of numeric values, or a scalar value.
#' @param y A LazyTensor, a vector of numeric values, or a scalar value.
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
    if(class(x)[1] != "LazyTensor")
        UseMethod("/", y)
    else
        UseMethod("/", x)
}

"/.LazyTensor" <- function(x, y) {
    res <- binaryop.LazyTensor(x, y, "/", is_operator = TRUE, dim_check_type = "sameor1")
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
#' @param x A LazyTensor, a vector or a matrix of numeric values, or a scalar value.
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
}


# square root ------------------------------------------------------------------
sqrt.default <- .Primitive("sqrt") # assign default as current definition

#' Element-wise square root.
#' @description
#' Symbolic unary operation for element-wise square root.
#' @details If `x` is a `LazyTensor`, `sqrt(x)` returns a `LazyTensor` that encodes, symbolically,
#' the element-wise square root of `x` ; else, computes R default square root function.
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param x A LazyTensor, a vector or a matrix of numeric values, or a scalar value.
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
#' @param x A LazyTensor, a vector or a matrix of numeric values, or a scalar value.
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
#' \itemize{
#'     \item{if **y = 2**,}{ `x^y` relies on the `"Square"` KeOps operation;}
#'     \item{if **y = 0.5**,}{ `x^y` uses on the `"Sqrt"` KeOps operation;}
#'     \item{if **y = -0.5**,}{ `x^y` uses on the `"Rsqrt"` KeOps operation.}
#' }
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param x A LazyTensor, a vector of numeric values, or a scalar value.
#' @param y A LazyTensor, a vector of numeric values, or a scalar value.
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
    if(class(x)[1] != "LazyTensor")
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
            res <- binaryop.LazyTensor(x, y, "Powf", dim_check_type = NA) # power operation
        }
    }
    else
        res <- binaryop.LazyTensor(x, y, "Powf") # power operation
    return(res)
}


# Euclidean scalar product -----------------------------------------------------
"|.default" <- .Primitive("|")

# TODO finish the doc for Euclidean scalar product when questions answered
# #' Euclidean scalar product.
# #' @description
# #' Symbolic binary operation for Euclidean scalar product.
# #' @usage x | y  or  (x | y)
# #' @details If `x` or `y` is a `LazyTensor`, `(x|y)` (or `x | y`) returns a `LazyTensor`
# #' that encodes, symbolically, the Euclidean scalar product between `x` and `y`, which must have the same shape.
# #' (In case one of the arguments is a vector or a scalar, it is first converted to `LazyTensor`).
# #' If none of the arguments is a `LazyTensor`, is equivalent to the "|" R operator.
# #'
# #' **Note**
# #' 
# #' `x` and `y` input arguments should have the same inner dimension.
# #' @author Chloe Serre-Combe, Amelie Vernay
# #' @param x A LazyTensor, a vector of numeric values, or a scalar value.
# #' @param y A LazyTensor, a vector of numeric values, or a scalar value.
# #' @return An object of class "LazyTensor" if the function is called with a `LazyTensor`,
# #' and an object of class "numeric", otherwise.
# #' @examples
# #' \dontrun{
# #' x <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows and 3 columns
# #' y <- matrix(runif(250 * 3), 250, 3) # arbitrary R matrix, 250 rows and 3 columns
# #' x_i <- LazyTensor(x, index = 'i')   # creating LazyTensor from matrix x, indexed by 'i'
# #' y_j <- LazyTensor(x, index = 'j')   # creating LazyTensor from matrix x, indexed by 'j'
# #' x_div_y <- x_i / y_j                # symbolic matrix
# #' }
# #' @export
"|" <- function(x, y) { 
    if(class(x)[1] != "LazyTensor")
        UseMethod("|", y)
    else
        UseMethod("|", x)
}
"|.LazyTensor" <- function(x, y) {
    res <- binaryop.LazyTensor(x, y, "|", is_operator = TRUE, dim_check_type = "same")
    res$formula <- paste("(", res$formula, ")", sep = "")
    return(res)
}


"%*%.default" <- .Primitive("%*%") # assign default as current definition

#' Matrix multiplication.
#' @description
#' Symbolic binary operation for element-wise matrix multiplication operator.
#' @usage x %*% y
#' @details If `x` or `y` is a `LazyTensor`, `x %*% y` .... TODO (sum_reduction)
#' If none of the arguments is a `LazyTensor`, is equivalent to the "%*%" R operator.
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param x A LazyTensor, a vector or a matrix of numeric values, or a scalar value.
#' @param y A LazyTensor, a vector or a matrix of numeric values, or a scalar value.
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
    if(class(x)[1] != "LazyTensor")
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
#' @details If `x` is a `LazyTensor`, `exp(x)` returns a `LazyTensor` that encodes, symbolically,
#' the element-wise exponential of `x` ; else, computes R default exponential.
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param x A LazyTensor, a vector of numeric values, or a scalar value.
#' @return An object of class "LazyTensor" if the function is called with a `LazyTensor`,
#' and an object of class "numeric", "matrix", or "array" otherwise, depending on the input class
#' (see R default `exp()` function).
#' @examples
#' \dontrun{
#' x <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows and 3 columns
#' x_i <- LazyTensor(x, index = 'i')   # creating LazyTensor from matrix x, indexed by 'i'
#' Exp_x <- exp(x_i)                   # symbolic matrix, 150 rows and 3 columns
#' }
#' @export
exp <- function(x) {
    UseMethod("exp")
}

exp.LazyTensor <- function(x) {
    res <- unaryop.LazyTensor(x, "Exp")
    return(res)
}


# logarithm --------------------------------------------------------------------
log.default <- .Primitive("log")

#' Element-wise logarithm.
#' @description
#' Symbolic unary operation for element-wise logarithm.
#' @details If `x` is a `LazyTensor`, `exp(x)` returns a `LazyTensor` that encodes, symbolically,
#' the element-wise exponential of `x` ; else, computes R default logarithm.
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param x A LazyTensor, a vector of numeric values, or a scalar value.
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
#' @param x A LazyTensor, a vector of numeric values, or a scalar value.
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


# cosine -----------------------------------------------------------------------
cos.default <- .Primitive("cos")

#' Element-wise cosine.
#' @description
#' Symbolic unary operation for element-wise cosine.
#' @details If `x` is a `LazyTensor`, `exp(x)` returns a `LazyTensor` that encodes, symbolically,
#' the element-wise cosine of `x` ; else, computes R default cosine.
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param x A LazyTensor, a vector of numeric values, or a scalar value.
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


# sine -------------------------------------------------------------------------
sin.default <- .Primitive("sin")

#' Element-wise sine.
#' @description
#' Symbolic unary operation for element-wise sine.
#' @details If `x` is a `LazyTensor`, `sin(x)` returns a `LazyTensor` that encodes, symbolically,
#' the element-wise sine of `x`; else, computes R default sine function.
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param x A `LazyTensor`, a vector or a matrix of numeric values, or a scalar value.
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


# arccosine --------------------------------------------------------------------
acos.default <- .Primitive("acos")

#' Element-wise arccosine.
#' @description
#' Symbolic unary operation for element-wise arccosine.
#' @details If `x` is a `LazyTensor`, `acos(x)` returns a `LazyTensor` that encodes, symbolically,
#' the element-wise arccosine of `x` ; else, computes R default arccosine function.
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param x A `LazyTensor`, a vector or a matrix of numeric values, or a scalar value.
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


# arcsine ----------------------------------------------------------------------
asin.default <- .Primitive("asin")

#' Element-wise arcsine.
#' @description
#' Symbolic unary operation for element-wise arcsine.
#' @details If `x` is a `LazyTensor`, `asin(x)` returns a `LazyTensor` that encodes, symbolically,
#' the element-wise arcsine of `x` ; else, computes R default arcsine function.
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param x A `LazyTensor`, a vector or a matrix of numeric values, or a scalar value.
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


# arctangent -------------------------------------------------------------------
atan.default <- .Primitive("atan")

#' Element-wise arctangent.
#' @description
#' Symbolic unary operation for element-wise arctangent.
#' @details If `x` is a `LazyTensor`, `atan(x)` returns a `LazyTensor` that encodes, symbolically,
#' the element-wise arctangent of `x` ; else, computes R default arctangent function.
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param x A `LazyTensor`, a vector or a matrix of numeric values, or a scalar value.
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
#' @param x A `LazyTensor`, a vector of numeric values, or a scalar value.
#' @param y A `LazyTensor`, a vector of numeric values, or a scalar value.
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
    if(class(x)[1] != "LazyTensor")
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
#' @details If `x` is a `LazyTensor`, `abs(x)` returns a `LazyTensor` that encodes, symbolically,
#' the element-wise absolute value of `x` ; else, computes R default absolute value function.
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param x A `LazyTensor`, a vector or a matrix of numeric values, or a scalar value.
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


# sign function ----------------------------------------------------------------
sign.default <- .Primitive("sign")

#' Element-wise sign.
#' @description
#' Symbolic unary operation for element-wise sign.
#' @details If `x` is a `LazyTensor`, `sign(x)` returns a `LazyTensor` that encodes, symbolically,
#' the element-wise sign of `x` in {-1, 0, +1} ; else, computes R default sign function.
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param x A `LazyTensor`, a vector or a matrix of numeric values, or a scalar value.
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


# round function ---------------------------------------------------------------
round.default <- .Primitive("round")

#' Element-wise rounding function.
#' @description
#' Symbolic binary operation for element-wise rounding function.
#' @details If `x` is a `LazyTensor`, `round(x, d)` returns a `LazyTensor` that encodes, symbolically,
#' the element-wise rounding of `x` to `d` decimal places ; else, computes R default rounding function.
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param x A `LazyTensor`, a vector or a matrix of numeric values, or a scalar value.
#' @param d A scalar value.
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


# Préciser que si on a plusieurs scalaires, on peut faire e.g. min(3, 4, 11)
# qui renvoie 11 mais pour les LazyTensor c'est juste min(x_i) qui renvoie
# l'élément minimal de x_i
# TODO 
# min function -----------------------------------------------------------------
min.default <- .Primitive("min")

# #' Element-wise min function.
# #' @description
# #' Minimum unary operation.
# #' @details If `x` is a `LazyTensor`, `min(x)` TODO... else it computes R default max function with 
# #' other specific arguments (see R default `min()` function).
# #' @author Chloe Serre-Combe, Amelie Vernay
# #' @param x A `LazyTensor`, a vector or a matrix of numeric values, or a scalar value.
# #' @return TODO otherwise, depending on the input class
# #' (see R default `min()` function).
# #' @examples
# #' \dontrun{
# #' }
# #' @export
min <- function(x, ...) {
    UseMethod("min")
}

min.LazyTensor <- function(x) {
    res <- unaryop.LazyTensor(x, "Min")
    return(res)
}

# Préciser que si on a plusieurs scalaires, on peut faire e.g. max(3, 4, 11)
# qui renvoie 11 mais pour les LazyTensor c'est juste max(x_i) qui renvoie
# l'élément maximal de x_i 
# TODO
# max function -----------------------------------------------------------------
max.default <- .Primitive("max")

# #' Element-wise max function.
# #' @description
# #' Maximum unary operation.
# #' @details If `x` is a `LazyTensor`, `max(x)` TODO.... else it computes R default max function with 
# #' other specific arguments (see R default `max()` function).
# #' @author Chloe Serre-Combe, Amelie Vernay
# #' @param x A `LazyTensor`, a vector or a matrix of numeric values, or a scalar value.
# #' @return TODO otherwise, 
# #' 
# #' @examples
# #' \dontrun{
# #' }
# #' @export
max <- function(x, ...) {
    UseMethod("max", x)
}

max.LazyTensor <- function(x) {
    res <- unaryop.LazyTensor(x, "Max")
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
#' @param x A `LazyTensor`, a vector or a matrix of numeric values, or a scalar value.
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
#' @param x A `LazyTensor`, a vector or a matrix of numeric values, or a scalar value.
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


# step function ----------------------------------------------------------------

#' Element-wise step function.
#' @description
#' Symbolic unary operation for element-wise step function.
#' @details If `x` is a `LazyTensor`, `step(x)` returns a `LazyTensor` that encodes, symbolically,
#' the element-wise step function of `x` ; else, computes R default step function with other specific arguments 
#' (see R default `step()` function).
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param x A `LazyTensor`, a vector or a matrix of numeric values, or a scalar value.
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
#' @param x A `LazyTensor`, a vector of numeric values, or a scalar value.
#' @return An object of class "LazyTensor".
#' @examples
#' \dontrun{
#' x <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows and 3 columns
#' x_i <- LazyTensor(x, index = 'i')   # creating LazyTensor from matrix x, indexed by 'i'
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
#' @details `clamp(x, y, z)` returns a `LazyTensor` that encodes, symbolically,
#' the element-wise clamping of ``x`` in ``(y, z)``. 
#' Broadcasting rules apply.
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param x A `LazyTensor`, a vector of numeric values, or a scalar value.
#' @param y A `LazyTensor`, a vector of numeric values, or a scalar value.
#' @param z A `LazyTensor`, a vector of numeric values, or a scalar value.
#' @return An object of class "LazyTensor".
#' @examples
#' \dontrun{

#' }
#' @export
clamp <- function(x, y, z) {
    if(is.numeric(y) && is.numeric(z) && (as.integer(y) - y) == 0 && (as.integer(z) - z) == 0)
        res <- unaryop.LazyTensor(x, "ClampInt", y, z)
    else
        res <- ternaryop.LazyTensor(x, y, z, "Clamp")
    return(res)
}



# clampint function ---------------------------------------------------------------

#' Element-wise clampint function.
#' @description
#' Symbolic ternary operation for element-wise clampint function.
#' @details `clampint(x, y, z)` returns a `LazyTensor` that encodes, symbolically,
#' the element-wise clamping of ``x`` in ``(y, z)`` which are integers. 
#' Broadcasting rules apply.
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param x A `LazyTensor`, a vector of numeric values, or a scalar value.
#' @param y An `integer`.
#' @param z An `integer`.
#' @return An object of class "LazyTensor".
#' @examples
#' \dontrun{

#' }
#' @export
clampint <- function(x, y, z) {
    if(!is.numeric(y) || !is.numeric(z) || (as.integer(y) - y) != 0 || (as.integer(z) - z) != 0) {
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
#' @details `ifelse(x, y, z)` returns a `LazyTensor` that encodes, symbolically,
#' `y` where ``x >= 0`` and ``z`` where ``x < 0``.  Broadcasting rules apply. 
#' `y` and `z` may be fixed integers or floats, or other LazyTensors.
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param x A `LazyTensor`, a vector of numeric values, or a scalar value.
#' @param y A `LazyTensor`, a vector of numeric values, or a scalar value.
#' @param z A `LazyTensor`, a vector of numeric values, or a scalar value.
#' @return An object of class "LazyTensor".
#' @examples
#' \dontrun{

#' }
#' @export
ifelse.LazyTensor <- function(x, y, z) {
    res <- ternaryop.LazyTensor(x, y, z, "IfElse")
}




# Squared Euclidean norm -------------------------------------------------------

#' Squared Euclidean norm.
#' @description
#' Symbolic unary operation for squared Euclidean norm.
#' @details `sqnorm2(x)` returns a `LazyTensor` that encodes, symbolically,
#' the squared Euclidean norm of `x`.
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param x A `LazyTensor`, a vector of numeric values, or a scalar value.
#' @return An object of class "LazyTensor".
#' @examples
#' \dontrun{
#' x <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows and 3 columns
#' x_i <- LazyTensor(x, index = 'i')   # creating LazyTensor from matrix x, indexed by 'i'
#' SqN_x <- sqnorm2(x_i)               # symbolic matrix, 150 rows and 3 columns
#' }
#' @export
sqnorm2 <- function(x) {
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
#' @param x A `LazyTensor`, a vector of numeric values, or a scalar value.
#' @return An object of class "LazyTensor".
#' @examples
#' \dontrun{
#' x <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows and 3 columns
#' x_i <- LazyTensor(x, index = 'i')   # creating LazyTensor from matrix x, indexed by 'i'
#' N_x <- norm2(x_i)               # symbolic matrix, 150 rows and 3 columns
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
#' the squared Euclidean distance between `x` and `y`.
#' 
#' **Note**
#' 
#' `x` and `y` input arguments should have the same inner dimension or be of dimension 1.
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param x A `LazyTensor`, a vector of numeric values, or a scalar value.
#' @param y A `LazyTensor`, a vector of numeric values, or a scalar value.
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

#' Generic squared eucidian norm.
#' @description
#' Symbolic binary operation for weighted squared norm of a LazyTensor.
#' @details `weightedsqnorm(x)` returns a `LazyTensor` that encodes, symbolically,
#' the weighted squared norm of a vector `x` with weights stored in the LazyTensor `s`.
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param x A `vector` of numeric values or a scalar value.
#' @param s A `LazyTensor`, a vector of numeric values, or a scalar value.
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
#' @param y A `LazyTensor`, a vector of numeric values, or a scalar value.
#' @param z A `LazyTensor`, a vector of numeric values, or a scalar value.
#' @return An object of class "LazyTensor".
#' @examples
#' \dontrun{

#' }
#' @export
weightedsqdist <- function(x, y, z) {
    res <- weightedsqnorm(x - y, z)
    return(res)
}



# TODO
# Reduction --------------------------------------------------------------------
#' Reduction operation.
#' @description
#' Applies a reduction to a `LazyTensor`.
#' @details
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param x A `LazyTensor`, a vector or a matrix of numeric values, or a scalar value.
#' @param opstr A `string` formula (like "Sum" or "Max").
#' @param  index A `character` that should be either **i** or **j** to specify whether if 
#' the reduction is indexed by **i** (rows), by **j** (columns).
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
    if(index == "i") 
        tag <- 1
    else 
        tag <- 0
    formula <- paste(opstr, "_Reduction(", x$formula, ",", tag, ")", sep = "")
    args <- x$args
    op <- keops_kernel(formula, args)
    res <- op(x$vars)
    return(res)
}


# TODO
# sum function -----------------------------------------------------------------
sum.default <- .Primitive("sum")

#' Summation operation or Sum reduction.
#' @description
#' Summation unary operation, or Sum reduction.
#' @details 
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param x A `LazyTensor`, a vector or a matrix of numeric values, or a scalar value.
#' @param index A `character` that should be either **i** or **j** or **NA** to specify whether if 
#' the summation is indexed by **i** (rows), by **j** (columns).
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
    else if(is.character(index))
        res <- reduction.LazyTensor(x, "Sum", index)
    else
        stop("`index` input argument should be a character `i`, `j` or NA.")
    return(res)
}


# sum reduction ----------------------------------------------------------------

#' Summation operation or Sum reduction.
#' @description
#' Summation unary operation, or Sum reduction.
#' @details 
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param x A `LazyTensor`, a vector or a matrix of numeric values, or a scalar value.
#' @param index A `character` that should be either **i** or **j** to specify whether if 
#' the summation is indexed by **i** (rows), by **j** (columns).
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
    res <- reduction.LazyTensor(x, "Sum", index)
}


# Basic example

#D <- 3
#M <- 100
#N <- 150
#E <- 4
#x <- matrix(runif(M * D), M, D)
#y <- matrix(runif(N * D), N, D)
#z <- matrix(runif(N * D), N, D)
#b <- matrix(runif(N * E), N, E)
#s <- 0.25
#
## creating LazyTensor from matrices
#x_i = LazyTensor(x, index = 'i')
#y_j = LazyTensor(y, index = 'j')
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
