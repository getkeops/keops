
# LAZYTENSOR CONFIGURATION =====================================================


# LazyTensor -------------------------------------------------------------------
#' Build and return a LazyTensor object.
#' @description
#' `LazyTensor` objects are wrappers around R matrices or vectors that are used 
#' to create symbolic formulas for the `KeOps` reduction operations.
#' @details
#' The `LazyTensor()` function builds a `LazyTensor`, which is a 
#' list containing the following elements:
#' 
#' - `formula`: a string defining the mathematical operation to 
#'     be computed by the `KeOps` routine - each variable is encoded with the
#'     pointer address of its argument, suffixed by 'i', 'j', or 'NA', to
#'     give it a unique identifier;
#' - `args`: a vector of arguments containing a unique identifier 
#'     associated to the type of the argument:
#'     + `Vi(n)`: vector indexed by `i` of dim `n`
#'     + `Vj(n)`: vector indexed by `j` of dim `n`
#'     + `Pm(n)`: fixed parameter of dim `n`
#' - `data`: A list of R matrices which will be the inputs of the `KeOps` 
#'   routine;
#' - `dimres`: An integer corresponding to the inner dimension of 
#'   the `LazyTensor`. `dimres` is used when creating new `LazyTensor`s 
#'   that result from operations, to keep track of the dimension.
#' 
#' **Note 1**
#' 
#' Setting the argument `is_complex` to `TRUE` will build a `ComplexLazyTensor`,
#' which is also a `LazyTensor`. Run `browseVignettes("rkeops")` and see 
#' "RKeOps LazyTensor" vignette for further details on how `ComplexLazyTensors`
#' are built.
#' 
#' **Note 2**
#' 
#' If `x` is an integer, `LazyTensor(x)` builds a `LazyTensor` whose
#' formula is simply `IntCst(x)` and contains all the necessary information;
#' `args` and `data` remain empty to avoid useless storage.
#' 
#' **Alternatives**
#' 
#' - `LazyTensor(x, "i")` is equivalent to `Vi(x)` (see [rkeops::Vi()] function)
#' - `LazyTensor(x, "j")` is equivalent to `Vi(x)` (see [rkeops::Vj()] function)
#' - `LazyTensor(x)` is equivalent to `Pm(x)` (see [rkeops::Pm()]function)
#'
#' Run `browseVignettes("rkeops")` to access the vignettes and see how to use
#' `LazyTensors`.
#' @author Joan Glaunes, Chloe Serre-Combe, Amelie Vernay
#' @param x A matrix or a vector of numeric values, or a scalar value
#' @param index A text string that should be either `"i"` or `"j"`, or an `NA` 
#' value (the default), to specify whether the `x` variable is indexed 
#' by `i` (rows), by `j` (columns), or is a fixed parameter across indices.
#' If `x` is a matrix, `index` must be `"i"` or `"j"`.
#' @param is_complex A boolean (default is FALSE). Whether we want to create 
#' a `ComplexLazyTensor` (`is_complex = TRUE`) or a `LazyTensor` 
#' (`is_complex = FALSE`).
#' @return An object of class "LazyTensor" or "ComplexLazyTensor".
#' @examples
#' \dontrun{
#' # Data
#' nx <- 100
#' ny <- 150
#' x <- matrix(runif(nx * 3), nrow = nx, ncol = 3) # arbitrary R matrix representing 
#'                                                 # 100 data points in R^3
#' y <- matrix(runif(ny * 3), nrow = ny, ncol = 3) # arbitrary R matrix representing 
#'                                                 # 150 data points in R^3
#' s <- 0.1                                        # scale parameter
#' 
#' # Turn our Tensors into KeOps symbolic variables:
#' x_i <- LazyTensor(x, "i")   # symbolic object representing an arbitrary row of x, 
#'                             # indexed by the letter "i"
#' y_j <- LazyTensor(y, "j")   # symbolic object representing an arbitrary row of y, 
#'                             # indexed by the letter "j"
#' 
#' # Perform large-scale computations, without memory overflows:
#' D_ij <- sum((x_i - y_j)^2)    # symbolic matrix of pairwise squared distances, 
#'                               # with 100 rows and 150 columns
#' K_ij <- exp(- D_ij / s^2)     # symbolic matrix, 100 rows and 150 columns
#' res <- sum(K_ij, index = "i") # actual R matrix (in fact a row vector of 
#'                               # length 150 here)
#'                               # containing the column sums of K_ij
#'                               # (i.e. the sums over the "i" index, for each 
#'                               # "j" index)
#' 
#' 
#' # Example : create ComplexLazyTensor:
#' z <- matrix(1i^ (-6:5), nrow = 4)                     # create a complex 4x3 matrix
#' z_i <- LazyTensor(z, index = 'i', is_complex = TRUE)  # create a ComplexLazyTensor, 
#'                                                       # indexed by 'i'
#'
#' }
#' @importFrom data.table address
#' @export
LazyTensor <- function(x, index = NA, is_complex = FALSE) {
    
    if(is.LazyTensor(x)) {
        stop("Input `x` is already a LazyTensor.")
    }
    
    if(!is_complex && is.complex(x)) {
        is_complex = TRUE
    }
    
    # init
    d <- NULL
    cat <- NULL
    
    if(is.character(x))
        stop(paste("`x` input argument should be a matrix, a vector",
                   "a scalar or a complex value.",
                   sep = ""))
    if(is.matrix(x) && is.na(index))
        stop("missing `index` argument.")
    if(!is.matrix(x) && !is.na(index))
        stop("`index` must be NA with a vector or a single value.")
    
    # integer case
    if(is.int(x)) {
        formula <- paste("IntCst(", as.character(x), ")", sep = "")
        dimres <- 1
        # build "constant integer LazyTensor"
        res <- list(formula = formula, dimres = dimres)
        class(res) <- "LazyTensor"
        return(res)
    }
    
    # 1) input is a matrix, treated as indexed variable, so index must be "i" or "j"
    if(is.matrix(x)) {
        d <- ncol(x)
        if(index == "i")
            cat <- "Vi"
        else
            cat <- "Vj"
    }
    # 2) else we assume x is a numeric vector, treated as parameter,
    # then converted to matrix
    else {
        d <- length(x)
        cat <- "Pm"
    }
    
    # Prefix with a letter because starting with a digit causes problems.
    # Suffix with 'i', 'j', or 'NA' to differentiate addresses of LazyTensor
    # created from same variables, and keep track of there index.
    var_name <- paste("A", address(x), index, sep = "")
    formula <- var_name
    data <- list(x)  # data lists all actual matrices necessary to evaluate 
    # the current formula, here only one.
    
    if(is_complex) {
        args <- str_c(var_name, "=", cat, "(", 2 * d, ")")
        
        # build ComplexLazyTensor
        res <- list(formula = formula, args = args, data = data)
        
        # format data in a "complex" way
        ReZ <- Re(x)
        ImZ <- Im(x)
        
        # If input x is a matrix (index != NA), ReImZ will be a matrix such that:
        # The first column of ReImZ is the real part of the first column of x;
        # the second column of ReImZ is the imaginary part of the first column of x;
        # the third column of ReImZ is the real part of the second column of x;
        # and so on.
        # The number of column of Z is twice the number of column of x.
        if(!is.na(index)) {
            ReImZ <- Reduce("cbind",
                            lapply(1:ncol(x),
                                   function(ind) return(cbind(ReZ[, ind],
                                                              ImZ[, ind]
                                   )
                                   )
                            )
            )
        }
        # If input x is a vector or a single complex value, ReImZ will be a vector.
        else {
            ReImZ <- Reduce("cbind",
                            lapply(1:length(x),
                                   function(ind) return(cbind(ReZ[ind],
                                                              ImZ[ind]
                                   )
                                   )
                            )
            )
            # Parameter LazyTensors have vector as data (but apparently it
            # doesn't matter is we leave it as a matrix...)
            # Uncomment below if needed.
            # ReImZ <- as.vector(ReImZ)
        }
        res$data[[1]] <- ReImZ
        
        # add ComplexLazyTensor class
        class(res) <- c("ComplexLazyTensor", "LazyTensor")
    }
    else {
        args <- str_c(var_name, "=", cat, "(", d, ")")
        # finally we build and return the LazyTensor object
        res <- list(formula = formula, args = args, data = data)
        class(res) <- "LazyTensor"
    }
    
    # add inner dimension
    res$dimres <- get_inner_dim(res)
    
    return(res)
}


# Vi ---------------------------------------------------------------------------

#' Wrapper LazyTensor indexed by "i".
#' @description
#' Simple wrapper returning an instance of `LazyTensor` indexed by "i".
#' Equivalent to `LazyTensor(x, index = "i")`.
#' @details See `?LazyTensor` for more details.
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param x A matrix of numeric values, or a scalar value.
#' @param is_complex A boolean (default is FALSE). Whether we want to create 
#' a `ComplexLazyTensor` (`is_complex = TRUE`) or a `LazyTensor` 
#' (`is_complex = FALSE`).
#' @return An object of class "LazyTensor" indexed by "i". See `?LazyTensor` for 
#' more details.
#' @examples
#' \dontrun{
#' x <- matrix(runif(150 * 3), 150, 3)
#' Vi_x <- Vi(x) # symbolic object representing an arbitrary row of x, 
#'               # indexed by the letter "i"
#' }
#' @export
Vi <- function(x, is_complex = FALSE){
    if(!is.matrix(x))
        stop("`x` must be a matrix.")
    
    res <- LazyTensor(x, index = "i", is_complex = is_complex)
    return(res)
}


# Vj ---------------------------------------------------------------------------

#' Wrapper LazyTensor indexed by "j".
#' @description
#' Simple wrapper returning an instance of `LazyTensor` indexed by "j".
#' Equivalent to `LazyTensor(x, index = "j")`.
#' @details See `?LazyTensor` for more details.
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param x A matrix of numeric values.
#' @param is_complex A boolean (default is FALSE). Whether we want to create 
#' a `ComplexLazyTensor` (`is_complex = TRUE`) or a `LazyTensor` 
#' (`is_complex = FALSE`).
#' @return An object of class "LazyTensor" indexed by "j".
#' @examples
#' \dontrun{
#' x <- matrix(runif(150 * 3), 150, 3)
#' Vj_x <- Vj(x) # symbolic object representing an arbitrary row of x, 
#'               # indexed by the letter "j"
#' }
#' @export
Vj <- function(x, is_complex = FALSE){
    if(!is.matrix(x))
        stop("`x` must be a matrix.")
    
    res <- LazyTensor(x, index = "j", is_complex = is_complex)
    return(res)
}


# Pm ---------------------------------------------------------------------------

#' Wrapper LazyTensor parameter.
#' @description
#' Simple wrapper returning an instance of a fixed parameter `LazyTensor`.
#' Equivalent to `LazyTensor(x)`.
#' @details See `?LazyTensor` for more details.
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param x A vector or a scalar value.
#' @param is_complex A boolean (default is FALSE). Whether if we want to create 
#' a `ComplexLazyTensor` (`is_complex = TRUE`) or a `LazyTensor` 
#' (`is_complex = FALSE`).
#' @return An object of class "LazyTensor" in parameter category. 
#' See `?LazyTensor` for more details.
#' @examples
#' \dontrun{
#' x <- 4
#' Pm_x <- Pm(x)
#' }
#' @export
Pm <- function(x, is_complex = FALSE){
    if(is.LazyTensor(x)) {
        stop("`x` input is already a LazyTensor.")
    }
    
    if((is.character(x) || is.matrix(x))) {
        # Should not be a character string, neither a matrix, nor a complex matrix.
        stop("`x` input must be a vector or a single value.")
    }
    
    res <- LazyTensor(x, is_complex = is_complex)
    return(res)
}



# LAZYTENSOR *NARYOP ===========================================================


# unary ------------------------------------------------------------------------

#' Build a unary operation
#' @description
#' Symbolically applies `opstr` operation to `x`.
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param x A `LazyTensor`, a `ComplexLazyTensor`, a vector of numeric values, 
#' or a scalar value.
#' @param opstr A text string corresponding to an operation.
#' @param opt_arg An optional argument which can be a scalar value.
#' @param opt_arg2 An optional argument which can be a scalar value.
#' @param res_type NA (default) or a character string among "LazyTensor" and 
#' "ComplexLazyTensor", to specify if a change of class is required for the 
#' result. (Useful especially when dealing with complex-to-real or
#' real-to-complex functions).
#' @param dim_res `NA` (default) or an integer corresponding to the inner
#' dimension of the output `LazyTensor`. If `NA`, `dim_res` is set to the
#' inner dimension of the input `LazyTensor`.
#' @return An object of class "LazyTensor" or "ComplexLazyTensor".
#' @examples
#' \dontrun{
#' x <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows, 3 columns
#' x_i <- LazyTensor(x, index = 'i')   # creating LazyTensor from matrix x, 
#'                                     # indexed by 'i'
#' una_x <- unaryop.LazyTensor(x_i, "Minus")   # symbolic matrix
#' 
#' una2_x <- unaryop.LazyTensor(x_i, "Pow", opt_arg = 3)  # symbolic matrix
#' 
#' # example with not NA dim_res:
#' ## set dim_res to 1 because the "Norm2" operation results on a (symbolic) scalar
#' una3_x <- unaryop.LazyTensor(x, "Norm2",
#'                              res_type = "LazyTensor",
#'                              dim_res = 1)
#' }
#' @export
unaryop.LazyTensor <- function(x, opstr, opt_arg = NA, opt_arg2 = NA,
                               res_type = NA, dim_res = NA) {
    # input checks
    if(is.matrix(x)){
        stop(
            paste(
                "`x` input argument should be a LazyTensor, a vector or a scalar.",
                "\nIf you want to use a matrix, convert it to LazyTensor first.", 
                sep = ""
            )
        )
    }
    
    if(!is.na(dim_res) && !is.int(dim_res)){
        stop(
            paste(
                "If not NA, `dim_res` input argument should be an integer. ",
                "Received ", dim_res, ".", sep = ""
            )
        )
    }
    
    # result type
    if(!is.na(res_type) && res_type == "ComplexLazyTensor")
        res_type <- c("ComplexLazyTensor", "LazyTensor")
    
    if(is.numeric(x) || is.complex(x))
        x <- LazyTensor(x)
    
    # result dimension
    if(is.na(dim_res)) {
        dim_res <- x$dimres
    }
    
    # Set `as.integer(dim_res)` to avoid printing potential
    # decimal zero: 4.0, 5.0, and so on...
    # If dim_res has a non zero decimal, the function stops anyway.
    dim_res <- as.integer(dim_res)
    
    # format formula depending on the arguments
    if(!is.na(opt_arg2))
        formula <- paste(opstr, "(", x$formula, ",", opt_arg, ",", 
                         opt_arg2, ")", sep = "")
    else if(!is.na(opt_arg))
        formula <- paste(opstr, "(", x$formula, ",", opt_arg, ")", sep = "")
    else 
        formula <- paste(opstr, "(", x$formula, ")", sep = "")
    
    res <- list(formula = formula, args = x$args, data = x$data)
    
    # result dimension
    res$dimres <- dim_res
    # result type
    if(is.na(res_type[1]))
        class(res) <- class(x)
    else
        class(res) <- res_type
    
    return(res)
}


# binary -----------------------------------------------------------------------

#' Build a binary operation
#' @description
#' Symbolically applies `opstr` operation to `x` and `y`.
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param x A `LazyTensor`, a `ComplexLazyTensor`, a vector of numeric values, 
#' or a scalar value.
#' @param y A `LazyTensor`, a `ComplexLazyTensor`, a vector of numeric values, 
#' or a scalar value.
#' @param opstr A text string corresponding to an operation.
#' @param is_operator A boolean used to specify if `opstr` is an operator like 
#' ``+``, ``-`` or a "genuine" function.
#' @param dim_check_type A string to specify if, and how, we should check input 
#' dimensions. Supported values are:
#' - `"same"`: `x` and `y` should have the same inner dimension;
#' - `"sameor1"` (default): `x` and `y` should have the same 
#'    inner dimension or at least one of them should be of dimension 1;
#' - `NA`: no dimension restriction.
#' @param res_type NA (default) or a character string among "LazyTensor" and 
#' "ComplexLazyTensor", to specify if a change of class is required for the 
#' result. (Useful especially when dealing with complex-to-real or
#' real-to-complex functions).
#' @param dim_res NA (default) or an integer corresponding to the inner
#' dimension of the output `LazyTensor`. If NA, `dim_res` is set to the
#' maximum between the inner dimensions of the two input `LazyTensor`s.
#' @param opt_arg NA (default) or list of optional arguments for the formula
#' encoding the binary operation on input LazyTensors.
#' @return An object of class "LazyTensor".
#' @examples
#' \dontrun{
#' x <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows, 3 columns
#' y <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows, 3 columns
#' x_i <- LazyTensor(x, index = 'i')   # LazyTensor from matrix x, indexed by 'i'
#' y_j <- LazyTensor(y, index = 'j')   # LazyTensor from matrix y, indexed by 'j'
#' # symbolic matrix:
#' bin_xy <- binaryop.LazyTensor(x_i, y_j, "+", is_operator = TRUE)
#' }
#' @export
binaryop.LazyTensor <- function(x, y, opstr, is_operator = FALSE,
                                dim_check_type = "sameor1", res_type = NA,
                                dim_res = NA, opt_arg = NA) {
    
    # input checks
    if(is.matrix(x))
        stop(
            paste(
                "`x` input argument should be a LazyTensor, a vector or ", 
                "a scalar.\nIf you want to use a matrix, convert it to ", 
                "LazyTensor first.",
                sep = ""
            )
        )
    
    if(is.matrix(y))
        stop(
            paste(
                "`y` input argument should be a LazyTensor, a vector or ", 
                "a scalar. \nIf you want to use a matrix, convert it to ", 
                "LazyTensor first.", 
                sep = ""
            )
        )
    
    if(is.numeric(x) || is.complex(x))
        x <- LazyTensor(x)
    
    if(is.numeric(y) || is.complex(y))
        y <- LazyTensor(y)
    
    # init
    formula_x <- x$formula
    formula_y <- y$formula
    data <- c(x$data, y$data)
    args <- c(x$args, y$args)
    
    dimres_x <- x$dimres
    dimres_y <- y$dimres
    class_res <- class(x)
    
    if(!is.na(dim_res) && !is.int(dim_res)){
        stop(
            paste(
                "If not NA, `dim_res` input argument should be an integer. ",
                "Received ", dim_res, ".", sep = ""
            )
        )
    }
    
    if(!is.na(res_type) && res_type == "ComplexLazyTensor")
        res_type <- c("ComplexLazyTensor", "LazyTensor")
    
    # check dimensions
    if (!is.na(dim_check_type)) {
        if(dim_check_type == "sameor1") {
            if (!check_inner_dim(x, y, check_type = dim_check_type)) {
                stop(
                    paste(
                        "Operation `", opstr, 
                        "` expects inputs of the same dimension or ",
                        "dimension 1. Received ",
                        get_inner_dim(x), " and ", get_inner_dim(y), ".", 
                        sep = ""
                    )
                )
            }
        }
        else if(dim_check_type == "same") {
            if (!check_inner_dim(x, y, check_type = dim_check_type)) {
                stop(
                    paste(
                        "Operation `", opstr,
                        "` expects inputs of the same dimension. Received ",
                        get_inner_dim(x), " and ", 
                        get_inner_dim(y), ".", sep = ""
                    )
                )
            }
        }
    }
    
    # result dimension
    if(is.na(dim_res)) {
        dim_res <- max(c(dimres_x, dimres_y))
    }
    
    # Set `as.integer(dim_res)` to avoid printing potential
    # decimal zero: 4.0, 5.0, and so on...
    # If dim_res has a non zero decimal, the function stops anyway.
    dim_res <- as.integer(dim_res)
    
    # special formula for operator
    
    if(is_operator)
        formula <- paste(formula_x, opstr, formula_y, sep = "")
    
    else if(!is_operator && is.na(opt_arg))
        formula <- paste(opstr, "(", formula_x, ",", formula_y, ")", sep = "")
    
    else if(!is_operator && !is.na(opt_arg))
        formula <- paste(opstr, "(", formula_x, ",", opt_arg$formula, ",",
                         formula_y, ")", sep = "")
    
    res <- list(formula = formula, args = args, data = data, dimres = dim_res)
    
    # only remove duplicate data if they have same index
    pos_args <- c()
    if(length(res$args) > 1) {
        for(k in 1:(length(res$args)-1)) {
            for(l in (k + 1):length(res$args))
                if(res$args[k] == res$args[l]) {
                    pos_args <- append(pos_args, l)
                }
        }
        
        res$data[pos_args] <- NULL
    }
    res$args <- unique(res$args)
    
    if(!is.na(res_type[1]))
        class(res) <- res_type
    else if((is.ComplexLazyTensor(x) || is.ComplexLazyTensor(y)) 
            || is.ComplexLazyTensor(opt_arg)) {
        class(res) <- c("ComplexLazyTensor", "LazyTensor")
    }
    else
        class(res) <- class_res
    
    return(res)
}


# ternary ----------------------------------------------------------------------

#' Build a ternary operation
#' @description
#' Symbolically applies `opstr` operation to `x`, `y` and `z`.
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param x A `LazyTensor`, a `ComplexLazyTensor`, a vector of numeric values, 
#' or a scalar value.
#' @param y A `LazyTensor`, a `ComplexLazyTensor`, a vector of numeric values, 
#' or a scalar value.
#' @param z A `LazyTensor`, a `ComplexLazyTensor`r, a vector of numeric values, 
#' or a scalar value.
#' @param opstr A text string corresponding to an operation.
#' @param dim_check_type A string to specify if, and how, we should check input 
#' dimensions.
#' Supported values are:
#' - `"same"`: `x` and `y` should have the same inner dimension;
#' -`"sameor1"` (default): `x` and `y` should have the same 
#'    inner dimension or at least one of them should be of dimension 1;
#' - `NA`: no dimension restriction.
#' @param dim_res NA (default) or an integer corresponding to the inner
#' dimension of the output `LazyTensor`. If NA, **dim_res** is set to the
#' maximum between the inner dimensions of the three input `LazyTensor`s.
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
#' # symbolic matrix:
#' tern_xyz <- ternaryop.LazyTensor(x_i, y_j, z_i, "IfElse")
#' }
#' @export
ternaryop.LazyTensor <- function(x, y, z, opstr, dim_check_type = "sameor1",
                                 dim_res = NA) {
    # check that there are no matrix
    # and convert numeric or complex values to LazyTensor
    names <- c("x", "y", "z")
    args <- list(x, y, z)
    for (i in 1:3) {
        if(is.matrix(args[[i]])) {
            stop(
                paste(
                    "`", 
                    names[i], 
                    "` input argument should be a LazyTensor, a ComplexLazyTensor,", 
                    " a vector or a scalar.",
                    "\nIf you want to use a matrix, convert it to LazyTensor first.", 
                    sep = ""
                )
            )
        }
        if(is.numeric(args[[i]]) || is.complex(args[[i]])) {
            args[[i]] <- LazyTensor(args[[i]])
        }
    }
    x <- args[[1]]
    y <- args[[2]]
    z <- args[[3]]
    
    if(!is.na(dim_res) && !is.int(dim_res)){
        stop(
            paste(
                "If not NA, `dim_res` input argument should be an integer. ",
                "Received ", dim_res, ".", sep = ""
            )
        )
    }
    
    # check dimensions
    if(dim_check_type == "sameor1") {
        if (!check_inner_dim(x, y, z, check_type = dim_check_type)) {
            stop(
                paste(
                    "Operation `", opstr, 
                    "` expects inputs of the same dimension or dimension 1. Received ",
                    get_inner_dim(x), ", ", get_inner_dim(y),
                    " and ", get_inner_dim(z), ".", sep = ""
                )
            )
        }
    }
    if(dim_check_type == "same") {
        if (!check_inner_dim(x, y, z, check_type = dim_check_type)) {
            stop(
                paste(
                    "Operation `", opstr, 
                    "` expects inputs of the same dimension. Received ",
                    get_inner_dim(x), ", ", get_inner_dim(y),
                    " and ", get_inner_dim(z), ".", sep = ""
                )
            )
        }
    }
    
    # result dimension
    if(is.na(dim_res)) {
        dim_res <- max(c(x$dimres, y$dimres, z$dimres))
    }
    
    # Set `as.integer(dim_res)` to avoid printing potential
    # decimal zero: 4.0, 5.0, and so on...
    # If dim_res has a non zero decimal, the function stops anyway.
    dim_res <- as.integer(dim_res)
    
    # format formula
    formula <- paste(opstr, "(", x$formula, ",", y$formula, ",", 
                     z$formula, ")", sep = "")
    
    data <- c(x$data, y$data, z$data)
    args <- c(x$args, y$args, z$args)
    dimres <- dim_res
    
    res <- list(formula = formula, args = args, data = data, dimres = dimres)
    
    # only remove duplicate data if they have same index
    pos_args <- c()
    if(length(res$args) > 1) {
        for(k in 1:(length(res$args)-1)) {
            for(l in (k + 1):length(res$args))
                if(res$args[k] == res$args[l]) {
                    pos_args <- append(pos_args, l)
                }
        }
        
        res$data[pos_args] <- NULL
    }
    res$args <- unique(res$args)
    
    if(is.ComplexLazyTensor(x) || is.ComplexLazyTensor(y))
        class(res) <- c("ComplexLazyTensor", "LazyTensor")
    else
        class(res) <- class(x)
    
    return(res)
}


# TYPE CHECKING ================================================================


#' is.LazyTensor?
#' @description
#' Checks whether the given input is a `LazyTensor` or not.
#' @details If `x` is a `LazyTensor`, `is.LazyTensor(x)` returns TRUE, else, 
#' returns FALSE.
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param x An object that we want to know if it is a `LazyTensor`.
#' @return A boolean, TRUE or FALSE.
#' @examples
#' \dontrun{
#' # basic example
#' D <- 3
#' M <- 100
#' x <- matrix(runif(M * D), M, D)
#' 
#' # create LazyTensor
#' x_i <- LazyTensor(x, index = 'i')
#' 
#' # call is.LazyTensor
#' is.LazyTensor(x_i) # returns TRUE
#' is.LazyTensor(x)   # returns FALSE
#' }
#' @export
is.LazyTensor <- function(x){
    return("LazyTensor" %in% class(x))
}


#' is.ComplexLazyTensor?
#' @description
#' Checks whether the given input is a `ComplexLazyTensor` or not.
#' @details If `x` is a `ComplexLazyTensor`, `is.ComplexLazyTensor(x)` 
#' returns TRUE, else, returns FALSE.
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param x An object that we want to know if it is a `ComplexLazyTensor`.
#' @return A boolean, TRUE or FALSE.
#' @examples
#' \dontrun{
#' # basic example
#' D <- 3
#' M <- 100
#' x <- matrix(runif(M * D), M, D)
#' z <- matrix(1i^(-6:5), nrow = 4) # complex 4x3 matrix
#' 
#' # create LazyTensor and ComplexLazyTensor
#' x_i <- LazyTensor(x, index = 'i')
#' z_i <- LazyTensor(z, index = 'i', is_complex = TRUE) # ComplexLazyTensor
#' 
#' # call is.ComplexLazyTensor
#' is.ComplexLazyTensor(z_i) # returns TRUE
#' is.ComplexLazyTensor(x_i) # returns FALSE
#' }
#' @export
is.ComplexLazyTensor <- function(x){
    return("ComplexLazyTensor" %in% class(x))
}


#' is.LazyParameter?
#' @description
#' Checks whether the given input is a `LazyTensor` encoding
#' a single scalar value. That is, if the input is a fixed parameter
#' `LazyTensor` of dimension 1.
#' @details If `x` is a fixed scalar parameter `LazyTensor`,
#' `is.LazyParameter(x)` returns TRUE, else, returns FALSE.
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param x A `LazyTensor` object that we want to know if it is a
#' `LazyParameter`.
#' @return A boolean, TRUE or FALSE.
#' @examples
#' \dontrun{
#' # basic example
#' scal <- 3.14
#' cplx <- 2 + 3i
#' v <- rep(3, 10)
#' x <- matrix(runif(100 * 3), 100, 3)
#' 
#' # create LazyTensor and ComplexLazyTensor
#' scal_LT <- LazyTensor(scal)
#' cplx_LT <- LazyTensor(cplx)
#' v_LT <- LazyTensor(v)
#' x_i <- LazyTensor(x, index = 'i')
#' 
#' # call is.LazyParameter
#' is.LazyParameter(scal_LT) # returns TRUE
#' is.LazyParameter(cplx_LT) # returns FALSE
#' is.LazyParameter(v_LT) # returns FALSE
#' is.LazyParameter(x_i) # returns FALSE
#' }
#' @export
is.LazyParameter <- function(x) {
    if(!is.LazyTensor(x)) {
        stop("`x` input must be a LazyTensor.")
    }
    
    bool_grep_int <- grep("IntCst\\(.*\\)", x$formula)
    if(any(bool_grep_int)) {
        return(TRUE)
    }
    
    return((length(x$args) == 1) && any(grep(".*=Pm\\(1\\)", x$args)))
}

#' is.ComplexLazyParameter?
#' @description
#' Checks whether the given input is a `ComplexLazyTensor` encoding
#' a single complex value. That is, if the input is a fixed parameter
#' `ComplexLazyTensor` of dimension 1.
#' @details If `x` is a fixed parameter `ComplexLazyTensor` encoding a
#' single complex value, `is.ComplexLazyParameter(x)`
#' returns TRUE, else, returns FALSE.
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param x A `LazyTensor` object that we want to know if
#' it is a `ComplexLazyParameter`.
#' @return A boolean, TRUE or FALSE.
#' @examples
#' \dontrun{
#' # basic example
#' scal <- 3.14
#' cplx <- 2 + 3i
#' v <- rep(3 + 7i, 10)
#' z <- matrix(2 + 1i^ (-6:5), nrow = 4)
#' 
#' # create LazyTensor and ComplexLazyTensor
#' scal_LT <- LazyTensor(scal)
#' cplx_LT <- LazyTensor(cplx)
#' v_LT <- LazyTensor(v)
#' z_i <- LazyTensor(z, index = 'i')
#' 
#' # call is.ComplexLazyParameter
#' is.ComplexLazyParameter(scal_LT) # returns FALSE
#' is.ComplexLazyParameter(cplx_LT) # returns TRUE
#' is.ComplexLazyParameter(v_LT) # returns FALSE
#' is.ComplexLazyParameter(x_i) # returns FALSE
#' }
#' @export
is.ComplexLazyParameter <- function(x) {
    if(!is.LazyTensor(x)) {
        stop("`x` input must be a LazyTensor or a ComplexLazyTensor.")
    }
    
    res <- (is.ComplexLazyTensor(x) && length(x$args) == 1) && 
        any(grep(".*=Pm\\(2\\)", x$args))
    
    return(res)
}


#' is.LazyVector?
#' @description
#' Checks whether the given input is a `LazyTensor` encoding
#' a vector or a single value.
#' @details If `x` is a vector parameter `LazyTensor`,
#' `is.LazyVector(x)` returns TRUE, else, returns FALSE.
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param x A `LazyTensor` object that we want to know if it is a `LazyVector`.
#' @return A boolean, TRUE or FALSE.
#' @examples
#' \dontrun{
#' # basic example
#' scal <- 3.14
#' cplx <- 2 + 3i
#' v <- rep(3, 10)
#' x <- matrix(runif(100 * 3), 100, 3)
#' 
#' # create LazyTensor and ComplexLazyTensor
#' scal_LT <- LazyTensor(scal)
#' cplx_LT <- LazyTensor(cplx)
#' v_LT <- LazyTensor(v)
#' x_i <- LazyTensor(x, index = 'i')
#' 
#' # call is.LazyVector
#' is.LazyVector(scal_LT) # returns TRUE
#' is.LazyVector(cplx_LT) # returns TRUE
#' is.LazyVector(v_LT) # returns TRUE
#' is.LazyVector(x_i) # returns FALSE
#' }
#' @export
is.LazyVector <- function(x) {
    if(!is.LazyTensor(x)) {
        stop("`x` input must be a LazyTensor or a ComplexLazyTensor.")
    }
    return(any(grep(".*=Pm\\(.*\\)", x$args)))
}


#' is.LazyMatrix?
#' @description
#' Checks whether the given input is a `LazyTensor` encoding
#' a matrix. 
#' @details If `x` is a matrix `LazyTensor`,
#' `is.LazyMatrix(x)` returns TRUE, else, returns FALSE.
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param x A `LazyTensor` object that we want to know if it is a `LazyMatrix`.
#' @return A boolean, TRUE or FALSE.
#' @examples
#' \dontrun{
#' # basic example
#' scal <- 3.14
#' cplx <- 2 + 3i
#' v <- rep(3, 10)
#' x <- matrix(runif(100 * 3), 100, 3)
#' 
#' # create LazyTensor and ComplexLazyTensor
#' scal_LT <- LazyTensor(scal)
#' cplx_LT <- LazyTensor(cplx)
#' v_LT <- LazyTensor(v)
#' x_i <- LazyTensor(x, index = 'i')
#' 
#' # call is.LazyMatrix
#' is.LazyMatrix(scal_LT) # returns FALSE
#' is.LazyMatrix(cplx_LT) # returns FALSE
#' is.LazyMatrix(v_LT) # returns FALSE
#' is.LazyMatrix(x_i) # returns TRUE
#' }
#' @export
is.LazyMatrix <- function(x) {
    if(!is.LazyTensor(x)) {
        stop("`x` input must be a LazyTensor or a ComplexLazyTensor.")
    }
    return(any(grep(".*=V.\\(.*\\)", x$args)))
}


#' Scalar integer test.
#' @description
#' Checks whether the given input is a scalar `integer` or not.
#' @details If `x` is a scalar`integer`, `is.int(x)` returns TRUE, 
#' else, returns FALSE.
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param x An object that we want to know if it is an `integer`.
#' @return A boolean, TRUE or FALSE.
#' @examples
#' \dontrun{
#' # basic example
#' A <- 3
#' B <- 3.4
#' C <- rep(3, 10)
#' 
#' is.int(A)  # returns TRUE
#' is.int(B)  # returns FALSE
#' is.int(C)  # returns FALSE
#' }
#' @export
is.int <- function(x) {
    res <- (is.numeric(x) && length(x) == 1) && ((as.integer(x) - x) == 0)
    return(res)
}




# GLOBAL CHECKS ================================================================


#' Get inner dimension.
#' @keywords internal
#' @description
#' Returns the inner dimension of a given `LazyTensor`.
#' @details If `x` is a `LazyTensor`, `get_inner_dim(x)` returns an integer
#' corresponding to the inner dimension of `x`.
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param x A `LazyTensor`.
#' @return An integer corresponding to the inner dimension of `x`.
#' @examples
#' \dontrun{
#' # basic example
#' D <- 3
#' M <- 100
#' s <- 0.25
#' x <- matrix(runif(M * D), M, D)
#' 
#' # create LazyTensor
#' x_i <- LazyTensor(x, index = 'i')
#' Pm_s <- LazyTensor(s, index = NA)
#' 
#' # call get_inner_dim
#' get_inner_dim(x_i) # returns 3
#' get_inner_dim(Pm_s) # returns 1
#' }
get_inner_dim <- function(x) {
    # Grab `x` inner dimension.
    # `x` must be a LazyTensor or a ComplexLazyTensor.
    if(is.int(x)) {
        return(1)
    }
    
    if(!is.LazyTensor(x))
        stop("`x` input argument should be a LazyTensor or a ComplexLazyTensor.")
    
    if(length(x$args) == 1) {
        end_x_inner_dim <- sub(".*\\(", "", x$args)
        x_inner_dim <- substr(end_x_inner_dim, 1, nchar(end_x_inner_dim) - 1)
        x_inner_dim <- as.integer(x_inner_dim)
    }
    
    else {
        x_inner_dim <- x$dimres
    }
    
    if(is.ComplexLazyTensor(x)) {
        # divide by 2 because of complex casting
        x_inner_dim <- (x_inner_dim / 2)
    }
    
    return(x_inner_dim)
}


#' Check inner dimensions for binary or ternary operations.
#' @keywords internal
#' @description
#' Verifies that the inner dimensions of two or three given `LazyTensor` 
#' are the same.
#' @details If `x` and `y` are of class `LazyTensor` or `ComplexLazyTensor`,
#' `check_inner_dim(x, y, check_type = "same")` returns `TRUE` if `x` and `y`
#' inner dimensions are the same, and `FALSE` otherwise, while
#' `check_inner_dim(x, y, check_type = "sameor1")` returns `TRUE` if `x` and `y`
#' inner dimensions are the same or if at least one of these equals 1,
#' and `FALSE` otherwise.
#' Same idea with a third input `z`.
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param x A `LazyTensor` or a `ComplexLazyTensor`.
#' @param y A `LazyTensor` or a `ComplexLazyTensor`.
#' @param z A `LazyTensor` or a `ComplexLazyTensor` (optional, default = NA).
#' @param check_type A character string among "same" and "sameor1" (default),
#' to specify the desired type of inner dimension verification 
#' (see @details section).
#' @return A boolean TRUE or FALSE.
check_inner_dim <- function(x, y, z = NA, check_type = "sameor1") {
    # Inputs must be LazyTensors or ComplexLazyTensors.
    if(!is.LazyTensor(x) || !is.LazyTensor(y)) {
        stop(
            "Input arguments should be of class 'LazyTensor' or 'ComplexLazyTensor'."
        )
    }
    if(!is.na(z)[1]) {
        if(!is.LazyTensor(z) && !is.ComplexLazyTensor(z)) {
            stop(
                "Input arguments should be of class 'LazyTensor' or 'ComplexLazyTensor'."
            )
        }
    }
    
    x_inner_dim <- x$dimres
    y_inner_dim <- y$dimres
    
    if(is.na(z)[1]) {
        # Check whether if x and y inner dimensions are the same or if at least one 
        # of these equals 1.
        if(check_type == "sameor1") {
            res <- ((x_inner_dim == y_inner_dim) || 
                        ((x_inner_dim == 1) || (y_inner_dim == 1)))
        }
        if(check_type == "same") {
            res <- ((x_inner_dim == y_inner_dim))
        }
    }
    else {
        z_inner_dim <- z$dimres
        # Check whether x, y and z inner dimensions are the same or if at least 
        # one of these equals 1.
        if(check_type == "sameor1") {
            unique_dims <- unique(append(c(x_inner_dim, y_inner_dim, z_inner_dim), 1))
            res <- length(unique_dims) <= 2
        }
        if(check_type == "same") {
            dims <- c(x_inner_dim, y_inner_dim, z_inner_dim)
            res <- all(dims == rep(x_inner_dim, length(dims)))
        }
    }
    return(res)
}


#' Check index.
#' @keywords internal
#' @description
#' Checks index for operation.
#' @details `check_index(index)` will return a boolean to check if `index` is 
#' a character and corresponding to `"i"` or `"j"`.
#' - if `index = "i"`, return `TRUE`.
#' - if `index = "j"`, return `TRUE`.
#' - else return `FALSE`.
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param  index to check.
#' @return A boolean TRUE or FALSE.
check_index <- function(index){
    res <- is.character(index) && (index %in% c("i", "j"))
    return(res)
}

#' Index to int.
#' @keywords internal
#' @description
#' Transforms `string` index input into integer.
#' @details `index_to_int(index)` returns an `integer`: `1` if 
#' `index == "i"` and `0` if `index == "j"`.
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param index A `character` that should be either `i` or `j`.
#' @return An `integer`.
index_to_int <- function(index) {
    if(!check_index(index)) {
        stop(paste0("`index` input argument should be a character,",
                    " either 'i' or 'j'."))
    }
    if(index == "i")
        res <- 0
    else
        res <- 1
    return(res)
}


# Reduction---------------------------------------------------------------------

#' Identifier.
#' @keywords internal
#' @description 
#' Returns the identifier/label of a `LazyTensor` which is contained in `arg`.
#' @details `identifier(arg)` will extract a unique identifier of the form
#' `"A0x.*"` from the argument `arg` which has the form `"A0x.*=Vi(3)"`.
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param arg A `string` corresponding to an element of the attribute `args` of 
#' a `LazyTensor`.
#' @return A `string`.
#' @examples
#' \dontrun{
#' x <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows, 3 columns
#' x_i <- LazyTensor(x, index = 'i')   # creating LazyTensor from matrix x, 
#'                                     # indexed by 'i'
#' arg <- x_i$args[1]                  # argument of the form "A0x.*=Vi(3)"
#' id <- identifier(arg)               # extracts "A0x.*"
#' }
identifier <- function(arg){
    if(!is.character(arg)) {
        stop("`arg` input argument should be a character string.")
    }
    id <- str_extract(string = arg, pattern = "A0x.*=")
    id <- substr(id,1,nchar(id)-1)
    return(id)
}


#' Fix variables.
#' @keywords internal
#' @description Assigns final labels to each variable for the `KeOps` routine.
#' @details `fix_variables(x)` will change the identifiers of `x` variables in 
#' `x$args` and `x$formula` into simpler ordered labels of the form `V<n>` where
#' `n` is the apparition order of the variable in the formula.
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param x  A `LazyTensor` or a `ComplexLazyTensor`.
#' @return A `LazyTensor` or a `ComplexLazyTensor`.
#' @examples
#' \dontrun{
#' x <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows, 3 columns
#' y <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows, 3 columns
#' x_i <- LazyTensor(x, index = 'i')   # creating LazyTensor from matrix x, 
#'                                     # indexed by 'i'
#' y_j <- LazyTensor(y, index = 'j')   # creating LazyTensor from matrix y, 
#'                                     # indexed by 'j'
#' 
#' a <- x_i + y_j       # combination of LazyTensors with variable labels 
#'                      # of the form "A0x.*"
#' 
#' b <- fix_variables(a) # combination of LazyTensors with variable labels 
#'                      # of the form "V0" and "V1"
#' b$formula            # returns "V0+V1"
#' b$args               # returns a vector containing "V0=Vi(3)" and "V1=Vj(3)"
#' }
fix_variables <- function(x, is_opt = FALSE) {
    if(!is.LazyTensor(x)) {
        stop("`x` input must be a LazyTensor or a ComplexLazyTensor.")
    }
    
    # Should do nothing to IntCst-only LazyTensors:
    if(is.null(x$args)) {
        # TRUE only if x is an IntCst LazyTensor (or a combination of)
        return(x)
    }
    tmp <- x
    for(i in 1:length(tmp$args)) {
        
        suffix_arg <- str_extract(string = tmp$args[i], pattern = "=.*")
        suffix_arg <- substr(suffix_arg, 2, nchar(suffix_arg))
        var_dim <- as.numeric(str_extract(string = tmp$arg[i],
                                          pattern = "(?<=\\()[0-9]+")
        )
        
        if(!is_opt) {
            tag <- paste("V", i-1, sep = "")
        }
        else {
            tag <- paste("OptV", i-1, sep = "")
        }
        
        id <- identifier(tmp$args[i])
        tmp$formula <- str_replace_all(tmp$formula, id, tag)
        tmp$args <- str_replace(tmp$args, id, tag)
    }
    
    return(tmp)
}


#' Fix internal reduction operation.
#' @keywords internal
#' @description Returns the internal reduction operation.
#' 
#' `r lifecycle::badge("deprecated")` `fix_op_reduction()` is not useful 
#' anymore because `rkeops` is using `pykeops` Python package as an internal
#' engine where this is managed.
#' @details `fix_op_reduction(reduction_op, with_weight)` will return the 
#' internal reduction operation according to `reduction_op` and a possible 
#' optional weight argument. Some advance operations defined at user level use, 
#' in fact, other internal reductions:
#' - If `reduction_op == "LogSumExp"`, the internal reduction operation
#'   is `"Max_SumShiftExp"` or `"Max_SumShiftExpWeight"` depending on 
#'   `with_weight`;
#' - If `reduction_op == "SumSoftMax"`, the internal reduction operation
#'   is `"Max_SumShiftExpWeight"`;
#' - Else, for every other value of `reduction_op`, the internal 
#'   reduction operation is `reduction_op`.
#' 
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param reduction_op A text `string` corresponding to a reduction.
#' @param with_weight A `boolean` which is `TRUE` when there is an optional 
#' argument corresponding to a weight argument.
#' @return A text `string`.
fix_op_reduction <- function(reduction_op, with_weight = FALSE) {
    
    lifecycle::deprecate_warn(
        "2.0.0", "fix_op_reduction()", 
        detail = "Not used anymore. See documentation")
    
    reduction_op_internal <- reduction_op
    
    if(reduction_op == "SumSoftMaxWeight") {
        # SumSoftMaxWeight relies on KeOps Max_SumShiftExpWeight reduction.
        reduction_op_internal <- "Max_SumShiftExpWeight"
    }
    else if(reduction_op == "LogSumExp") {
        # LogSumExp relies also on Max_SumShiftExp or 
        # Max_SumShiftExpWeight reductions
        if(with_weight) {
            # here we want to compute a log-sum-exp with weights:
            # log(sum_j(exp(f_ij)g_ij))
            reduction_op_internal <- "Max_SumShiftExpWeight"
        } else {
            # here we want to compute a usual log-sum-exp:
            # log(sum_j(exp(f_ij)))
            reduction_op_internal <- "Max_SumShiftExp"
        }
    }
    return(reduction_op_internal)
}


#' Preprocess reduction operation.
#' @keywords internal
#' @description
#' Returns a `function` for a reduction to a `LazyTensor` and it is called in 
#' `rkeops::reduction.LazyTensor()`.
#' @details `preprocess_reduction(x, opstr, index)` will:
#' - if `index = "i"`, return a `function` corresponding to the 
#'   `opstr` reduction of `x` over the `i` indexes;
#' - if `index = "j"`, return a `function` corresponding to the 
#'   `opstr` reduction of `x` over the `j` indexes.
#' 
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param x A `LazyTensor` or a `ComplexLazyTensor`.
#' @param opstr A `string` formula (like "Sum" or "Max").
#' @param index A `character` that should be either `i` or `j` to specify 
#' whether if the reduction is indexed by `i` (rows), or `j` (columns).
#' @param opt_arg An optional argument: an `integer` (e.g. for "Kmin" 
#' reduction), a `character`, a `LazyTensor` or a `ComplexLazyTensor`. `NULL` 
#' if not used (default).
#' @return A `function`.
#' @seealso [rkeops::reduction.LazyTensor()]
#' @examples
#' \dontrun{
#' x <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows, 3 columns
#' x_i <- LazyTensor(x, index = 'i')   # creating LazyTensor from matrix x, 
#'                                     # indexed by 'i'
#' 
#' op <- preprocess_reduction(x_i, "Sum", "i")
#' }
preprocess_reduction <- function(x, opstr, index, opt_arg = NULL) {
    
    # init
    formula <- NULL
    
    tag <- index_to_int(index)
    
    # Change the identifiers of every variables for the KeOps routine
    tmp <- fix_variables(x)
    args <- tmp$args
    # change internal reduction operation if needed (DEPRECATED)
    # fix_op_reducrtion() not neeeded anymore
    opstr_internal <- opstr
    
    # manage optional arguments
    if(!is.null(opt_arg)) {
        if(is.LazyTensor(opt_arg)) {
            tmp_opt <- fix_variables(opt_arg, is_opt = TRUE)
            # put `opt_arg$formula` at the middle of the formula
            formula <- paste(opstr_internal,  "_Reduction(",  tmp$formula, 
                             ",", tmp_opt$formula, ",", tag, ")", sep = "")
            args <- c(tmp$args, tmp_opt$args)
        } else if(is.int(opt_arg)) {
            # put `opt_arg` in the middle of the formula
            formula <- paste(opstr_internal,  "_Reduction(",  tmp$formula, 
                             ",",  opt_arg, ",", tag, ")", sep = "")
        }
        
    } else {
        formula <- paste(opstr_internal, "_Reduction(", tmp$formula, ",", 
                         tag, ")", sep = "")
    }
    
    op <- keops_kernel(formula, args)
    return(op)
}


#' Warning for ComplexLazyTensor/LazyTensor operations.
#' @keywords internal
#' @description 
#' Returns a warning message when binary operations are used with a
#' `LazyTensor` and a `ComplexLazyTensor`. These operations might not work
#' with the current rkeops version.
#' 
#' This function is only called in `real2complex.LazyTensor`, which is only
#' used with binary operations involving a `LazyTensor` and
#' a `ComplexLazyTensor`.
#' 
#' @author Chloe Serre-Combe, Amelie Vernay
#' @return A warning message.
cplx_warning <- function(warn = TRUE) {
    if(warn) {
        msg <- paste(
            "Operations involving both LazyTensors and",
            "ComplexLazyTensors may not work with the actual rkeops version.",
            "This should be fixed in a future release.", sep = " "
        )
        warning(msg)
    }
}
