
# LAZYTENSOR CONFIGURATION =====================================================


# LazyTensor -------------------------------------------------------------------
#' Build and return a LazyTensor object.
#' @description
#' `LazyTensor`s objects are wrappers around R matrices or vectors that are used 
#' to create symbolic formulas for the KeOps reduction operations.
#' @details
#' The `LazyTensor()` function builds a `LazyTensor`, which is a 
#' list containing the following elements:
#' \itemize{
#'     \item{**formula**:}{ A string defining the mathematical operation to 
#'     be computed by the KeOps routine}
#'     \item{**args**:}{ A vector of arguments containing a unique identifier 
#'     associated to the type of the argument :
#'     \itemize{
#'         \item{**Vi(n)**:}{ vector indexed by **i** of dim **n**}
#'         \item{**Vj(n)**:}{ vector indexed by **j** of dim **n**}
#'         \item{**Pm(n)**:}{ fixed parameter of dim **n**}
#'     }}
#'     \item{**vars**:}{ A list of R matrices which will be the inputs of the 
#'                       KeOps routine}
#'     \item{**dimres**:}{ An integer corresponding to the inner dimension of
#'                        the LazyTensor. **dimres** is used when creating new
#'                        LazyTensors that result from operations,
#'                        to keep track of the dimension.}
#' }
#' 
#' 
#' **Note**
#' 
#' Setting the argument `is_complex` to `TRUE` will build a `ComplexLazyTensor`,
#' which is also a `LazyTensor`. Run `browseVignettes("rkeops")` and see 
#' "RKeOps LazyTensor" vignette for further details on how `ComplexLazyTensor`s
#' are build.
#' 
#' 
#' **Alternatives**
#' 
#' \itemize{
#'    \item LazyTensor(x, "i") is equivalent to Vi(x) (see **Vi()** function)
#'    \item LazyTensor(x, "j") is equivalent to Vi(x) (see **Vj()** function)
#'    \item LazyTensor(x) is equivalent to Pm(x) (see **Pm()** function)
#' }
#'
#' Run `browseVignettes("rkeops")` to access the vignettes and see how to use
#' `LazyTensor`s.
#' @author Joan Glaunes, Chloe Serre-Combe, Amelie Vernay
#' @param x A matrix or a vector of numeric values, or a scalar value
#' @param index A text string that should be either **i** or **j**, or an **NA** 
#' value (the default), to specify whether if the **x** variable is indexed 
#' by **i** (rows), by **j** (columns), or is a fixed parameter across indices.
#' If **x** is a matrix, **index** must be **i** or **j**.
#' @param is_complex A boolean (default is FALSE). Whether if we want to create a
#' `ComplexLazyTensor` (is_complex = TRUE) or a `LazyTensor` (is_complex = FALSE).
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
  
  # 1) input is a matrix, treated as indexed variable, so index must be "i" or "j"
  if(is.matrix(x)) {
    d <- ncol(x)
    if(index == "i")
      cat = "Vi"
    else
      cat = "Vj"
  }
  # 2) else we assume x is a numeric vector, treated as parameter,
  # then converted to matrix
  else {
    d <- length(x)
    cat <- "Pm"
  }
  
  # Now we define "formula", a string specifying the variable for KeOps C++ codes.
  if(is.int(x)) {
    var_name <- paste("IntCst(", as.character(x), ")", sep = "")
  }
  else {
    var_name <- paste("A", address(x), index, sep = "")
  }
  formula <- var_name
  vars <- list(x)  # vars lists all actual matrices necessary to evaluate 
                   # the current formula, here only one.
  
  if(is_complex) {
    args <- str_c(var_name, "=", cat, "(", 2 * d, ")")
    
    # build ComplexLazyTensor
    res <- list(formula = formula, args = args, vars = vars)
    
    # format vars in a "complex" way
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
      # Parameter LazyTensors have vector as vars (but apparently it
      # doesn't matter is we leave it as a matrix...)
      # Uncomment below if needed.
      #ReImZ <- as.vector(ReImZ)
    }
    res$vars[[1]] <- ReImZ
    
    # add ComplexLazyTensor class
    class(res) <- c("ComplexLazyTensor", "LazyTensor")
  }
  else {
    args <- str_c(var_name, "=", cat, "(", d, ")")
    # finally we build and return the LazyTensor object
    res <- list(formula = formula, args = args, vars = vars)
    class(res) <- "LazyTensor"
  }
  
  # add inner dimension
  res$dimres <- get_inner_dim(res)
  
  return(res)
}


# Vi ---------------------------------------------------------------------------

#' Wrapper LazyTensor indexed by "i".
#' @description
#' Simple wrapper that return an instantiation of `LazyTensor` indexed by "i".
#' Equivalent to `LazyTensor(x, index = "i")`.
#' @details See `?LazyTensor` for more details.
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param x A matrix of numeric values, or a scalar value.
#' @param is_complex A boolean (default is FALSE). Whether if we want to create a
#' `ComplexLazyTensor` (is_complex = TRUE) or a `LazyTensor` (is_complex = FALSE).
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
#' Simple wrapper that return an instantiation of `LazyTensor` indexed by "j".
#' Equivalent to `LazyTensor(x, index = "j")`.
#' @details See `?LazyTensor` for more details.
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param x A matrix of numeric values.
#' @param is_complex A boolean (default is FALSE). Whether if we want to create a
#' `ComplexLazyTensor` (is_complex = TRUE) or a `LazyTensor` (is_complex = FALSE).
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
#' Simple wrapper that return an instantiation of a fixed parameter `LazyTensor`.
#' Equivalent to `LazyTensor(x)`.
#' @details See `?LazyTensor` for more details.
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param x A vector or a scalar value.
#' @param is_complex A boolean (default is FALSE). Whether if we want to create a
#' `ComplexLazyTensor` (is_complex = TRUE) or a `LazyTensor` (is_complex = FALSE).
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
#' Symbolically applies **opstr** operation to **x**.
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param x A `LazyTensor`, a `ComplexLazyTensor`, a vector of numeric values, 
#' or a scalar value.
#' @param opstr A text string corresponding to an operation.
#' @param opt_arg An optional argument which can be a scalar value.
#' @param opt_arg2 An optional argument which can be a scalar value.
#' @param res_type NA (default) or a character string among "LazyTensor" and 
#' "ComplexLazyTensor", to specify if a change of class is required for the result.
#' (Useful especially when dealing with complex-to-real or
#' real-to-complex functions).
#' @param dim_res NA (default) or an integer corresponding to the inner
#' dimension of the output `LazyTensor`. If NA, **dim_res** is set to the
#' inner dimension of the input `LazyTensor`.
#' @return An object of class "LazyTensor" or "ComplexLazyTensor".
#' @examples
#' \dontrun{
#' x <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows and 3 columns
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
  
  ## result dimension
  #if(is.na(dim_res)) {
  #  dim_res <- get_inner_dim(x)
  #}
  
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
  
  res <- list(formula = formula, args = x$args, vars = x$vars)
  
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
#' Symbolically applies **opstr** operation to **x** and **y**.
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param x A `LazyTensor`, a `ComplexLazyTensor`, a vector of numeric values, or 
#' a scalar value.
#' @param y A `LazyTensor`, a `ComplexLazyTensor`, a vector of numeric values, or 
#' a scalar value.
#' @param opstr A text string corresponding to an operation.
#' @param is_operator A boolean used to specify if **opstr** is an operator like 
#' ``+``, ``-`` or a "genuine" function.
#' @param dim_check_type A string to specify if, and how, we should check input 
#' dimensions.
#' Supported values are:
#' \itemize{
#'    \item {**"same"**:}{ **x** and **y** should have the same inner dimension;}
#'    \item {**"sameor1"** (default):}{ **x** and **y** should have the same inner 
#'    dimension or at least one of them should be of dimension 1;}
#'    \item {**NA**:}{ no dimension restriction.}
#' }
#' @param res_type NA (default) or a character string among "LazyTensor" and 
#' "ComplexLazyTensor", to specify if a change of class is required for the result.
#' (Useful especially when dealing with complex-to-real or
#' real-to-complex functions).
#' @param dim_res NA (default) or an integer corresponding to the inner
#' dimension of the output `LazyTensor`. If NA, **dim_res** is set to the
#' maximum between the inner dimensions of the two input `LazyTensor`s.
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
        "`x` input argument should be a LazyTensor, a vector or a scalar.",
        "\nIf you want to use a matrix, convert it to LazyTensor first.", 
        sep = ""
      )
    )
  
  if(is.matrix(y))
    stop(
      paste(
        "`y` input argument should be a LazyTensor, a vector or a scalar.",
        "\nIf you want to use a matrix, convert it to LazyTensor first.", 
        sep = ""
      )
    )
  
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
  
  if(is.numeric(x) || is.complex(x))
    x <- LazyTensor(x)
  
  if(is.numeric(y) || is.complex(y))
    y <- LazyTensor(y)
  
  # check dimensions
  if (!is.na(dim_check_type)) {
    if(dim_check_type == "sameor1") {
      if (!check_inner_dim(x, y, check_type = dim_check_type)) {
        stop(
          paste(
            "Operation `", opstr, 
            "` expects inputs of the same dimension or dimension 1. Received ",
            get_inner_dim(x), " and ", get_inner_dim(y), ".", sep = ""
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
            get_inner_dim(x), " and ", get_inner_dim(y), ".", sep = ""
          )
        )
      }
    }
  }
  
  ## result dimension
  #if(is.na(dim_res)) {
  #  dim_res <- max(c(get_inner_dim(x), get_inner_dim(y)))
  #}
  
  # result dimension
  if(is.na(dim_res)) {
    dim_res <- max(c(x$dimres, y$dimres))
  }
  
  # Set `as.integer(dim_res)` to avoid printing potential
  # decimal zero: 4.0, 5.0, and so on...
  # If dim_res has a non zero decimal, the function stops anyway.
  dim_res <- as.integer(dim_res)
  
  # special formula for operator
  if(is_operator)
    formula <- paste(x$formula, opstr, y$formula, sep = "")
  
  else if(!is_operator && is.na(opt_arg))
    formula <- paste(opstr, "(", x$formula, ",", y$formula, ")", sep = "")
  
  else if(!is_operator && !is.na(opt_arg))
    formula <- paste(opstr, "(", x$formula, ",", opt_arg$formula, ",",
                     y$formula, ")", sep = "")
  
  vars <- c(x$vars, y$vars)
  args <- unique(c(x$args, y$args))
  dimres <- dim_res
  
  res <- list(formula = formula, args = args, vars = vars, dimres = dimres)
  res$vars <- res$vars[!duplicated(res$vars)] # remove doublon
  
  if(!is.na(res_type[1]))
    class(res) <- res_type
  else if((is.ComplexLazyTensor(x) || is.ComplexLazyTensor(y)) 
          || is.ComplexLazyTensor(opt_arg)) {
    class(res) <- c("ComplexLazyTensor", "LazyTensor")
  }
  else
    class(res) <- class(x)
  
  return(res)
}


# ternary ----------------------------------------------------------------------

#' Build a ternary operation
#' @description
#' Symbolically applies **opstr** operation to **x**, **y** and **z**.
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
#' \itemize{
#'    \item {**"same"**:}{ **x** and **y** should have the same inner dimension;}
#'    \item {**"sameor1"** (default):}{ **x** and **y** should have the same inner 
#'    dimension or at least one of them should be of dimension 1;}
#'    \item {**NA**:}{ no dimension restriction.}
#' }
#' @param dim_res NA (default) or an integer corresponding to the inner
#' dimension of the output `LazyTensor`. If NA, **dim_res** is set to the
#' maximum between the inner dimensions of the three input `LazyTensor`s.
#' @return An object of class "LazyTensor".
#' @examples
#' \dontrun{
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
  
  ## result dimension
  #if(is.na(dim_res)) {
  #  dim_res <- max(c(get_inner_dim(x), get_inner_dim(y), get_inner_dim(z)))
  #}
  
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
  
  vars <- c(x$vars, y$vars, z$vars)
  args <- unique(c(x$args, y$args, z$args))
  dimres <- dim_res
  
  res <- list(formula = formula, args = args, vars = vars, dimres = dimres)
  res$vars <- res$vars[!duplicated(res$vars)] # remove doublon
  
  if(is.ComplexLazyTensor(x) || is.ComplexLazyTensor(y))
    class(res) <- c("ComplexLazyTensor", "LazyTensor")
  else
    class(res) <- class(x)
  
  return(res)
}


# TYPE CHECKING ================================================================


#' is.LazyTensor?
#' @description
#' Checks whether if the given input is a `LazyTensor` or not.
#' @details If `x` is a `LazyTensor`, `is.LazyTensor(x)` returns TRUE, else, 
#' returns FALSE.
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param x An object we want to know if it is a `LazyTensor`.
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
#' Checks whether if the given input is a `ComplexLazyTensor` or not.
#' @details If `x` is a `ComplexLazyTensor`, `is.ComplexLazyTensor(x)` 
#' returns TRUE, else, returns FALSE.
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param x An object we want to know if it is a `ComplexLazyTensor`.
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


#' is.LazyScalar?
#' @description
#' Checks whether if the given input is a `LazyTensor` encoding
#' a single scalar value. That is, if the input is a fixed parameter
#' `LazyTensor` of dimension 1.
#' @details If `x` is a fixed parameter `LazyTensor` with dimension 1,
#' `is.LazyScalar(x)` returns TRUE, else, returns FALSE.
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param x An object we want to know if it is a `LazyScalar`.
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
#' # call is.LazyScalar
#' is.LazyScalar(scal_LT) # returns TRUE
#' is.LazyScalar(cplx_LT) # returns FALSE
#' is.LazyScalar(v_LT) # returns FALSE
#' is.LazyScalar(x_i) # returns FALSE
#' }
#' @export
is.LazyScalar <- function(x) {
  return((length(x$args) == 1) && any(grep(".*=Pm\\(1\\)", x$args)))
}

#' is.ComplexLazyScalar?
#' @description
#' Checks whether if the given input is a `ComplexLazyTensor` encoding
#' a single complex value. That is, if the input is a fixed parameter
#' `ComplexLazyTensor` of dimension 1.
#' @details If `x` is a fixed parameter `ComplexLazyTensor` encoding a
#' single complex value, `is.ComplexLazyScalar(x)`
#' returns TRUE, else, returns FALSE.
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param x An object we want to know if it is a `ComplexLazyScalar`.
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
#' # call is.ComplexLazyScalar
#' is.ComplexLazyScalar(scal_LT) # returns FALSE
#' is.ComplexLazyScalar(cplx_LT) # returns TRUE
#' is.ComplexLazyScalar(v_LT) # returns FALSE
#' is.ComplexLazyScalar(x_i) # returns FALSE
#' }
#' @export
is.ComplexLazyScalar <- function(x) {
  res <- (is.ComplexLazyTensor(x) && length(x$args) == 1) && 
    any(grep(".*=Pm\\(2\\)", x$args))
  return(res)
}


#' is.LazyVector?
#' @description
#' Checks whether if the given input is a `LazyTensor` encoding
#' a vector or a single value.
#' @details If `x` is a vector parameter `LazyTensor`,
#' `is.LazyVector(x)` returns TRUE, else, returns FALSE.
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param x An object we want to know if it is a `LazyVector`.
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
  return(any(grep(".*=Pm\\(.*\\)", x$args)))
}


#' is.LazyMatrix?
#' @description
#' Checks whether if the given input is a `LazyTensor` encoding
#' a matrix. 
#' @details If `x` is a matrix `LazyTensor`,
#' `is.LazyMatrix(x)` returns TRUE, else, returns FALSE.
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param x An object we want to know if it is a `LazyMatrix`.
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
  return(any(grep(".*=V.\\(.*\\)", x$args)))
}


#' Scalar integer test.
#' @description
#' Checks whether if the given input is a scalar `integer` or not.
#' @details If `x` is a scalar`integer`, `is.int(x)` returns TRUE, 
#' else, returns FALSE.
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param x An object we want to know if it is a `integer`.
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
  if(!is.LazyTensor(x))
    stop("`x` input argument should be a LazyTensor or a ComplexLazyTensor.")
  
  #end_x_inner_dim <- sub(".*\\(", "", x$args)
  #x_inner_dim <- substr(end_x_inner_dim, 1, nchar(end_x_inner_dim) - 1)
  #x_inner_dim <- as.integer(x_inner_dim)
  
  if(length(x$args) == 1) {
    end_x_inner_dim <- sub(".*\\(", "", x$args)
    x_inner_dim <- substr(end_x_inner_dim, 1, nchar(end_x_inner_dim) - 1)
    x_inner_dim <- as.integer(x_inner_dim)
  }
  
  else {
    x_inner_dim <- x$dimres
  }
  
  if(is.ComplexLazyTensor(x))
    x_inner_dim <- (x_inner_dim / 2)
  
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
    z_inner_dim <- get_inner_dim(z)
    # Check whether if x, y and z inner dimensions are the same or if at least 
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
#' Check index for operation.
#' @details `check_index(index)` will return a boolean to check if **index** is 
#' a character and corresponding to **"i"** or **"j"**.
#' \itemize{
#'   \item if **index = "i"**, return **TRUE**.
#'   \item if **index = "j"**, return **TRUE**.
#'   \item else return **FALSE**.
#' }
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param  index to check.
#' @return A boolean TRUE or FALSE.
check_index <- function(index){
  res <- is.character(index) && (index %in% c("i", "j"))
  return(res)
}


#' Index to int.
#' @description
#' Transform `string` index input into integer.
#' @details `index_to_int(index)` returns an `integer`: **1** if 
#' **index == "i"** and **0** if **index == "j"**.
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param index A `character` that should be either **i** or **j**.
#' @return An `integer`.
#' @export
index_to_int <- function(index) {
  if(!check_index(index)) {
    stop(paste0("`index` input argument should be a character,",
                " either 'i' or 'j'."))
  }
  if(index == "i")
    res <- 1
  else
    res <- 0
  return(res)
}


# Reduction---------------------------------------------------------------------




#' Preprocess reduction operation.
#' @keywords internal
#' @description
#' Returns a `function` for a reduction to a `LazyTensor` and it is called in 
#' `rkeops::reduction.LazyTensor()`.
#' @details `preprocess_reduction(x, opstr, index)` will :
#' \itemize{
#'   \item{ if **index = "i"**, return a `function` corresponding to the **opstr** 
#'   reduction of **x** over the "i" indexes;}
#'   \item{ if **index = "j"**, return a `function` corresponding to the **opstr** 
#'   reduction of **x** over the "j" indexes.}
#' }
#' @author Chloe Serre-Combe, Amelie Vernay
#' @param x A `LazyTensor` or a `ComplexLazyTensor`.
#' @param opstr A `string` formula (like "Sum" or "Max").
#' @param index A `character` that should be either **i** or **j** to specify 
#' whether if the reduction is indexed by **i** (rows), or **j** (columns).
#' @param opt_arg An optional argument : an `interger` (for "Kmin" reduction),
#' a `character`, `LazyTensor` or a `ComplexLazyTensor`.
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
#' @export
preprocess_reduction <- function(x, opstr, index, opt_arg = NA) {
  tag <- index_to_int(index)
  args <- x$args
  
  if(!any(is.na(opt_arg))) {
    if(is.LazyTensor(opt_arg)) {
      # put `opt_arg$formula` at the end of the formula
      formula <- paste( opstr,  "_Reduction(",  x$formula, 
                        ",",  tag, ",", opt_arg$formula, ")", sep = "")
      args <- c(x$args, opt_arg$args)
    }
    
    else if(is.int(opt_arg)) {
      # put `opt_arg` in the middle of the formula
      formula <- paste( opstr,  "_Reduction(",  x$formula, 
                        ",",  opt_arg, ",", tag, ")", sep = "")
    }
    
    else if(is.character(opt_arg)) {
      # put `opt_arg` at the end of the formula
      formula <- paste( opstr,  "_Reduction(",  x$formula, 
                        ",",  tag, ",", opt_arg, ")", sep = "")
    }
    
  }
  else {
    formula <- paste(opstr, "_Reduction(", x$formula, ",", 
                     tag, ")", sep = "")
  }
  
  op <- keops_kernel(formula, args)
  return(op)
}