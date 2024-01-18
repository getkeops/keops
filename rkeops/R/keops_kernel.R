#' Defines a new KeOps operator to compute a specific symbolic formula
#' 
#' @description
#' This function is the core of the KeOps library, it allows you to create 
#' new operators based on kernel operation and matrix reduction described as a 
#' mathematic formula.
#' 
#' @details
#' The use of the function `keops_kernel` is detailled in the vignettes, 
#' especially how to write formulae, specified input arguments, how to format 
#' data to apply the created operators, etc. Run `browseVignettes("rkeops")` 
#' to access the vignettes.
#' 
#' KeOps operators are defined thanks to formula, i.e. a text string describing 
#' the mathematical operations that you want to apply to your data, and a 
#' list defining the input arguments of your formula.
#' 
#' The function `keops_kernel()` compiles and imports a new operator that 
#' implements the formula given in input, it returns a function that can be 
#' used to compute the result of the formula on actual data.
#' 
#' The returned function expects a list of arguments, as data matrices, whose 
#' order corresponds to the order given in `args` to `keops_kernel()`. 
#' We use a list to avoid useless copies of data.
#' 
#' **Important**: Formula are assumed to include a reduction (e.g. 
#' `Sum_Reduction(,<axis>)`), with an `<axis>` parameter indicating on which
#' dimension is done the reduction:
#' - `0` means a reduction over `i` (meaning that the result is a `Vj` 
#' variable).
#' - `1` means a reduction over `j` (meaning that the result is a `Vi` 
#' variable).
#' 
#' **Note:** Data are input as a list, because list are references and since 
#' argument passing is done by copy in R, it is better to copy a list of 
#' reference than the actual input data, especially for big matrices.
#' 
#' You should be careful with the input dimension of your data, so that
#' it correspond to the input dimension specified in `args` 
#' (see inner or outer dimension in `browseVignettes("rkeops")`.
#' 
#' It is possible to compute partial derivatives of user defined operators 
#' with the function [rkeops::keops_grad()].
#' 
#' @author Ghislain Durif
#' 
#' @param formula text string, an operator formula (see Details).
#' @param args vector of text string, formula arguments (see Details).
#' @param sum_scheme character string, method used to sum up results for 
#' reductions. This option may be changed only with reductions 
#' `Sum`, `MaxSumShiftExp`, `LogSumExp`, `Max_SumShiftExpWeight`, 
#' `LogSumExpWeight`, `SumSoftMaxWeight`. Default value `"auto"` will set this 
#' option to `"block_sum"` for these reductions. Possible values are:
#' - `"direct_sum"`: direct summation.
#' - `"block_sum"`: use an intermediate accumulator in each block before 
#' accumulating in the output. This improves accuracy for large sized data.
#' - `"kahan_scheme"`: use Kahan summation algorithm to compensate for 
#' round-off errors. This improves accuracy for large sized data.
#' 
#' @return a function that can be used to compute the value of the 
#' symbolic formula on actual data. This function takes as input a list of 
#' data corresponding to the formula arguments and return the computed values 
#' (generally a vector or a matrix depending on the reduction). It has an 
#' additional character input parameter `inner_dim` indicating if the inner 
#' dimension (c.f. `browseVignettes("rkeops")`) corresponds to columns, i.e. 
#' `inner_dim="col"` (default), or rows, i.e. `inner_dim="row"`, in the data.
#' 
#' @importFrom stringr str_length str_count
#' @importFrom reticulate np_array source_python
#' @importFrom checkmate assert_choice assert_character assert_logical 
#' assert_string
#' 
#' @seealso [rkeops::keops_grad()]
#' 
#' @examples
#' \dontrun{
#' set_rkeops_options()
#' 
#' ## Example 1
#' # Defining a function that computes for each j the sum over i
#' # of the scalar products between `x_i` and `y_j` (both 3d vectors), 
#' # i.e. the sum over the rows of the result of the matrix product `X * t(Y)`
#' # where `x_i` and `y_j` are the respective rows of the matrices `X` and `Y`.
#' op <- keops_kernel(
#'     formula = "Sum_Reduction((x|y), 1)", args = c("x=Vi(3)", "y=Vj(3)"))
#' # data
#' nx <- 10
#' ny <- 15
#' # x_i = rows of the matrix X
#' X <- matrix(runif(nx*3), nrow=nx, ncol=3)
#' # y_j = rows of the matrix Y
#' Y <- matrix(runif(ny*3), nrow=ny, ncol=3)
#' # compute the result (here, by default `inner_dim="col"` and 
#' # columns corresponds to the inner dimension)
#' res <- op(list(X,Y))
#' 
#' ## Example 1 bis
#' # In example 1, the inner dimension (i.e. the common dimension of vectors 
#' # `x_i` and `y_j` corresponds to columns of the matrices `X` and `Y` resp.).
#' # We know consider the inner dimension to be the rows of the matrices `X` 
#' # and `Y`.
#' 
#' # data
#' nx <- 10
#' ny <- 15
#' # x_i = columns of the matrix X
#' X <- matrix(runif(nx*3), nrow=3, ncol=nx)
#' # y_j = columns of the matrix Y
#' Y <- matrix(runif(ny*3), nrow=3, ncol=ny)
#' # compute the result (we specify `inner_dim="row"` to indicate that the rows 
#' # corresponds to the inner dimension)
#' res <- op(list(X,Y), inner_dim="row")
#' 
#' ## Example 2
#' # Defining a function that computes the convolution with a Gaussian kernel 
#' # i.e. the sum over i of `e^(lambda * ||x_i - y_j||^2) * beta_j` where `x_i`, 
#' # `y_j` and `beta_j` are 3d vectors, and `lambda` is a scalar parameter.
#' op = keops_kernel(
#'     formula = "Sum_Reduction(Exp(lambda*SqNorm2(x-y))*beta, 1)",
#'     args = c("x=Vi(3)", "y=Vj(3)", "beta=Vj(3)", "lambda=Pm(1)"))
#' 
#' # data
#' nx <- 10
#' ny <- 15
#' # x_i = rows of the matrix X
#' X <- matrix(runif(nx*3), nrow=nx, ncol=3)
#' # y _j = rows of the matrix Y
#' Y <- matrix(runif(ny*3), nrow=ny, ncol=3)
#' # beta_j = rows of the matrix beta
#' beta <- matrix(runif(ny*3), nrow=ny, ncol=3)
#' # !! important !! y and beta should have the same dimension
#' 
#' # parameter
#' lambda <- 0.25
#' 
#' # compute the result
#' res <- op(list(X, Y, beta, lambda))
#' }
#' @export
keops_kernel <- function(formula, args, sum_scheme = "auto") {

    # check input
    assert_string(formula)
    assert_character(args, min.len = 1)
    assert_choice(
        sum_scheme, c("auto", "direct_sum", "block_sum", "kahan_scheme"))
    
    # args parsing
    args_info <- parse_args(formula, args)
    
    # return function calling the corresponding compile operator
    # inner_dim should be "row" or "col"
    function(input=NULL, inner_dim="col") {
        
        ## storing some context
        op_env <- list(
            formula = formula, args = args, 
            args_info = args_info,
            sum_scheme = sum_scheme
        )
        
        # return context if no input
        if(missing(input) | is.null(input))
            return(op_env)
        
        ## !! important !!
        # input: should be a list, because list are references and since 
        #   argument passing is done by copy in R, it is better to copy a list 
        #   of reference than the actual input data, especially for big 
        #   matrices. If NULL or missing, op_env (context) is returned
        # inner_dim: "col" for columns, "row" for rows
        
        ## Use pykeops.numpy.Genred
        # load GenredR
        GenredR <- NULL
        reticulate::source_python(
            system.file(file.path("python", "generic_red_R.py"), 
                        package = "rkeops"))
        # Note: not possible to use <Reduction>_Reduction(..., axis)
        # syntax with PyKeOps --> need to format formula from RKeOps to PyKeOps
        pykeops_formula <- get_pykeops_formula(formula)
        aliases <- NULL
        if(length(args) > 1) {
            aliases <- args
        } else {
            aliases <- list(args)
        }
        # manage specific case for weighted reduction
        routine <- NULL
        if(pykeops_formula$weighted_reduction) {
            routine <- GenredR(
                formula = pykeops_formula$main_formula,
                aliases = aliases,
                reduction_op = pykeops_formula$reduction_op,
                axis = pykeops_formula$axis,
                formula2 = pykeops_formula$opt_arg,
                sum_scheme = sum_scheme
            )
        } else {
            routine <- GenredR(
                formula = pykeops_formula$main_formula,
                aliases = aliases,
                reduction_op = pykeops_formula$reduction_op,
                axis = pykeops_formula$axis,
                opt_arg = pykeops_formula$opt_arg,
                sum_scheme = sum_scheme
            )
        }
        
        # check input type
        if(!is.list(input)) {
            msg <- str_c(
                "The 'input' argument should be a list of data ",
                "corresponding to the formula input arguments."
            )
            stop(msg)
        }
        
        # check input length
        if(length(input) != length(op_env$args)) {
            msg <- str_c(
                "The number of elements in the 'input' argument ",
                "does not correspond to the number of input arguments ",
                "in the formula."
            )
            stop(msg)
        }
        
        # reorder input (if named list and named arg in the formula)
        if(!is.null(names(input)) && length(names(input) == length(input))) {
            input <- input[op_env$args_info$var_name]
        }
        
        # remove input name
        input <- unname(input)
        
        # cast all input to matrix type
        input <- lapply(input, as.matrix)
        
        # check inner_dim
        assert_choice(inner_dim, c("row", "col"))
        
        # precision
        dtype <- get_rkeops_options("precision")

        # computing backend
        backend <-  get_rkeops_options("backend")
        device_id <- get_rkeops_options("device_id")

        # inner dimension
        # PyKeOps: "C" = row-major, "F" = column-major
        # always use "C"-order and transpose R input data if inner_dim = "row"
        data_order <- "C"
        if(inner_dim == "row") input <- lapply(input, t)
        
        # change R arrays into numpy arrays
        input <- lapply(
            input, function(x) np_array(x, dtype=dtype, order=data_order)
        )
        res <- routine(input, backend = backend, device_id = device_id)
        return(res)
    }
}
