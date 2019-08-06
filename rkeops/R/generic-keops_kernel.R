#' Defines a new operators
#' @description
#' FIXME
#' @details
#' FIXME
#' @author Ghislain Durif
#' @param formula text string, an operator formula.
#' @param args vector of text string, operator parameters.
#' @return
#' FIXME
#' @examples
#' \dontrun{
#' # Define and test a function that computes for each i the sum over j
#' # of the square of the scalar products of `x_i` and `y_j` (both 3d vectors)
#' F <- keops_kernel(formula = "Sum_Reduction(Square((x,y)),0)",
#'                   args = c("x=Vi(3)", "y=Vj(3)"))
#' x <- rnorm(3,2000)
#' y <- rnorm(3,5000)
#' res <- F(x,y)
#' 
#' # Define and test the convolution with a Gauss kernel i.e. the sum 
#' # over j of `e^(lambda*||x_i - y_j||^2) * beta_j` where `x_i`, `y_j` 
#' # and `beta_j` are 3d vectors
#' F = keops_kernel(formula = "Sum_Reduction(Exp(lambda*SqNorm2(x-y))*beta,0)",
#'                  args = c("x=Vi(3)", "y=Vj(3)", 
#'                           "beta=Vj(3)", "lambda=Pm(1)"))
#' x <- rnorm(3,2000)
#' y <- rnorm(3,5000)
#' beta <- rnorm(3,5000)
#' lambda <- 0.25
#' res <- F(x, y, beta, lambda)
#' }
#' @export
keops_kernel <- function(formula, args) {
    # check input
    if(!is.character(formula))
        stop("`formula` input parameter should be a text string")
    if(!is.vector(args) & !is.character(args))
        stop("`args` input parameter should be a vector of text strings")
    
    # check formula and args formating
    # FIXME
    
    # hash name to compile formula in a shared library file
    dllname <- create_dllname(formula, args)
    dllfilename <- file.path(get_build_dir(), dllname)
    
    # operator name
    operator_name <- paste0("keops", dllname) # FIXME
    
    # compile operator if necessary
    if(!file.exists(dllfilename) | get_rkeops_option("verbosity")) {
        compile_formula() # FIXME
    }
    
    # load shared library
    # FIXME
    # dyn.load(dllfilename, local = FALSE)
    
    # return function calling the corresponding compile operator
    # FIXME
    # function(args) {
    #     return(.Call(operator_name, args))
    # }
}