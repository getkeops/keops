#' Defines a new operators
#' @description
#' FIXME
#' @details
#' FIXME
#' See <http://www.kernel-operations.io/keops/api/math-operations.html>
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
    var_aliases <- format_var_aliases(args)
    
    # hash name to compile formula in a shared library file
    dllname <- create_dllname(formula, args)
    dllfilename <- file.path(get_build_dir(), 
                             paste0("librkeops", dllname, .Platform$dynlib.ext))
    
    # compile operator if necessary
    if(!file.exists(dllfilename) | get_rkeops_option("verbosity")) {
        compile_formula(formula, var_aliases$var_aliases, dllname) # FIXME
    }
    
    # load shared library
    r_genred <- load_dll(path = get_build_dir(),
                         dllname = paste0("librkeops", dllname), 
                         object = "r_genred",
                         genred=TRUE)
    
    # return function calling the corresponding compile operator
    function(args, param) {
        return(r_genred(args, param))
    }
}