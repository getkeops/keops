#' Compute the gradient of a rkeops operator
#' @description
#' FIXME
#' @details
#' The use of the function `keops_grad` is detailled in the vignettes. 
#' Run `browseVignettes("rkeops")` to access the vignettes.
#' FIXME
#' @author Ghislain Durif
#' @param operator a function returned by `keops_kernel` implementing a 
#' formula.
#' @param var a text string or an integer number indicating regarding to which 
#' variable/parameter (given by name or by position starting at 0) the 
#' gradient of the formula should be computed.
#' @return FIXME
#' @importFrom stringr str_match_all
#' @examples
#' \dontrun{
#' formula <- "Sum_Reduction(SqNorm2(x-y), 0)"
#' args <- c("x=Vi(0,3)", "y=Vj(1,3)")
#' op <- keops_kernel(formula, args)
#' grad_op <- keops_grad(op, var=0)
#' }
#' @export
keops_grad <- function(operator, var) {
    stop("gradient computation is not ready yet, you can compute gradient by directly the formula with `Grad` keyword. See the vignettes.")
    # check input (string or integer)
    if(is.numeric(var)) var <- as.integer(var)
    if(length(var) > 1 & !is.character(var) & !is.integer(var)) 
        stop(paste0("`var` input argument should be the name (string) ", 
                    "or position (integer) of an argument in the formula"))
    # get operator context (formula, variable, etc.)
    env <- operator()
    # parse formula and args to derive position of new variable
    var_aliases <- env$var_aliases
    extra_var <- parse_extra_args(env$formula, env$args)
    # position new variable = length arg list - 1 (last)
    posnewvar <- max(c(var_aliases$var_pos, extra_var$var_pos)) + 1
    # position of the variable to be derived
    pos <- NULL
    if(is.character(var)) {
        pos <- env$var_aliases$var_pos[which(env$var_aliases$var_pos == var)]
    } else if(is.numeric(var)) {
        pos <- var
    } else {
        stop("`var` input argument should be a text string or an integer.")
    }
    # new formula
    new_formula <- paste0("GradFromPos(", env$formula, ",", pos, ",", 
                          posnewvar, ")")
    # define new op
    return(keops_kernel(new_formula, env$args))
}