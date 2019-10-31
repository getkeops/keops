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
#' @param var a text string or an integer indicating regarding to which 
#' variable/parameter (given by name or by position starting at 1) the 
#' gradient of the formula should be computed.
#' @return FIXME
#' @examples
#' \dontrun{
#' }
#' @export
keops_grad <- function(operator, var) {
    # get operator context (formula, variable, etc.)
    env <- operator()
    # position new variable = length arg list - 1 (last)
    posnewvar <- length(env$var_aliases$args)
    # position of the variable to be derived
    pos <- NULL
    if(is.character(var)) {
        pos <- env$var_aliases$var_pos[which(env$var_aliases$var_pos == var)]
    } else if(is.numeric(var)) {
        pos <- var
    } else {
        stop("`var` input argument should be a text string or an integer.")
    }
}