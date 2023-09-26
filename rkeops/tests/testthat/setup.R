# requirements
library(checkmate)
library(withr)

# helper function to skip tests if Python is not available on the system
skip_if_no_python <- function() {
    have_python <- reticulate::py_available()
    if(!have_python) skip("Python not available on system for testing")
}

# helper function to skip tests if `keopscore` is not available
skip_if_no_keopscore <- function() {
    skip_if_no_python()
    have_keopscore <- rkeops:::check_keopscore(warn=FALSE)
    if(!have_keopscore) skip("'keopscore' not available for testing")
}

# helper function to skip tests if `pykeops` is not available
skip_if_no_pykeops <- function() {
    skip_if_no_python()
    have_pykeops <- rkeops:::check_pykeops(warn=FALSE)
    if(!have_pykeops) skip("'pykeops' not available for testing")
}

# helper function to skip tests if not interactive mode
skip_if_not_interactive <- function() {
    if(!interactive()) skip("Test only run in interactive mode")
}

# function to repeat check of formula arg
# input expected: args = c("x=Vi(3)", "y=Vj(3)", "beta=Vj(3)", "lambda=Pm(1)")
check_parse_args <- function(formula, args, arg_order = NULL, decl = "dim") {
    
    if(missing(arg_order)) arg_order <- seq(1, length(args), 1)
    
    out <- parse_args(formula, args)
    checkmate::expect_list(out, len = 6)
    expect_equal(out$var_name, c("x", "y", "beta", "lambda")[arg_order])
    expect_equal(out$var_type, c("Vi", "Vj", "Vj", "Pm")[arg_order])
    expect_equal(out$var_pos, c(0, 1, 2, 3)[arg_order])
    checkmate::expect_choice(out$decl, decl)
}
