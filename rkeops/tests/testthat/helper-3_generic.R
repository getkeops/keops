# function to repeat check of formula arg
# input expected: args = c("x=Vi(3)", "y=Vj(3)", "beta=Vj(3)", "lambda=Pm(1)")
check_args <- function(args, arg_order = NULL) {
    
    if(missing(arg_order)) arg_order <- seq(1, length(args), 1)
    
    out <- format_var_aliases(args)
    expect_is(out, "list")
    expect_equal(out$var_name, c("x", "y", "beta", "lambda")[arg_order])
    expect_equal(out$var_type, c("Vi", "Vj", "Vj", "Pm")[arg_order])
    expect_equal(out$var_pos, c(0, 1, 2, 3)[arg_order])
    expect_equal(out$var_aliases, 
                 paste0(c("decltype(Vi(0,3)) x;", 
                          "decltype(Vj(1,3)) y;", 
                          "decltype(Vj(2,3)) beta;", 
                          "decltype(Pm(3,1)) lambda;")[arg_order],
                        collapse = ""))
}
