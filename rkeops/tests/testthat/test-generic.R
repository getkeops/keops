context("generic functions")

test_that("format_var_aliases",  {
    # short syntax
    args <- c("x=Vi(3)", "y=Vj(3)", 
             "beta=Vj(3)", "lambda=Pm(1)")
    check_args(args)
    # long syntax
    args <- c("x=Vi(0,3)", "y=Vj(1,3)", 
             "beta=Vj(2,3)", "lambda=Pm(3,1)")
    check_args(args)
    # long syntax unordered
    arg_order <- c(1,3,0,2)
    args <- c("x=Vi(0,3)", "y=Vj(1,3)", 
              "beta=Vj(2,3)", "lambda=Pm(3,1)")[arg_order]
    check_args(args, arg_order)
    # errors
    expect_error(format_var_aliases(5), 
                 "`args` input argument should be a vector of text strings.", 
                 fixed = TRUE)
    expect_error(format_var_aliases(c(3,5)), 
                 "`args` input argument should be a vector of text strings.", 
                 fixed = TRUE)
    expect_error(format_var_aliases("test"), 
                 "Issue with input value(s) 'test'", 
                 fixed = TRUE)
    expect_error(format_var_aliases(c("test1", "test2")), 
                 "Issue with input value(s) 'test1', 'test2'", 
                 fixed = TRUE)
    expect_error(format_var_aliases(c("x=Vi(3)", "Vj(3)")), 
                 "Issue with input value(s) 'Vj(3)'", 
                 fixed = TRUE)
    expect_error(format_var_aliases(c("x=Vi(3)", "y=Vj(3")), 
                 "Issue with input value(s) 'y=Vj(3'", 
                 fixed = TRUE)
    expect_error(format_var_aliases(c("x=Vi(0,3)", "y=Vj(1,3")), 
                 "Issue with input value(s) 'y=Vj(1,3'", 
                 fixed = TRUE)
    expect_error(format_var_aliases(c("x=Vi(3)", "y=Vj(1,3)")), 
                 paste0("Issue with input argument consistency, use either ", 
                        "'(dim)' or '(pos, dim)' for 'Vi', 'Vj' and 'Pm' ", 
                        "(see help page or vignette for more details)."), 
                 fixed = TRUE)
})
