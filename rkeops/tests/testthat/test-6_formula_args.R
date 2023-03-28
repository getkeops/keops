test_that("format_var_aliases", {
    # short syntax
    args <- c("x=Vi(3)", "y=Vj(3)", 
              "beta=Vj(3)", "lambda=Pm(1)")
    check_format_var_aliases(args)
    # long syntax
    args <- c("x=Vi(0,3)", "y=Vj(1,3)", 
              "beta=Vj(2,3)", "lambda=Pm(3,1)")
    check_format_var_aliases(args)
    # long syntax unordered
    arg_order <- c(1,3,4,2)
    args <- c("x=Vi(0,3)", "y=Vj(1,3)", 
              "beta=Vj(2,3)", "lambda=Pm(3,1)")[arg_order]
    check_format_var_aliases(args, arg_order)
    # errors
    expect_error(format_var_aliases(5))
    expect_error(format_var_aliases(c(3,5)))
    expect_error(
        format_var_aliases("test"), 
        "Issue with input value(s): 'test'",
        fixed = TRUE)
    expect_error(
        format_var_aliases(c("test1", "test2")), 
        "Issue with input value(s): 'test1', 'test2'", 
        fixed = TRUE)
    expect_error(
        format_var_aliases(c("x=Vi(3)", "Vj(3)")), 
        "Issue with input value(s): 'Vj(3)'", 
        fixed = TRUE)
    expect_error(
        format_var_aliases(c("x=Vi(3)", "y=Vj(3")), 
        "Issue with input value(s): 'y=Vj(3'", 
        fixed = TRUE)
    expect_error(
        format_var_aliases(c("x=Vi(0,3)", "y=Vj(1,3")), 
        "Issue with input value(s): 'y=Vj(1,3'", 
        fixed = TRUE)
    expect_error(
        format_var_aliases(c("x=Vi(3)", "y=Vj(1,3)")), 
        paste0("Issue with input argument consistency, use either ", 
               "'(dim)' or '(pos, dim)' for 'Vi', 'Vj' and 'Pm' ", 
               "(see help page or vignette for more details)."), 
        fixed = TRUE)
})

test_that("parse_extra_args", {
    
    formula <- "Sum_Reduction(Exp(lambda * SqNorm2(x - y)) * beta, 0)"
    args <- c("x=Vi(3)", "y=Vj(3)", "beta=Vj(6)", "lambda=Pm(1)")
    out <- parse_extra_args(formula, args)
    expect_equal(out$var_type, NULL)
    expect_equal(out$var_pos, NULL)
    expect_equal(out$var_dim, NULL)
    
    formula <- "Sum_Reduction(Exp(lambda * SqNorm2(x - y)) * Vj(6), 0)"
    args <- c("x=Vi(3)", "y=Vj(3)", "lambda=Pm(1)")
    out <- parse_extra_args(formula, args)
    expect_equal(out$var_type, "Vj")
    expect_equal(out$var_pos, 3)
    expect_equal(out$var_dim, 6)
    
    formula <- "Sum_Reduction(Exp(Pm(1) * SqNorm2(x - y)) * Vj(6), 0)"
    args <- c("x=Vi(3)", "y=Vj(3)")
    out <- parse_extra_args(formula, args)
    expect_equal(out$var_type, c("Pm", "Vj"))
    expect_equal(out$var_pos, c(2, 3))
    expect_equal(out$var_dim, c(1, 6))
    
    
    formula <- "Sum_Reduction(Exp(lambda * SqNorm2(x - y)) * Vj(2,6), 0)"
    args <- c("x=Vi(0,3)", "y=Vj(1,3)", "lambda=Pm(3,1)")
    out <- parse_extra_args(formula, args)
    expect_equal(out$var_type, "Vj")
    expect_equal(out$var_pos, 2)
    expect_equal(out$var_dim, 6)
    
    formula <- "Sum_Reduction(Exp(Pm(2,1) * SqNorm2(x - y)) * Vj(3,6), 0)"
    args <- c("x=Vi(0,3)", "y=Vj(1,3)")
    out <- parse_extra_args(formula, args)
    expect_equal(out$var_type, c("Pm", "Vj"))
    expect_equal(out$var_pos, c(2, 3))
    expect_equal(out$var_dim, c(1, 6))
    
})
