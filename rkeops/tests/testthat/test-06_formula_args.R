test_that("parse_args", {
    formula <- "Sum_Reduction(Exp(lambda * SqNorm2(x - y)) * beta, 0)"
    # short syntax
    args <- c("x=Vi(3)", "y=Vj(3)", 
              "beta=Vj(3)", "lambda=Pm(1)")
    check_parse_args(formula, args, decl = "dim")
    # long syntax
    args <- c("x=Vi(0,3)", "y=Vj(1,3)", 
              "beta=Vj(2,3)", "lambda=Pm(3,1)")
    check_parse_args(formula, args, decl = "pos_dim")
    # long syntax unordered
    arg_order <- c(1,3,4,2)
    args <- c("x=Vi(0,3)", "y=Vj(1,3)", 
              "beta=Vj(2,3)", "lambda=Pm(3,1)")[arg_order]
    check_parse_args(formula, args, arg_order, decl = "pos_dim")
    # errors
    expect_error(parse_args("", 5))
    expect_error(parse_args(c(3,5)))
    expect_error(
        parse_args("", "test"), 
        "Issue with input value(s): 'test'",
        fixed = TRUE)
    expect_error(
        parse_args("", c("test1", "test2")), 
        "Issue with input value(s): 'test1', 'test2'", 
        fixed = TRUE)
    expect_error(
        parse_args("", c("x=Vi(3)", "Vj(3)")), 
        "Issue with input value(s): 'Vj(3)'", 
        fixed = TRUE)
    expect_error(
        parse_args("", c("x=Vi(3)", "y=Vj(3")), 
        "Issue with input value(s): 'y=Vj(3'", 
        fixed = TRUE)
    expect_error(
        parse_args("", c("x=Vi(0,3)", "y=Vj(1,3")), 
        "Issue with input value(s): 'y=Vj(1,3'", 
        fixed = TRUE)
    expect_error(
        parse_args("", c("x=Vi(3)", "y=Vj(1,3)")), 
        str_c(
            "Issue with input argument consistency, use either ", 
            "'(dim)' or '(pos, dim)' with 'Vi', 'Vj' and 'Pm' ",
            "for all arguments ",
            "(see help page or vignette for more details)."
        ), 
        fixed = TRUE)
    
    # extra args
    formula <- "Sum_Reduction(Exp(Pm(1) * SqNorm2(x - y)) * Vj(6), 0)"
    args <- c("x=Vi(3)", "y=Vj(3)")
    expect_error(parse_args(formula, args))
})

test_that("parse_extra_args", {
    
    formula <- "Sum_Reduction(Exp(lambda * SqNorm2(x - y)) * beta, 0)"
    args <- c("x=Vi(3)", "y=Vj(3)", "beta=Vj(6)", "lambda=Pm(1)")
    expect_false(parse_extra_args(formula))
    
    formula <- "Sum_Reduction(Exp(lambda * SqNorm2(x - y)) * Vj(6), 0)"
    args <- c("x=Vi(3)", "y=Vj(3)", "lambda=Pm(1)")
    expect_true(parse_extra_args(formula))
    
    formula <- "Sum_Reduction(Exp(Pm(1) * SqNorm2(x - y)) * Vj(6), 0)"
    args <- c("x=Vi(3)", "y=Vj(3)")
    expect_true(parse_extra_args(formula))
    
    
    formula <- "Sum_Reduction(Exp(lambda * SqNorm2(x - y)) * Vj(2,6), 0)"
    args <- c("x=Vi(0,3)", "y=Vj(1,3)", "lambda=Pm(3,1)")
    expect_true(parse_extra_args(formula))
    
    formula <- "Sum_Reduction(Exp(Pm(2,1) * SqNorm2(x - y)) * Vj(3,6), 0)"
    args <- c("x=Vi(0,3)", "y=Vj(1,3)")
    expect_true(parse_extra_args(formula))
    
})
