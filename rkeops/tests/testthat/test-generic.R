context("generic functions")

test_that("compile_formula", {
    set_rkeops_options()
    # matrix product then sum
    formula = "Sum_Reduction((x|y), 1)"
    args = c("x=Vi(3)", "y=Vj(3)")
    var_aliases <- format_var_aliases(args)$var_aliases
    dllname <- "test_compile_formula_dll"
    ## run
    res <- tryCatch(compile_formula(formula, var_aliases, dllname),
                    error = function(e) {print(e); return(NULL)})
    ## check
    expect_false(is.null(res))
    ## testing formula
    # load shared library
    r_genred <- load_dll(path = get_build_dir(),
                         dllname = paste0("librkeops", dllname), 
                         object = "r_genred",
                         genred=TRUE)
    ## data (reduction index in column)
    nx <- 10
    ny <- 15
    x <- matrix(runif(nx*3), nrow=3, ncol=nx)
    y <- matrix(runif(ny*3), nrow=3, ncol=ny)
    # run
    param <- c(get_rkeops_options("runtime"),
               list(inner_dim=0))
    input <- list(x, y)
    res <- r_genred(input, param)
    # check result
    expected_res <- colSums(t(x) %*% y)
    expect_true(sum(abs(res - expected_res)) < 1E-5)
})

test_that("format_var_aliases", {
    # short syntax
    args <- c("x=Vi(3)", "y=Vj(3)", 
             "beta=Vj(3)", "lambda=Pm(1)")
    check_args(args)
    # long syntax
    args <- c("x=Vi(0,3)", "y=Vj(1,3)", 
             "beta=Vj(2,3)", "lambda=Pm(3,1)")
    check_args(args)
    # long syntax unordered
    arg_order <- c(1,3,4,2)
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

test_that("keops_kernel", {
    set_rkeops_options()
    # matrix product then sum
    formula = "Sum_Reduction((x|y), 1)"
    args = c("x=Vi(3)", "y=Vj(3)")
    # define and compile operator
    op <- tryCatch(keops_kernel(formula, args),
                   error = function(e) {print(e); return(NULL)})
    expect_false(is.null(op))
    
    ## data (reduction index in column)
    nx <- 10
    ny <- 15
    x <- matrix(runif(nx*3), nrow=3, ncol=nx)
    y <- matrix(runif(ny*3), nrow=3, ncol=ny)
    # run
    input <- list(x, y)
    expected_res <- colSums(t(x) %*% y)
    run_op(op, input, expected_res, inner_dim=0)
    
    ## data (reduction index in row)
    nx <- 10
    ny <- 15
    x <- t(matrix(runif(nx*3), nrow=3, ncol=nx))
    y <- t(matrix(runif(ny*3), nrow=3, ncol=ny))
    # run
    input <- list(x, y)
    expected_res <- colSums(x %*% t(y))
    run_op(op, input, expected_res, inner_dim=1)
    
    ## data (named input wrong order)
    nx <- 10
    ny <- 15
    x <- matrix(runif(nx*3), nrow=3, ncol=nx)
    y <- matrix(runif(ny*3), nrow=3, ncol=ny)
    # run
    input <- list(y=y, x=x)
    expected_res <- colSums(t(x) %*% y)
    run_op(op, input, expected_res, inner_dim=0)
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
    expect_equal(out$var_type, NULL)
    expect_equal(out$var_pos, NULL)
    expect_equal(out$var_dim, NULL)
    
})
