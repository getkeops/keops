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
               list(inner_dim=0, nx=nx, ny=ny))
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
    
    ## computation on GPU ?
    if(Sys.getenv("TEST_GPU") == "1") use_gpu()
    
    ## bad formula
    formula = "Sum_Reduction((x|y, 1)"
    args = c("x=Vi(3)", "y=Vj(3)")
    # define and compile operator
    expect_error(keops_kernel(formula, args))
    
    ## matrix product then sum
    formula = "Sum_Reduction((x|y), 1)"
    args = c("x=Vi(3)", "y=Vj(3)")
    # define and compile operator
    op <- keops_kernel(formula, args)
    expect_true(is.function(op))
    
    # data
    # (standard: reduction index over rows, inner dimension over columns)
    nx <- 10
    ny <- 15
    x <- matrix(runif(nx*3), nrow=nx, ncol=3)
    y <- matrix(runif(ny*3), nrow=ny, ncol=3)
    # run nok (bad operator input)
    expect_error(op(x, y))
    input <- c(x, y)
    expect_error(op(input))
    # run ok
    input <- list(x, y)
    res <- op(input, inner_dim = 1)
    expect_true(is.matrix(res))
    expect_equal(dim(res), c(ny, 1))
    expected_res <- colSums(x %*% t(y))
    expect_true(sum(abs(res - expected_res)) < 1E-4)
    
    # run (named input wrong order)
    input <- list(y=y, x=x)
    res <- op(input, inner_dim = 1)
    expect_true(is.matrix(res))
    expect_equal(dim(res), c(ny, 1))
    expect_true(sum(abs(res - expected_res)) < 1E-4)
    
    # data
    # (reduction index over columns, inner dimension over rows)
    nx <- 10
    ny <- 15
    x <- matrix(runif(nx*3), nrow=3, ncol=nx)
    y <- matrix(runif(ny*3), nrow=3, ncol=ny)
    # run
    input <- list(x, y)
    res <- op(input, inner_dim = 0)
    expect_true(is.matrix(res))
    expect_equal(dim(res), c(1, ny))
    expected_res <- colSums(t(x) %*% y)
    expect_true(sum(abs(res - expected_res)) < 1E-4)
    
    
    ## Squared norm
    formula = "Sum_Reduction(SqNorm2(x-y), 1)"
    args = c("x=Vi(0,3)", "y=Vj(1,3)")
    # define and compile operator
    op <- keops_kernel(formula, args)
    expect_true(is.function(op))
    
    # data
    nx <- 10
    ny <- 15
    x <- matrix(runif(nx*3), nrow=nx, ncol=3)
    y <- matrix(runif(ny*3), nrow=ny, ncol=3)
    
    # expected res
    inter_res <- matrix(0, nrow = nx, ncol = ny)
    for(i in 1:nx) {
        for(j in 1:ny) {
            inter_res[i,j] = sum((x[i,] - y[j,])^2)
        }
    }
    expected_res <- apply(inter_res, 2, sum)
    
    # run
    input <- list(x, y)
    res <- op(input, inner_dim = 1)
    expect_true(is.matrix(res))
    expect_equal(dim(res), c(ny, 1))
    expect_true(sum(abs(res - expected_res)) < 1E-4)
    
    
    ## Vector of parameter
    formula = "Sum_Reduction(x+y, 1)"
    args = c("x=Vi(3)", "y=Pm(3)")
    # define and compile operator
    op <- keops_kernel(formula, args)
    expect_true(is.function(op))
    
    # data
    nx <- 10
    x <- matrix(runif(nx*3), nrow=nx, ncol=3)
    y <- 1:3
    # run
    input <- list(x, y)
    res <- op(input, inner_dim = 1)
    expect_true(is.matrix(res))
    expect_equal(dim(res), c(1, 3))
    expected_res <- apply(x + matrix(rep(y, nx), byrow = TRUE, ncol = 3), 2, sum)
    expect_true(sum(abs(res - expected_res)) < 1E-4)
})

test_that("keops_grad", {
    set_rkeops_options()
    
    ## computation on GPU ?
    if(Sys.getenv("TEST_GPU") == "1") use_gpu()
    
    ## define an operator (squared norm reduction)
    formula <- "Sum_Reduction(SqNorm2(x-y), 0)"
    args <- c("x=Vi(0,3)", "y=Vj(1,3)")
    op <- keops_kernel(formula, args)
    
    ## define and compile the gradient regarding var 0
    grad_op <- keops_grad(op, var=0)
    expect_true(is.function(grad_op))
    
    # # data
    # (standard: reduction index over rows, inner dimension over columns)
    nx <- 10
    ny <- 15
    x <- matrix(runif(nx*3), nrow=nx, ncol=3)
    y <- matrix(runif(ny*3), nrow=ny, ncol=3)
    eta <- matrix(1, nrow=nx, ncol=1)
    # run
    input <- list(x, y, eta)
    expected_res <- t(sapply(1:nx, function(i) {
        tmp <- sapply(1:ny, function(j) {
            return(2 * (x[i,]-y[j,]))
        })
        return(apply(tmp,1,sum))
    }))
    res <- grad_op(input)
    expect_true(is.matrix(res))
    expect_equal(dim(res), c(nx, 3))
    expect_true(sum(abs(res - expected_res)) < 1E-4)
    
    # data
    # (reduction index over columns, inner dimension over rows)
    nx <- 10
    ny <- 15
    x <- matrix(runif(nx*3), nrow=3, ncol=nx)
    y <- matrix(runif(ny*3), nrow=3, ncol=ny)
    eta <- matrix(1, nrow=1, ncol=nx)
    # run
    input <- list(x, y, eta)
    expected_res <- expected_res <- sapply(1:nx, function(i) {
        tmp <- sapply(1:ny, function(j) {
            return(2 * (x[,i]-y[,j]))
        })
        return(apply(tmp,1,sum))
    })
    res <- grad_op(input, inner_dim = 0)
    expect_true(is.matrix(res))
    expect_equal(dim(res), c(3, nx))
    expect_true(sum(abs(res - expected_res)) < 1E-4)
    
    
    ## define and compile the gradient regarding var x
    grad_op <- keops_grad(op, var="x")
    expect_true(is.function(grad_op))
    
    # # data
    # (standard: reduction index over rows, inner dimension over columns)
    nx <- 10
    ny <- 15
    x <- matrix(runif(nx*3), nrow=nx, ncol=3)
    y <- matrix(runif(ny*3), nrow=ny, ncol=3)
    eta <- matrix(1, nrow=nx, ncol=1)
    # run
    input <- list(x, y, eta)
    expected_res <- t(sapply(1:nx, function(i) {
        tmp <- sapply(1:ny, function(j) {
            return(2 * (x[i,]-y[j,]))
        })
        return(apply(tmp,1,sum))
    }))
    res <- grad_op(input)
    expect_true(is.matrix(res))
    expect_equal(dim(res), c(nx, 3))
    expect_true(sum(abs(res - expected_res)) < 1E-4)
    
    # data
    # (reduction index over columns, inner dimension over rows)
    nx <- 10
    ny <- 15
    x <- matrix(runif(nx*3), nrow=3, ncol=nx)
    y <- matrix(runif(ny*3), nrow=3, ncol=ny)
    eta <- matrix(1, nrow=1, ncol=nx)
    # run
    input <- list(x, y, eta)
    expected_res <- expected_res <- sapply(1:nx, function(i) {
        tmp <- sapply(1:ny, function(j) {
            return(2 * (x[,i]-y[,j]))
        })
        return(apply(tmp,1,sum))
    })
    res <- grad_op(input, inner_dim = 0)
    expect_true(is.matrix(res))
    expect_equal(dim(res), c(3, nx))
    expect_true(sum(abs(res - expected_res)) < 1E-4)
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
