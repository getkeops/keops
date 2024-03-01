test_that("get_gradient_formula", {
    
    ## no formula?
    formula <- ""
    args <- c("x=Vi(0,3)", "y=Vj(1,3)")
    var <- "x"
    args_info <- parse_args(formula, args)
    expect_error(get_gradient_formula(formula, args, var, args_info), NA)
    
    ## no args?
    formula <- "Sum_Reduction(SqNorm2(x-y), 0)"
    args <- character(0)
    var <- "x"
    args_info <- parse_args(formula, args)
    expect_error(get_gradient_formula(formula, args, var, args_info))
    
    ## no arg info?
    formula <- "Sum_Reduction(SqNorm2(x-y), 0)"
    args <- c("x=Vi(0,3)", "y=Vj(1,3)")
    var <- "x"
    args_info <- list()
    expect_error(get_gradient_formula(formula, args, var, args_info))
    
    
    ## ok
    formula <- "Sum_Reduction(SqNorm2(x-y), 0)"
    args <- c("x=Vi(0,3)", "y=Vj(1,3)")
    var <- "x"
    args_info <- parse_args(formula, args)
    res <- get_gradient_formula(formula, args, var, args_info)
    checkmate::expect_list(res)
    expect_equal(names(res), c("new_formula", "new_args"))
    checkmate::expect_string(
        res$new_formula, 
        pattern = stringr::str_c(
            "^Grad\\(", stringr::str_escape(formula), ",", var, ",", 
            "var[0-9a-zA-Z]{5}\\)$")
    )
    expect_equal(length(res$new_args), length(args) + 1)
    checkmate::expect_string(
        tail(res$new_args, 1), 
        pattern = stringr::str_c(
            "^var[0-9a-zA-Z]{5}", stringr::str_escape("=Vi(2,1)"), "$")
    )
})

test_that("keops_grad", {
    skip_if_no_python()
    skip_if_no_keopscore()
    skip_if_no_pykeops()
    
    set_rkeops_options()
    
    ## computation on GPU ?
    if(Sys.getenv("TEST_GPU") == "1") use_gpu()
    
    ## use float64 precision in test (to match R precision)
    rkeops_use_float64()
    
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
    expected_res <- t(sapply(1:ny, function(j) {
        tmp <- sapply(1:nx, function(i) {
            return(2 * (x[i,]-y[j,]))
        })
        return(apply(tmp,1,sum))
    }))
    res <- grad_op(input)
    expect_true(is.matrix(res))
    expect_equal(dim(res), c(ny, 3))
    expect_true(sum(abs(res - expected_res)) < 1E-9)
    
    # data
    # (reduction index over columns, inner dimension over rows)
    nx <- 10
    ny <- 15
    x <- matrix(runif(nx*3), nrow=3, ncol=nx)
    y <- matrix(runif(ny*3), nrow=3, ncol=ny)
    eta <- matrix(1, nrow=1, ncol=nx)
    # run
    input <- list(x, y, eta)
    expected_res <- sapply(1:ny, function(j) {
        tmp <- sapply(1:nx, function(i) {
            return(2 * (x[,i]-y[,j]))
        })
        return(apply(tmp,1,sum))
    })
    res <- grad_op(input, inner_dim = "row")
    expect_true(is.matrix(res))
    expect_equal(dim(res), c(ny, 3))
    expect_true(sum(abs(res - t(expected_res))) < 1E-9)
    
    
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
    expected_res <- t(sapply(1:ny, function(j) {
        tmp <- sapply(1:nx, function(i) {
            return(2 * (x[i,]-y[j,]))
        })
        return(apply(tmp,1,sum))
    }))
    res <- grad_op(input)
    expect_true(is.matrix(res))
    expect_equal(dim(res), c(ny, 3))
    expect_true(sum(abs(res - expected_res)) < 1E-9)
    
    # data
    # (reduction index over columns, inner dimension over rows)
    nx <- 10
    ny <- 15
    x <- matrix(runif(nx*3), nrow=3, ncol=nx)
    y <- matrix(runif(ny*3), nrow=3, ncol=ny)
    eta <- matrix(1, nrow=1, ncol=nx)
    # run
    input <- list(x, y, eta)
    expected_res <- sapply(1:ny, function(j) {
        tmp <- sapply(1:nx, function(i) {
            return(2 * (x[,i]-y[,j]))
        })
        return(apply(tmp,1,sum))
    })
    res <- grad_op(input, inner_dim = "row")
    expect_true(is.matrix(res))
    expect_equal(dim(res), c(ny, 3))
    expect_true(sum(abs(res - t(expected_res))) < 1E-9)
    
    ## direct gradient
    formula <- "Grad(Sum_Reduction(SqNorm2(x-y), 0),x,eta)"
    args <- c("x=Vi(0,3)", "y=Vj(1,3)", "eta=Vi(2,1)")
    op2 <- keops_kernel(formula, args)
    
    # # data
    # (standard: reduction index over rows, inner dimension over columns)
    nx <- 10
    ny <- 15
    x <- matrix(runif(nx*3), nrow=nx, ncol=3)
    y <- matrix(runif(ny*3), nrow=ny, ncol=3)
    eta <- matrix(1, nrow=nx, ncol=1)
    # run
    input <- list(x, y, eta)
    expected_res <- t(sapply(1:ny, function(j) {
        tmp <- sapply(1:nx, function(i) {
            return(2 * (x[i,]-y[j,]))
        })
        return(apply(tmp,1,sum))
    }))
    res1 <- grad_op(input)
    res2 <- op2(input)
    
    expect_true(is.matrix(res2))
    expect_equal(dim(res2), c(ny, 3))
    expect_true(sum(abs(res1 - res2)) < 1E-9)
    expect_true(sum(abs(res2 - expected_res)) < 1E-9)

})
