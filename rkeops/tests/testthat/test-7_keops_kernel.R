test_that("keops_kernel", {
    skip_if_no_python()
    skip_if_no_keopscore()
    skip_if_no_pykeops()
    
    set_rkeops_options()
    
    ## computation on GPU ?
    if(Sys.getenv("TEST_GPU") == "1") rkeops_use_gpu()
    
    ## use float64 precision in test (to match R precision)
    rkeops_use_float64()
    
    ## bad formula
    formula = "Sum_Reduction((x|y, 1)"
    args = c("x=Vi(3)", "y=Vj(3)")
    # define and compile operator
    op <- keops_kernel(formula, args)
    # define input
    nx <- 10
    ny <- 15
    x <- matrix(runif(nx*3), nrow=nx, ncol=3)
    y <- matrix(runif(ny*3), nrow=ny, ncol=3)
    # expect error because bad formula
    expect_error(op(list(x, y)))
    
    ## matrix product then sum
    formula = "Sum_Reduction((x|y), 0)"
    args = c("x=Vi(3)", "y=Vj(3)")
    # define and compile operator
    op <- keops_kernel(formula, args)
    expect_true(is.function(op))
    
    # operator information
    checkmate::expect_list(op(), len = 4)
    expect_equal(
        names(op()), 
        c("formula", "args", "args_info", "sum_scheme")
    )
    
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
    res1 <- op(input, inner_dim = "col")
    expect_true(is.matrix(res1))
    expect_equal(dim(res1), c(ny, 1))
    expected_res1 <- colSums(x %*% t(y))
    expect_true(sum(abs(res1 - expected_res1)) < 1E-9)
    
    # run (named input, good order)
    input <- list(x=x, y=y)
    res <- op(input, inner_dim = "col")
    expect_true(is.matrix(res))
    expect_equal(dim(res), c(ny, 1))
    expected_res <- colSums(x %*% t(y))
    expect_true(sum(abs(res - expected_res)) < 1E-9)
    
    # run (named input, wrong order)
    input <- list(y=y, x=x)
    res <- op(input, inner_dim = "col")
    expect_true(is.matrix(res))
    expect_equal(dim(res), c(ny, 1))
    expected_res <- colSums(x %*% t(y))
    expect_true(sum(abs(res - expected_res)) < 1E-9)
    
    # data
    # (reduction index over columns, inner dimension over rows)
    # nx <- 10
    # ny <- 15
    # x <- matrix(runif(nx*3), nrow=3, ncol=nx)
    # y <- matrix(runif(ny*3), nrow=3, ncol=ny)
    x <- t(x)
    y <- t(y)
    # run
    input <- list(x, y)
    res2 <- op(input, inner_dim = "row")
    expect_true(is.matrix(res2))
    expect_equal(dim(res2), c(ny, 1))
    expected_res2 <- colSums(t(x) %*% y)
    expect_true(sum(abs(res2 - expected_res2)) < 1E-9)
    expect_true(sum(abs(res1 - res2)) < 1E-9)
    
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
    expected_res <- apply(inter_res, 1, sum)
    
    # run
    input <- list(x, y)
    res <- op(input, inner_dim = "col")
    expect_true(is.matrix(res))
    expect_equal(dim(res), c(nx, 1))
    expect_true(sum(abs(res - expected_res)) < 1E-9)
    
    
    ## Vector of parameter
    formula = "Sum_Reduction(x+y, 0)"
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
    res <- op(input, inner_dim = "col")
    expect_true(is.matrix(res))
    expect_equal(dim(res), c(1, 3))
    expected_res <- apply(
        x + matrix(rep(y, nx), byrow = TRUE, ncol = 3), 2, sum)
    expect_true(sum(abs(res - expected_res)) < 1E-9)
    
    
    ## weighted reduction (e.g. SumSoftMaxWeight)
    formula = "SumSoftMaxWeight_Reduction(SqNorm2(x-y), w, 1)"
    args = c("x=Vi(3)", "y=Vj(3)", "w=Vj(3)")
    # define and compile operator
    op <- keops_kernel(formula, args)
    expect_true(is.function(op))
    
    # data
    nx <- 10
    ny <- 15
    x <- matrix(runif(nx*3), nrow=nx, ncol=3)
    y <- matrix(runif(ny*3), nrow=ny, ncol=3)
    
    # with weights equal to 1
    w <- matrix(1, nrow = ny, ncol = 3)
    
    # run
    input <- list(x, y, w)
    res <- op(input, inner_dim = "col")
    expect_true(is.matrix(res))
    expect_equal(dim(res), c(nx, 3))
    expect_equal(res, matrix(1, nx, 3))
    
    # with random weights
    w <- matrix(1, nrow = ny, ncol = 3)
    input <- list(x, y, w)
    res <- op(input, inner_dim = "col")
    expect_true(is.matrix(res))
    expect_equal(dim(res), c(nx, 3))
    expect_equal(res, matrix(1, nx, 3))
    
    S_val <- t(sapply(
        1:nrow(x), function(id_x) 
            sapply(1:nrow(y), function(id_y) sum((x[id_x,] - y[id_y,])^2))
    ))
    exp_S_val <- t(apply(
        S_val, 1, 
        function(vec) {
            return(exp(vec - max(vec)))
        }
    ))
    expected_res <- exp_S_val %*% w / matrix(apply(exp_S_val, 1, sum), nx, 3)
    expect_equal(res, expected_res, tolerance = 1e-5)
    
})
