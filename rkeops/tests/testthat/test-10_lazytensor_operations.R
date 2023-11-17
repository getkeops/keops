skip_if_no_python()
skip_if_no_keopscore()
skip_if_no_pykeops()

# TEST ARITHMETIC OPERATIONS ===================================================


test_that("+", {
    
    # check that base operation is still working
    expect_equal(2+3, 5)
    expect_equal(+1, 0+1)
    expect_equal(c(1,2) + c(1,2), c(2,4))
    expect_equal(matrix(1, 2, 2) + matrix(1, 2, 2), matrix(2, 2, 2))
    expect_equal(matrix(1, 2, 2) + 1, matrix(2, 2, 2))
    expect_equal(1 + matrix(1, 2, 2), matrix(2, 2, 2))
    
    # basic example
    D <- 3
    E <- 7
    M <- 100
    N <- 150
    x <- matrix(runif(M * D), M, D)
    y <- matrix(runif(N * D), N, D)
    z <- matrix(runif(N * E), N, E)
    x_i <- LazyTensor(x, index = 'i')
    y_j <- LazyTensor(y, index = 'j')
    z_j <- LazyTensor(z, index = 'j')
    xc_i <- LazyTensor(x, index = 'i', is_complex = TRUE)
    yc_j <- LazyTensor(y, index = 'j', is_complex = TRUE)
    
    # check results & formulas
    expect_equal(D + M, 103)
    
    expect_true(is.LazyTensor(x_i + y_j))
    
    expect_warning(tmp <- is.ComplexLazyTensor(x_i + xc_i))
    expect_true(tmp)
    
    expect_warning(tmp <- is.ComplexLazyTensor(xc_i + x_i))
    expect_true(tmp)
    
    expect_true(is.ComplexLazyTensor(xc_i + yc_j))
    
    obj <- x_i + y_j
    expect_equal(length(obj$args), 2)
    expect_equal(length(obj$data), 2)
    bool_grep_formula <- grep("A0x.*i\\+A0x.*j", obj$formula)
    expect_equal(bool_grep_formula, 1)
    
    expect_warning(obj <- x_i + xc_i)
    expect_equal(length(obj$args), 2)
    expect_equal(length(obj$data), 2)
    bool_grep_formula <- grep(
        "Add\\(Real2Complex\\(A0x.*i\\),A0x.*i\\)", obj$formula)
    expect_equal(bool_grep_formula, 1)
    
    expect_warning(obj <- xc_i + x_i)
    expect_equal(length(obj$args), 2)
    expect_equal(length(obj$data), 2)
    bool_grep_formula <- grep(
        "Add\\(A0x.*i,Real2Complex\\(A0x.*i\\)\\)", obj$formula)
    expect_equal(bool_grep_formula, 1)
    
    expect_warning(obj <-  xc_i + 3)
    expect_equal(length(obj$args), 1)
    expect_equal(length(obj$data), 1)
    bool_grep_formula <- grep("Add\\(A0x.*i,Real2Complex\\(IntCst\\(3\\)\\)\\)", obj$formula)
    expect_equal(bool_grep_formula, 1)
    
    expect_warning(obj <-  3 + xc_i)
    expect_equal(length(obj$args), 1)
    expect_equal(length(obj$data), 1)
    bool_grep_formula <- grep(
        "Add\\(Real2Complex\\(IntCst\\(3\\)\\),A0x.*i\\)", obj$formula)
    expect_equal(bool_grep_formula, 1)
    
    obj <-  x_i + 3
    expect_equal(length(obj$args), 1)
    expect_equal(length(obj$data), 1)
    bool_grep_formula <- grep("A0x.*i\\+IntCst\\(3\\)", obj$formula)
    expect_equal(bool_grep_formula, 1)
    
    obj <-  3.14 + x_i
    expect_equal(length(obj$args), 2)
    expect_equal(length(obj$data), 2)
    bool_grep_formula <- grep("A0x.*NA\\+A0x.*i", obj$formula)
    expect_equal(bool_grep_formula, 1)
    
    # errors
    expect_error(
        x_i + z_j,
        paste(
            "Operation `+` expects inputs of the same dimension or dimension 1.", 
            " Received 3 and 7.", 
            sep = ""
        ),
        fixed = TRUE
    )
    expect_error(+x_i)
    expect_error(+xc_i)
})


test_that("-", {
    
    # check that base operation is still working
    expect_equal(8-3, 5)
    expect_equal(-1, 0-1)
    expect_equal(c(3,6) - c(1,2), c(2,4))
    expect_equal(matrix(2, 2, 2) - matrix(1, 2, 2), matrix(1, 2, 2))
    expect_equal(matrix(2, 2, 2) - 1, matrix(1, 2, 2))
    expect_equal(2 - matrix(1, 2, 2), matrix(1, 2, 2))
    
    # basic example
    D <- 3
    E <- 7
    M <- 100
    N <- 150
    x <- matrix(runif(M * D), M, D)
    y <- matrix(runif(N * D), N, D)
    z <- matrix(runif(N * E), N, E)
    x_i <- LazyTensor(x, index = 'i')
    y_j <- LazyTensor(y, index = 'j')
    z_j <- LazyTensor(z, index = 'j')
    xc_i <- LazyTensor(x, index = 'i', is_complex = TRUE)
    yc_j <- LazyTensor(y, index = 'j', is_complex = TRUE)
    
    # check results & formulas
    expect_equal(D - D, 0)
    expect_equal(-D, -3)
    
    expect_true(is.LazyTensor(x_i - y_j))
    
    expect_warning(tmp <- is.ComplexLazyTensor(x_i - xc_i))
    expect_true(tmp)
    
    expect_true(is.ComplexLazyTensor(xc_i - yc_j))
    
    obj <- x_i - y_j
    expect_equal(length(obj$args), 2)
    expect_equal(length(obj$data), 2)
    bool_grep_formula <- grep("A0x.*i-A0x.*j", obj$formula)
    expect_equal(bool_grep_formula, 1)
    
    obj <- xc_i - yc_j
    expect_equal(length(obj$args), 2)
    expect_equal(length(obj$data), 2)
    bool_grep_formula <- grep("Subtract\\(A0x.*i,A0x.*j\\)", obj$formula)
    expect_equal(bool_grep_formula, 1)
    
    expect_warning(obj <- xc_i - y_j)
    expect_equal(length(obj$args), 2)
    expect_equal(length(obj$data), 2)
    bool_grep_formula <- grep(
        "Subtract\\(A0x.*i,Real2Complex\\(A0x.*j\\)\\)", obj$formula)
    expect_equal(bool_grep_formula, 1)
    
    expect_warning(obj <- x_i - yc_j)
    bool_grep_formula <- grep(
        "Subtract\\(Real2Complex\\(A0x.*i\\),A0x.*j\\)", obj$formula)
    expect_equal(bool_grep_formula, 1)
    
    expect_warning(obj <- xc_i - 3)
    expect_equal(length(obj$args), 1)
    expect_equal(length(obj$data), 1)
    bool_grep_formula <- grep(
        "Subtract\\(A0x.*i,Real2Complex\\(IntCst\\(3\\)\\)\\)", obj$formula)
    expect_equal(bool_grep_formula, 1)
    
    expect_warning(obj <- 3.3 - xc_i)
    expect_equal(length(obj$args), 2)
    expect_equal(length(obj$data), 2)
    bool_grep_formula <- grep(
        "Subtract\\(Real2Complex\\(A0x.*NA\\),A0x.*i\\)", obj$formula)
    expect_equal(bool_grep_formula, 1)
    
    obj <-  x_i - 3
    expect_equal(length(obj$args), 1)
    expect_equal(length(obj$data), 1)
    bool_grep_formula <- grep("A0x.*i-IntCst\\(3\\)", obj$formula)
    expect_equal(bool_grep_formula, 1)
    
    obj <-  -x_i
    expect_equal(length(obj$args), 1)
    expect_equal(length(obj$data), 1)
    bool_grep_formula <- grep("Minus\\(A0x.*i\\)", obj$formula)
    expect_equal(bool_grep_formula, 1)
    
    obj <-  3.14 - x_i
    expect_equal(length(obj$args), 2)
    expect_equal(length(obj$data), 2)
    bool_grep_formula <- grep("A0x.*NA-A0x.*i", obj$formula)
    expect_equal(bool_grep_formula, 1)
    
    # errors
    expect_error(
        x_i - z_j,
        paste(
            "Operation `-` expects inputs of the same dimension or dimension 1.", 
            " Received 3 and 7.", sep = ""
        ),
        fixed = TRUE
    )
})


test_that("*", {
    
    # check that base operation is still working
    expect_equal(2*3, 6)
    expect_equal(c(1,2) * c(1,2), c(1,4))
    expect_equal(matrix(1, 2, 2) * matrix(1, 2, 2), matrix(1, 2, 2))
    expect_equal(matrix(1, 2, 2) * 1, matrix(1, 2, 2))
    expect_equal(1 * matrix(1, 2, 2), matrix(1, 2, 2))
    
    # basic example
    D <- 3
    E <- 7
    M <- 100
    N <- 150
    x <- matrix(runif(M * D), M, D)
    y <- matrix(runif(N * D), N, D)
    z <- matrix(runif(N * E), N, E)
    x_i <- LazyTensor(x, index = 'i')
    y_j <- LazyTensor(y, index = 'j')
    z_j <- LazyTensor(z, index = 'j')
    xc_i <- LazyTensor(x, index = 'i', is_complex = TRUE)
    yc_j <- LazyTensor(y, index = 'j', is_complex = TRUE)
    
    # check results & formulas
    expect_equal(D * M, 300)
    
    expect_true(is.LazyTensor(x_i * y_j))
    
    expect_warning(tmp <- is.ComplexLazyTensor(x_i * yc_j))
    expect_true(tmp)
    
    expect_warning(tmp <- is.ComplexLazyTensor(xc_i * y_j))
    expect_true(tmp)
    
    expect_true(is.ComplexLazyTensor(xc_i * yc_j))
    
    obj <- x_i * y_j
    expect_equal(length(obj$args), 2)
    expect_equal(length(obj$data), 2)
    bool_grep_formula <- grep("A0x.*i\\*A0x.*j", obj$formula)
    expect_equal(bool_grep_formula, 1)
    
    expect_warning(obj <- x_i * xc_i)
    expect_equal(length(obj$args), 2)
    expect_equal(length(obj$data), 2)
    bool_grep_formula <- grep(
        "ComplexMult\\(Real2Complex\\(A0x.*i\\),A0x.*i\\)", obj$formula)
    expect_equal(bool_grep_formula, 1)
    
    expect_warning(obj <- xc_i * x_i)
    expect_equal(length(obj$args), 2)
    expect_equal(length(obj$data), 2)
    bool_grep_formula <- grep(
        "ComplexMult\\(A0x.*i,Real2Complex\\(A0x.*i\\)\\)", obj$formula)
    expect_equal(bool_grep_formula, 1)
    
    obj <- xc_i * Pm(2i)
    expect_equal(length(obj$args), 2)
    expect_equal(length(obj$data), 2)
    bool_grep_formula <- grep("ComplexScal\\(A0x.*i,A0x.*NA\\)", obj$formula)
    expect_equal(bool_grep_formula, 1)
    
    obj <- Pm(2i) * xc_i
    expect_equal(length(obj$args), 2)
    expect_equal(length(obj$data), 2)
    bool_grep_formula <- grep("ComplexScal\\(A0x.*NA,A0x.*i\\)", obj$formula)
    expect_equal(bool_grep_formula, 1)
    
    obj <- xc_i * Pm(2)
    expect_equal(length(obj$args), 1)
    expect_equal(length(obj$data), 1)
    bool_grep_formula <- grep(
        "ComplexRealScal\\(A0x.*i,IntCst\\(2\\)\\)", obj$formula)
    expect_equal(bool_grep_formula, 1)
    
    obj <- xc_i * 2
    expect_equal(length(obj$args), 1)
    expect_equal(length(obj$data), 1)
    bool_grep_formula <- grep(
        "ComplexRealScal\\(A0x.*i,IntCst\\(2\\)\\)", obj$formula)
    expect_equal(bool_grep_formula, 1)
    
    obj <- 2 * xc_i
    expect_equal(length(obj$args), 1)
    expect_equal(length(obj$data), 1)
    bool_grep_formula <- grep(
        "ComplexRealScal\\(IntCst\\(2\\),A0x.*i\\)", obj$formula)
    expect_equal(bool_grep_formula, 1)
    
    obj <-  x_i * 3
    expect_equal(length(obj$args), 1)
    expect_equal(length(obj$data), 1)
    bool_grep_formula <- grep("A0x.*i\\*+IntCst\\(3\\)", obj$formula)
    expect_equal(bool_grep_formula, 1)
    
    obj <-  3.14 * x_i
    expect_equal(length(obj$args), 2)
    expect_equal(length(obj$data), 2)
    bool_grep_formula <- grep("A0x.*NA\\*A0x.*i", obj$formula)
    expect_equal(bool_grep_formula, 1)
    
    # errors
    expect_error(
        x_i * z_j,
        paste(
            "Operation `*` expects inputs of the same dimension or dimension 1.",
            " Received 3 and 7.", sep = ""
        ),
        fixed = TRUE
    )
})


test_that("/", {
    
    # check that base operation is still working
    expect_equal(4/2, 2)
    expect_equal(c(3,2) / c(1,2), c(3,1))
    expect_equal(matrix(10, 2, 2) / matrix(2, 2, 2), matrix(5, 2, 2))
    expect_equal(matrix(1, 2, 2) / 10, matrix(0.1, 2, 2))
    expect_equal(1 / matrix(2, 2, 2), matrix(0.5, 2, 2))
    
    # basic example
    D <- 3
    E <- 7
    M <- 100
    N <- 150
    x <- matrix(runif(M * D), M, D)
    y <- matrix(runif(N * D), N, D)
    z <- matrix(runif(N * E), N, E)
    x_i <- LazyTensor(x, index = 'i')
    y_j <- LazyTensor(y, index = 'j')
    z_j <- LazyTensor(z, index = 'j')
    xc_i <- LazyTensor(x, index = 'i', is_complex = TRUE)
    yc_j <- LazyTensor(y, index = 'j', is_complex = TRUE)
    
    
    # check results & formulas
    expect_equal(D / M, 0.03)
    
    expect_true(is.LazyTensor(x_i / y_j))
    
    expect_warning(tmp <- is.ComplexLazyTensor(x_i / yc_j))
    expect_true(tmp)
    
    expect_warning(tmp <- is.ComplexLazyTensor(xc_i / y_j))
    expect_true(tmp)
    
    expect_true(is.ComplexLazyTensor(xc_i / yc_j))
    
    obj <- x_i / y_j
    expect_equal(length(obj$args), 2)
    expect_equal(length(obj$data), 2)
    bool_grep_formula <- grep("A0x.*i/A0x.*j", obj$formula)
    expect_equal(bool_grep_formula, 1)
    
    expect_warning(obj <- xc_i / y_j)
    expect_equal(length(obj$args), 2)
    expect_equal(length(obj$data), 2)
    bool_grep_formula <- grep(
        "ComplexDivide\\(A0x.*i,Real2Complex\\(A0x.*j\\)\\)", obj$formula)
    expect_equal(bool_grep_formula, 1)
    
    expect_warning(obj <- y_j / xc_i)
    expect_equal(length(obj$args), 2)
    expect_equal(length(obj$data), 2)
    bool_grep_formula <- grep(
        "ComplexDivide\\(Real2Complex\\(A0x.*j\\),A0x.*i\\)", obj$formula)
    expect_equal(bool_grep_formula, 1)
    
    expect_warning(obj <- yc_j / 3)
    expect_equal(length(obj$args), 1)
    expect_equal(length(obj$data), 1)
    bool_grep_formula <- grep(
        "ComplexDivide\\(A0x.*j,Real2Complex\\(IntCst\\(3\\)\\)\\)", 
        obj$formula
    )
    expect_equal(bool_grep_formula, 1)
    
    expect_warning(obj <- 3 / xc_i)
    expect_equal(length(obj$args), 1)
    expect_equal(length(obj$data), 1)
    bool_grep_formula <- grep(
        "ComplexDivide\\(Real2Complex\\(IntCst\\(3\\)\\),A0x.*i\\)", 
        obj$formula
    )
    expect_equal(bool_grep_formula, 1)
    
    obj <-  x_i / 3
    expect_equal(length(obj$args), 1)
    expect_equal(length(obj$data), 1)
    bool_grep_formula <- grep("A0x.*i/IntCst\\(3\\)", obj$formula)
    expect_equal(bool_grep_formula, 1)
    
    obj <-  3.14 / x_i
    expect_equal(length(obj$args), 2)
    expect_equal(length(obj$data), 2)
    bool_grep_formula <- grep("A0x.*NA/A0x.*i", obj$formula)
    expect_equal(bool_grep_formula, 1)
    
    # errors
    expect_error(
        x_i / z_j,
        paste(
            "Operation `/` expects inputs of the same dimension or dimension 1.",
            " Received 3 and 7.", sep = ""
        ),
        fixed = TRUE
    )
})


test_that("^", {
    
    # check that base operation is still working
    expect_equal(2^2, 4)
    expect_equal(c(3,2) ^ 2, c(9,4))
    expect_equal(matrix(2, 2, 2) ^ 2, matrix(4, 2, 2))
    
    # basic example
    D <- 3
    M <- 100
    N <- 150
    x <- matrix(runif(M * D), M, D)
    y <- matrix(runif(N * D), N, D)
    x_i <- LazyTensor(x, index = 'i')
    y_j <- LazyTensor(y, index = 'j')
    xc_i <- LazyTensor(x, index = 'i', is_complex = TRUE)
    yc_j <- LazyTensor(y, index = 'j', is_complex = TRUE)
    
    # check results & formulas
    expect_equal(D^D, 27)
    
    expect_s3_class(x_i^y_j, "LazyTensor")
    expect_s3_class(x_i^3, "LazyTensor")
    expect_s3_class(x_i^yc_j, "ComplexLazyTensor")
    expect_s3_class(xc_i^y_j, "ComplexLazyTensor")
    expect_s3_class(xc_i^yc_j, "ComplexLazyTensor")
    expect_s3_class(xc_i^2, "ComplexLazyTensor")
    expect_s3_class(xc_i^3, "ComplexLazyTensor")
    expect_s3_class(xc_i^0.5, "ComplexLazyTensor")
    
    
    obj <- x_i^y_j
    expect_equal(length(obj$args), 2)
    expect_equal(length(obj$data), 2)
    bool_grep_formula <- grep("Powf\\(A0x.*i,A0x.*j\\)", obj$formula)
    expect_equal(bool_grep_formula, 1)
    
    obj <-  x_i^3
    expect_equal(length(obj$args), 1)
    expect_equal(length(obj$data), 1)
    bool_grep_formula <- grep("Pow\\(A0x.*i,3\\)", obj$formula)
    expect_equal(bool_grep_formula, 1)
    
    obj <-  x_i^(-0.5)
    expect_equal(length(obj$args), 1)
    expect_equal(length(obj$data), 1)
    bool_grep_formula <- grep("Rsqrt\\(A0x.*i\\)", obj$formula)
    expect_equal(bool_grep_formula, 1)
    
    obj <-  x_i^(0.5)
    expect_equal(length(obj$args), 1)
    expect_equal(length(obj$data), 1)
    bool_grep_formula <- grep("Sqrt\\(A0x.*i\\)", obj$formula)
    expect_equal(bool_grep_formula, 1)
    
    obj <-  y_j^2
    expect_equal(length(obj$args), 1)
    expect_equal(length(obj$data), 1)
    bool_grep_formula <- grep("Square\\(A0x.*j\\)", obj$formula)
    expect_equal(bool_grep_formula, 1)
    
    obj <-  x_i^0.314
    expect_equal(length(obj$args), 2)
    expect_equal(length(obj$data), 2)
    bool_grep_formula <- grep("Powf\\(A0x.*i,A0x.*NA\\)", obj$formula)
    expect_equal(bool_grep_formula, 1)
    
    obj <- 3.14^x_i
    expect_equal(length(obj$args), 2)
    expect_equal(length(obj$data), 2)
    bool_grep_formula <- grep("Powf\\(A0x.*NA,A0x.*i\\)", obj$formula)
    expect_equal(bool_grep_formula, 1)
    
})


test_that("square", {
    
    # check that base operation is still working
    expect_equal(square(2), 4)
    expect_equal(square(c(3,2)), c(9,4))
    expect_equal(square(matrix(2, 2, 2)), matrix(4, 2, 2))
    
    # basic example
    D <- 3
    M <- 100
    N <- 150
    x <- matrix(runif(M * D), M, D)
    x_i <- LazyTensor(x, index = 'i')
    xc_i <- LazyTensor(x, index = 'i', is_complex = TRUE)
    
    expect_true(square(6) == 36)
    expect_s3_class(square(x_i), "LazyTensor")
    expect_s3_class(square(xc_i), "ComplexLazyTensor")
    
    obj <-  square(x_i)
    expect_equal(length(obj$args), 1)
    expect_equal(length(obj$data), 1)
    bool_grep_formula <- grep("Square\\(A0x.*i\\)", obj$formula)
    expect_equal(bool_grep_formula, 1)
    
})


test_that("sqrt", {
    
    # check that base operation is still working
    expect_equal(sqrt(4), 2)
    expect_equal(sqrt(c(9,4)), c(3,2))
    expect_equal(sqrt(matrix(4, 2, 2)), matrix(2, 2, 2))
    
    # basic example
    D <- 3
    M <- 100
    N <- 150
    x <- matrix(runif(M * D), M, D)
    x_i <- LazyTensor(x, index = 'i')
    xc_i <- LazyTensor(x, index = 'i', is_complex = TRUE)
    
    expect_true(sqrt(36) == 6)
    expect_s3_class(sqrt(x_i), "LazyTensor")
    expect_s3_class(sqrt(xc_i), "ComplexLazyTensor")
    
    obj <-  sqrt(x_i)
    expect_equal(length(obj$args), 1)
    expect_equal(length(obj$data), 1)
    bool_grep_formula <- grep("Sqrt\\(A0x.*i\\)", obj$formula)
    expect_equal(bool_grep_formula, 1)
    
})


test_that("rsqrt", {
    
    # check that base operation is still working
    expect_equal(rsqrt(100), 0.1)
    expect_equal(rsqrt(c(4,16)), c(0.5, 0.25))
    expect_equal(rsqrt(matrix(4, 2, 2)), matrix(0.5, 2, 2))
    
    # basic example
    D <- 3
    M <- 100
    N <- 150
    x <- matrix(runif(M * D), M, D)
    x_i <- LazyTensor(x, index = 'i')
    xc_i <- LazyTensor(x, index = 'i', is_complex = TRUE)
    
    expect_true(rsqrt(4) == 0.5)
    expect_s3_class(rsqrt(x_i), "LazyTensor")
    expect_s3_class(rsqrt(xc_i), "ComplexLazyTensor")
    
    obj <-  rsqrt(x_i)
    expect_equal(length(obj$args), 1)
    expect_equal(length(obj$data), 1)
    bool_grep_formula <- grep("Rsqrt\\(A0x.*i\\)", obj$formula)
    expect_equal(bool_grep_formula, 1)
    
})


test_that("|", {
    
    # check that base operation is still working
    expect_equal(TRUE | FALSE, TRUE)
    expect_equal(FALSE | TRUE, TRUE)
    expect_equal(FALSE | FALSE, FALSE)
    expect_equal(c(TRUE, FALSE) | c(FALSE, FALSE), c(TRUE, FALSE))
    expect_equal(
        matrix(FALSE, 2, 2) | matrix(TRUE, 2, 2), 
        matrix(TRUE, 2, 2)
    )
    
    # basic example
    D <- 3
    E <- 7
    M <- 100
    N <- 150
    x <- matrix(runif(M * D), M, D)
    y <- matrix(runif(N * D), N, D)
    z <- matrix(runif(N * E), N, E)
    x_i <- LazyTensor(x, index = 'i')
    y_j <- LazyTensor(y, index = 'j')
    z_j <- LazyTensor(z, index = 'j')
    
    xc_i <- LazyTensor(x, index = 'i', is_complex = TRUE)
    yc_j <- LazyTensor(y, index = 'j', is_complex = TRUE)
    
    # check results & formulas
    expect_equal(D | M, TRUE)
    
    expect_true(is.LazyTensor(x_i | y_j))
    expect_true(is.ComplexLazyTensor(xc_i | yc_j))
    expect_true(is.ComplexLazyTensor(xc_i | y_j))
    expect_true(is.ComplexLazyTensor(x_i | yc_j))
    
    obj <- x_i | y_j
    expect_equal(length(obj$args), 2)
    expect_equal(length(obj$data), 2)
    bool_grep_formula <- grep("\\(A0x.*i\\|A0x.*j\\)", obj$formula)
    expect_equal(bool_grep_formula, 1)
    
    # errors
    expect_error(
        x_i | z_j,
        "Operation `|` expects inputs of the same dimension. Received 3 and 7.",
        fixed = TRUE
    )
    
    expect_error(
        x_i | 3,
        "Operation `|` expects inputs of the same dimension. Received 3 and 1.",
        fixed = TRUE
    )
})


test_that("exp", {
    
    # check that base operation is still working
    expect_equal(exp(0), 1)
    expect_equal(exp(1), 2.718282, tolerance = 1e-5)
    expect_equal(exp(c(3,2)), c(20.085537, 7.389056), tolerance = 1e-5)
    expect_equal(exp(matrix(2, 2, 2)), matrix(7.389056, 2, 2), tolerance = 1e-5)
    
    # basic example
    D <- 3
    M <- 100
    N <- 150
    x <- matrix(runif(M * D), M, D)
    y <- matrix(runif(N * D), N, D)
    x_i <- LazyTensor(x, index = 'i')
    y_j <- LazyTensor(y, index = 'j')
    # basic example with complex exponential
    z <- matrix(1i^(-6:5), nrow = 4)                      # complex 4x3 matrix
    z_i <- LazyTensor(z, index = 'i', is_complex = TRUE)  # ComplexLazyTensor
    
    # check results, formulas & classes
    expect_equal(exp(0), 1)
    expect_true(is.LazyTensor(x_i))
    expect_true(is.ComplexLazyTensor(z_i))
    
    obj <- exp(x_i)
    expect_equal(length(obj$args), 1)
    expect_equal(length(obj$data), 1)
    bool_grep_formula <- grep("Exp\\(A0x.*i\\)", obj$formula)
    expect_equal(bool_grep_formula, 1)
    
    obj <- exp(x_i - y_j)
    expect_equal(length(obj$args), 2)
    expect_equal(length(obj$data), 2)
    bool_grep_formula <- grep("Exp\\(A0x.*i-A0x.*j\\)", obj$formula)
    expect_equal(bool_grep_formula, 1)
    
    obj <- exp(z_i)
    expect_equal(length(obj$args), 1)
    expect_equal(length(obj$data), 1)
    bool_grep_formula <- grep("ComplexExp\\(A0x.*i\\)", obj$formula)
    expect_equal(bool_grep_formula, 1)
})


test_that("log", {
    
    # check that base division is still working
    expect_equal(log(1), 0)
    expect_equal(log(10), 2.302585, tolerance = 1e-5)
    expect_equal(log(c(20.085537, 7.389056)), c(3,2), tolerance = 1e-5)
    expect_equal(log(matrix(7.389056, 2, 2)), matrix(2, 2, 2), tolerance = 1e-5)
    
    # basic example
    D <- 3
    M <- 100
    N <- 150
    x <- matrix(runif(M * D), M, D)
    y <- matrix(runif(N * D), N, D)
    x_i <- LazyTensor(x, index = 'i')
    y_j <- LazyTensor(y, index = 'j')
    # basic example with complex exponential
    z <- matrix(1i^(-6:5), nrow = 4)                      # complex 4x3 matrix
    z_i <- LazyTensor(z, index = 'i', is_complex = TRUE)  # ComplexLazyTensor
    
    # check results, formulas & classes
    expect_equal(log(1), 0)
    expect_true(!is.LazyTensor(log(x)))
    expect_true(is.LazyTensor(log(x_i)))
    expect_true(is.ComplexLazyTensor(log(z_i)))
    
    obj <- log(x_i)
    expect_equal(length(obj$args), 1)
    expect_equal(length(obj$data), 1)
    bool_grep_formula <- grep("Log\\(A0x.*i\\)", obj$formula)
    expect_equal(bool_grep_formula, 1)
    
    obj <-  log(x_i - y_j)
    expect_equal(length(obj$args), 2)
    expect_equal(length(obj$data), 2)
    bool_grep_formula <- grep("Log\\(A0x.*i-A0x.*j\\)", obj$formula)
    expect_equal(bool_grep_formula, 1)
    
    obj <- log(z_i)
    expect_equal(length(obj$args), 1)
    expect_equal(length(obj$data), 1)
    bool_grep_formula <- grep("Log\\(A0x.*i\\)", obj$formula)
    expect_equal(bool_grep_formula, 1)
})


test_that("inv", {
    
    # check that base operation is still working
    expect_equal(inv(10), 0.1)
    expect_equal(inv(1), 1)
    expect_equal(inv(c(10,4)), c(0.1,0.25))
    expect_equal(inv(matrix(2, 2, 2)), matrix(0.5, 2, 2))
    
    # basic example
    D <- 3
    M <- 100
    N <- 150
    x <- matrix(runif(M * D), M, D)
    y <- matrix(runif(N * D), N, D)
    x_i <- LazyTensor(x, index = 'i')
    y_j <- LazyTensor(y, index = 'j')
    # basic example with complex exponential
    z <- matrix(1i^(-6:5), nrow = 4)                      # complex 4x3 matrix
    z_i <- LazyTensor(z, index = 'i', is_complex = TRUE)  # ComplexLazyTensor
    
    # check results, formulas & classes
    expect_equal(inv(1), 1)
    expect_true(!is.LazyTensor(inv(x)))
    expect_true(is.LazyTensor(inv(x_i)))
    expect_true(is.ComplexLazyTensor(inv(z_i)))
    
    obj <- inv(x_i)
    expect_equal(length(obj$args), 1)
    expect_equal(length(obj$data), 1)
    bool_grep_formula <- grep("Inv\\(A0x.*i\\)", obj$formula)
    expect_equal(bool_grep_formula, 1)
    
    obj <-  inv(x_i + y_j)
    expect_equal(length(obj$args), 2)
    expect_equal(length(obj$data), 2)
    bool_grep_formula <- grep("Inv\\(A0x.*i\\+A0x.*j\\)", obj$formula)
    expect_equal(bool_grep_formula, 1)
    
    obj <-  inv(z_i)
    expect_equal(length(obj$args), 1)
    expect_equal(length(obj$data), 1)
    bool_grep_formula <- grep("Inv\\(A0x.*i\\)", obj$formula)
    expect_equal(bool_grep_formula, 1)
})


test_that("cos", {
    
    # check that base operation is still working
    expect_equal(cos(3.14159), -1, tolerance = 1e-5)
    expect_equal(cos(0), 1)
    expect_equal(cos(c(0,3.14159)), c(1, -1), tolerance = 1e-5)
    expect_equal(cos(matrix(0, 2, 2)), matrix(1, 2, 2))
    
    # basic example
    D <- 3
    M <- 100
    N <- 150
    x <- matrix(runif(M * D), M, D)
    y <- matrix(runif(N * D), N, D)
    x_i <- LazyTensor(x, index = 'i')
    y_j <- LazyTensor(y, index = 'j')
    # basic example with complex exponential
    z <- matrix(1i^(-6:5), nrow = 4)                      # complex 4x3 matrix
    z_i <- LazyTensor(z, index = 'i', is_complex = TRUE)  # ComplexLazyTensor
    
    # check results, formulas & classes
    expect_equal(cos(0), 1)
    expect_true(!is.LazyTensor(x))
    expect_true(is.LazyTensor(cos(x_i)))
    expect_true(is.ComplexLazyTensor(cos(z_i)))
    
    obj <- cos(x_i)
    expect_equal(length(obj$args), 1)
    expect_equal(length(obj$data), 1)
    bool_grep_formula <- grep("Cos\\(A0x.*i\\)", obj$formula)
    expect_equal(bool_grep_formula, 1)
    
    obj <-  cos(x_i + y_j)
    expect_equal(length(obj$args), 2)
    expect_equal(length(obj$data), 2)
    bool_grep_formula <- grep("Cos\\(A0x.*i\\+A0x.*j\\)", obj$formula)
    expect_equal(bool_grep_formula, 1)
    
    obj <-  cos(z_i)
    expect_equal(length(obj$args), 1)
    expect_equal(length(obj$data), 1)
    bool_grep_formula <- grep("Cos\\(A0x.*i\\)", obj$formula)
    expect_equal(bool_grep_formula, 1)
})


test_that("sin", {
    
    # check that base operation is still working
    expect_equal(sin(3.14159), 0, tolerance = 1e-5)
    expect_equal(sin(0), 0)
    expect_equal(sin(c(0,3.14159)), c(0,0), tolerance = 1e-5)
    expect_equal(sin(matrix(0, 2, 2)), matrix(0, 2, 2))
    
    # basic example
    D <- 3
    M <- 100
    N <- 150
    x <- matrix(runif(M * D), M, D)
    y <- matrix(runif(N * D), N, D)
    x_i <- LazyTensor(x, index = 'i')
    y_j <- LazyTensor(y, index = 'j')
    # basic example with complex exponential
    z <- matrix(1i^(-6:5), nrow = 4)                      # complex 4x3 matrix
    z_i <- LazyTensor(z, index = 'i', is_complex = TRUE)  # ComplexLazyTensor
    
    # check results, formulas & classes
    expect_equal(sin(0), 0)
    expect_true(!is.LazyTensor(sin(x)))
    expect_true(is.LazyTensor(sin(x_i)))
    expect_true(is.ComplexLazyTensor(sin(z_i)))
    
    obj <- sin(x_i)
    expect_equal(length(obj$args), 1)
    expect_equal(length(obj$data), 1)
    bool_grep_formula <- grep("Sin\\(A0x.*i\\)", obj$formula)
    expect_equal(bool_grep_formula, 1)
    
    obj <-  sin(x_i + y_j)
    expect_equal(length(obj$args), 2)
    expect_equal(length(obj$data), 2)
    bool_grep_formula <- grep("Sin\\(A0x.*i\\+A0x.*j\\)", obj$formula)
    expect_equal(bool_grep_formula, 1)
    
    obj <- sin(z_i)
    expect_equal(length(obj$args), 1)
    expect_equal(length(obj$data), 1)
    bool_grep_formula <- grep("Sin\\(A0x.*i\\)", obj$formula)
    expect_equal(bool_grep_formula, 1)
})


test_that("acos", {
    
    # check that base operation is still working
    expect_equal(acos(0) * 2, 3.14159, tolerance = 1e-5)
    expect_equal(acos(1), 0)
    expect_equal(acos(c(0,1)), c(3.14159 / 2, 0), tolerance = 1e-5)
    expect_equal(
        acos(matrix(0, 2, 2)), matrix(3.14159 / 2, 2, 2), tolerance = 1e-5)
    
    # basic example
    D <- 3
    M <- 100
    N <- 150
    x <- matrix(runif(M * D), M, D)
    y <- matrix(runif(N * D), N, D)
    x_i <- LazyTensor(x, index = 'i')
    y_j <- LazyTensor(y, index = 'j')
    xc_i <- LazyTensor(x, index = 'i', is_complex = TRUE)
    yc_j <- LazyTensor(y, index = 'j', is_complex = TRUE)
    
    # check results, formulas & classes
    expect_equal(acos(1), 0)
    expect_true(is.matrix(acos(x)))
    expect_true(is.LazyTensor(acos(x_i)))
    expect_true(is.ComplexLazyTensor(acos(xc_i)))
    
    obj <- acos(x_i)
    expect_equal(length(obj$args), 1)
    expect_equal(length(obj$data), 1)
    bool_grep_formula <- grep("Acos\\(A0x.*i\\)", obj$formula)
    expect_equal(bool_grep_formula, 1)
    
    obj <-  acos(x_i + y_j)
    expect_equal(length(obj$args), 2)
    expect_equal(length(obj$data), 2)
    bool_grep_formula <- grep("Acos\\(A0x.*i\\+A0x.*j\\)", obj$formula)
    expect_equal(bool_grep_formula, 1)
})


test_that("asin", {
    
    # check that base operation is still working
    expect_equal(asin(1) * 2, 3.14159, tolerance = 1e-5)
    expect_equal(asin(0), 0)
    expect_equal(asin(c(1,0)), c(3.14159 / 2, 0), tolerance = 1e-5)
    expect_equal(
        asin(matrix(1, 2, 2)), matrix(3.14159 / 2, 2, 2), tolerance = 1e-5)
    
    
    # basic example
    D <- 3
    M <- 100
    N <- 150
    x <- matrix(runif(M * D), M, D)
    y <- matrix(runif(N * D), N, D)
    x_i <- LazyTensor(x, index = 'i')
    y_j <- LazyTensor(y, index = 'j')
    
    # check results, formulas & classes
    expect_equal(asin(0), 0)
    expect_true(class(asin(x))[1] != "LazyTensor")
    expect_true(class(asin(x_i)) == "LazyTensor")
    
    obj <- asin(x_i)
    expect_equal(length(obj$args), 1)
    expect_equal(length(obj$data), 1)
    bool_grep_formula <- grep("Asin\\(A0x.*i\\)", obj$formula)
    expect_equal(bool_grep_formula, 1)
    
    obj <-  asin(x_i + y_j)
    expect_equal(length(obj$args), 2)
    expect_equal(length(obj$data), 2)
    bool_grep_formula <- grep("Asin\\(A0x.*i\\+A0x.*j\\)", obj$formula)
    expect_equal(bool_grep_formula, 1)
})


test_that("atan", {
    
    # check that base operation is still working
    expect_equal(atan(1) * 4, 3.14159, tolerance = 1e-5)
    expect_equal(atan(0), 0)
    expect_equal(atan(c(1,0)), c(3.14159 / 4, 0), tolerance = 1e-5)
    expect_equal(
        atan(matrix(1, 2, 2)), matrix(3.14159 / 4, 2, 2), tolerance = 1e-5)
    
    # basic example
    D <- 3
    M <- 100
    N <- 150
    x <- matrix(runif(M * D), M, D)
    y <- matrix(runif(N * D), N, D)
    x_i <- LazyTensor(x, index = 'i')
    y_j <- LazyTensor(y, index = 'j')
    
    # check results, formulas & classes
    expect_equal(atan(0), 0)
    expect_true(class(atan(x))[1] != "LazyTensor")
    expect_true(class(atan(x_i)) == "LazyTensor")
    
    obj <- atan(x_i)
    expect_equal(length(obj$args), 1)
    expect_equal(length(obj$data), 1)
    bool_grep_formula <- grep("Atan\\(A0x.*i\\)", obj$formula)
    expect_equal(bool_grep_formula, 1)
    
    obj <-  atan(x_i + y_j)
    expect_equal(length(obj$args), 2)
    expect_equal(length(obj$data), 2)
    bool_grep_formula <- grep("Atan\\(A0x.*i\\+A0x.*j\\)", obj$formula)
    expect_equal(bool_grep_formula, 1)
})


test_that("atan2", {
    
    # check that base operation is still working
    expect_equal(atan2(1, 1) * 4, 3.14159, tolerance = 1e-5)
    expect_equal(atan2(0, 1), 0)
    expect_equal(atan2(c(1, 0), c(1,1)), c(3.14159 / 4, 0), tolerance = 1e-5)
    expect_equal(
        atan2(matrix(1, 2, 2), matrix(1, 2, 2)), 
        matrix(3.14159 / 4, 2, 2), tolerance = 1e-5)
    
    # basic example
    D <- 3
    E <- 4
    M <- 100
    N <- 150
    x <- matrix(runif(M * D), M, D)
    y <- matrix(runif(N * D), N, D)
    x_i <- LazyTensor(x, index = 'i')
    y_j <- LazyTensor(y, index = 'j')
    z <- matrix(runif(N * E), N, E)
    x_i <- LazyTensor(x, index = 'i')
    y_j <- LazyTensor(y, index = 'j')
    z_j <- LazyTensor(z, index = 'j')
    
    # check results, formulas & classes
    expect_equal(atan2(0, 0), 0)
    expect_true(class(atan2(x_i, y_j)) == "LazyTensor")
    
    obj <- atan2(x_i, y_j)
    expect_equal(length(obj$args), 2)
    expect_equal(length(obj$data), 2)
    bool_grep_formula <- grep("Atan2\\(A0x.*i,A0x.*j\\)", obj$formula)
    expect_equal(bool_grep_formula, 1)
    
    # errors
    expect_error(atan2(x_i, z_j),
                 paste("Operation `Atan2` expects inputs of the",
                       " same dimension. Received 3 and 4.",
                       sep = ""),
                 fixed = TRUE)
    expect_error(atan2(x_i, 3),
                 paste("Operation `Atan2` expects inputs of the",
                       " same dimension. Received 3 and 1.",
                       sep = ""),
                 fixed = TRUE)
})


test_that("%*%", {
    
    # check that base operation is still working
    expect_equal(
        matrix(1:10, nrow = 2) %*% matrix(1:10, ncol = 2),
        matrix(c(95, 110, 220, 260), ncol = 2))
    expect_equal(
        matrix(1:10, ncol = 2) %*% matrix(1, nrow = 2, ncol = 1),
        as.matrix(rowSums(matrix(1:10, ncol = 2))))
    
    # basic example
    x <- matrix(c(1, 2, 3), 2, 3, byrow = TRUE)
    y <- matrix(c(3, 4, 5), 2, 3, byrow = TRUE)
    x_i <- LazyTensor(x, index = 'i')
    y_j <- LazyTensor(y, index = 'j')
    
    # check results
    a <- 3 %*% 100
    expect_equal(a[1], 300)
    
    res <- x_i %*% y_j
    expect_false(is.LazyTensor(res))
    expect_equal(dim(res), c(2, 3))
    expect_true(is.matrix(res))
    expected_res <- matrix(0, nrow = 2, ncol = 3)
    for(k in 1:3) {
        expected_res[[1, k]] <- x[[1, k]]*(y[[1, k]]+y[[2, k]])
        expected_res[[2, k]] <- x[[2, k]]*(y[[1, k]]+y[[2, k]])
    }
    expect_true(sum(abs(res - expected_res)) < 1E-5)
    
    res <-  x_i %*% Pm(2)
    expect_false(is.LazyTensor(res))
    expect_equal(dim(res), c(2, 3))
    expect_true(is.matrix(res))
    expected_res <- x * 2
    expect_true(sum(abs(res - expected_res)) < 1E-5)
    
    expect_error(
        x %*% y_j,
        paste(
            "`x` input argument should be a LazyTensor, a vector or a scalar.",
            "\nIf you want to use a matrix, convert it to LazyTensor first." ,
            sep = ""
        ),
        fixed = TRUE
    )
})


test_that("abs", {
    
    # check that base operation is still working
    expect_equal(abs(10), 10)
    expect_equal(abs(-10), 10)
    expect_equal(abs(c(5,-3)), c(5,3))
    expect_equal(abs(matrix(-2, 2, 2)), matrix(2, 2, 2))
    
    # basic example
    D <- 3
    M <- 100
    N <- 150
    x <- matrix(runif(M * D), M, D)
    y <- matrix(runif(N * D), N, D)
    x_i <- LazyTensor(x, index = 'i')
    y_j <- LazyTensor(y, index = 'j')
    xc_i <- LazyTensor(x, index = 'i', is_complex = TRUE)
    
    # check results, formulas & classes
    expect_equal(abs(-D), 3)
    expect_true(!is.LazyTensor(x))
    expect_true(is.LazyTensor(abs(x_i)))
    
    obj <- abs(x_i)
    expect_equal(length(obj$args), 1)
    expect_equal(length(obj$data), 1)
    bool_grep_formula <- grep("Abs\\(A0x.*i\\)", obj$formula)
    expect_equal(bool_grep_formula, 1)
    
    obj <-  abs(-y_j)
    expect_equal(length(obj$args), 1)
    expect_equal(length(obj$data), 1)
    bool_grep_formula <- grep("Abs\\(Minus\\(A0x.*j\\)\\)", obj$formula)
    expect_equal(bool_grep_formula, 1)
    
    obj <-  abs(-xc_i)
    expect_equal(length(obj$args), 1)
    expect_equal(length(obj$data), 1)
    bool_grep_formula <- grep("ComplexAbs\\(Minus\\(A0x.*i\\)\\)", obj$formula)
    expect_equal(bool_grep_formula, 1)
    
})


test_that("sign", {
    
    # check that base operation is still working
    expect_equal(sign(10), 1)
    expect_equal(sign(-10), -1)
    expect_equal(sign(c(5,-3)), c(1,-1))
    expect_equal(sign(matrix(-2, 2, 2)), matrix(-1, 2, 2))
    
    # basic example
    D <- 3
    M <- 100
    N <- 150
    x <- matrix(runif(M * D), M, D)
    y <- matrix(runif(N * D), N, D)
    x_i <- LazyTensor(x, index = 'i')
    y_j <- LazyTensor(y, index = 'j')
    
    # check results, formulas & classes
    expect_equal(sign(-D), -1)
    expect_true(class(sign(x))[1] != "LazyTensor")
    expect_true(class(sign(x_i)) == "LazyTensor")
    
    obj <- sign(x_i)
    expect_equal(length(obj$args), 1)
    expect_equal(length(obj$data), 1)
    bool_grep_formula <- grep("Sign\\(A0x.*i\\)", obj$formula)
    expect_equal(bool_grep_formula, 1)
    
    obj <-  sign(-y_j)
    expect_equal(length(obj$args), 1)
    expect_equal(length(obj$data), 1)
    bool_grep_formula <- grep("Sign\\(Minus\\(A0x.*j\\)\\)", obj$formula)
    expect_equal(bool_grep_formula, 1)
})


test_that("round", {
    
    # check that base operation is still working
    expect_equal(round(3), 3)
    expect_equal(round(4.5), 4)
    expect_equal(round(3.567, 2), 3.57)
    expect_equal(round(c(3.567, 4), 2), c(3.57, 4.00))
    expect_equal(round(matrix(3.567, 2, 2), 2), matrix(3.57, 2, 2))
    
    # basic example
    D <- 3
    M <- 100
    N <- 150
    x <- matrix(runif(M * D), M, D)
    y <- matrix(runif(N * D), N, D)
    x_i <- LazyTensor(x, index = 'i')
    y_j <- LazyTensor(y, index = 'j')
    xc_i <- LazyTensor(x, index = 'i', is_complex = TRUE)
    
    # check results, formulas & class
    expect_equal(round(pi, 2), 3.14)
    expect_false(is.LazyTensor(round(pi, 3)))
    expect_true(is.LazyTensor(round(x_i, 3)))
    expect_true(is.ComplexLazyTensor(round(xc_i, 3)))
    
    obj <- round(x_i, 3)
    expect_equal(length(obj$args), 1)
    expect_equal(length(obj$data), 1)
    bool_grep_formula <- grep("Round\\(A0x.*i,3\\)", obj$formula)
    expect_equal(bool_grep_formula, 1)
    
    obj <- round(y_j, 3.14)
    expect_equal(length(obj$args), 1)
    expect_equal(length(obj$data), 1)
    bool_grep_formula <- grep("Round\\(A0x.*j,3.14\\)", obj$formula)
    expect_equal(bool_grep_formula, 1)
    
    obj <- round(Pm(3), 3)
    expect_null(obj$args)
    expect_null(obj$data)
    bool_grep_formula <- grep("Round\\(IntCst\\(.*\\),3\\)", obj$formula)
    expect_equal(bool_grep_formula, 1)
    
    # errors
    expect_error(round(x_i, x), 
                 "`digits` input argument should be a scalar.", 
                 fixed = TRUE)
    
    expect_error(round(x_i, 2i), 
                 "`digits` input argument should be a scalar.", 
                 fixed = TRUE)
})


test_that("xlogx", {
    
    # check that base operation is still working
    expect_equal(xlogx(1), 0)
    expect_equal(xlogx(10), 23.02585, tolerance = 1e-5)
    expect_equal(xlogx(c(2,3)), c(1.386294, 3.295837), tolerance = 1e-5)
    expect_equal(
        xlogx(matrix(2, 2, 2)), matrix(1.386294, 2, 2), tolerance = 1e-5)
    
    # basic example
    D <- 3
    M <- 100
    N <- 150
    x <- matrix(runif(M * D), M, D)
    y <- matrix(runif(N * D), N, D)
    x_i <- LazyTensor(x, index = 'i')
    y_j <- LazyTensor(y, index = 'j')
    
    # check results, formulas & classes
    expect_equal(xlogx(1), 0)
    expect_equal(xlogx(0), 0) # check manually added limit
    expect_true(is.LazyTensor(xlogx(x_i)))
    
    obj <- xlogx(x_i)
    expect_equal(length(obj$args), 1)
    expect_equal(length(obj$data), 1)
    bool_grep_formula <- grep("XLogX\\(A0x.*i\\)", obj$formula)
    expect_equal(bool_grep_formula, 1)
    
    obj <- xlogx(Pm(3))
    expect_null(obj$args)
    expect_null(obj$data)
    bool_grep_formula <- grep("XLogX\\(IntCst\\(.*\\)\\)", obj$formula)
    expect_equal(bool_grep_formula, 1)
})


test_that("sinxdivx", {
    
    # check that base operation is still working
    expect_equal(sinxdivx(1), sin(1), tolerance = 1e-5)
    expect_equal(sinxdivx(1), 0.841471, tolerance = 1e-5)
    expect_equal(sinxdivx(c(3,2)), c(0.0470400, 0.4546487), tolerance = 1e-5)
    expect_equal(
        sinxdivx(matrix(1, 2, 2)), matrix(0.841471, 2, 2), tolerance = 1e-5)
    
    # basic example
    D <- 3
    M <- 100
    N <- 150
    x <- matrix(runif(M * D), M, D)
    y <- matrix(runif(N * D), N, D)
    x_i <- LazyTensor(x, index = 'i')
    y_j <- LazyTensor(y, index = 'j')
    
    # check results, formulas & classes
    expect_equal(sinxdivx(1), sin(1))
    expect_equal(sinxdivx(0), 1) # check manually added limit
    expect_true(is.LazyTensor(sinxdivx(x_i)))
    
    obj <- sinxdivx(x_i)
    expect_equal(length(obj$args), 1)
    expect_equal(length(obj$data), 1)
    bool_grep_formula <- grep("SinXDivX\\(A0x.*i\\)", obj$formula)
    expect_equal(bool_grep_formula, 1)
    
    obj <- sinxdivx(Pm(3))
    expect_null(obj$args)
    expect_null(obj$data)
    bool_grep_formula <- grep("SinXDivX\\(IntCst\\(.*\\)\\)", obj$formula)
    expect_equal(bool_grep_formula, 1)
})


test_that("step", {
    # basic example
    D <- 3
    M <- 100
    N <- 150
    x <- matrix(runif(M * D), M, D)
    x_i <- LazyTensor(x, index = 'i')
    xc_i <- LazyTensor(x, index = 'i', is_complex = TRUE)
    
    # R stat::step function, by default (R example in step doc)
    ctl <- c(4.17,5.58,5.18,6.11,4.50,4.61,5.17,4.53,5.33,5.14)
    trt <- c(4.81,4.17,4.41,3.59,5.87,3.83,6.03,4.89,4.32,4.69)
    group <- gl(2, 10, 20, labels = c("Ctl","Trt"))
    weight <- c(ctl, trt)
    lm.D9 <- lm(weight ~ group)
    expect_false(is.LazyTensor(step(lm.D9, trace = 0)))
    expect_s3_class(step(lm.D9, trace = 0), "lm")
    
    # check results, formulas & classes
    expect_true(is.LazyTensor(step(x_i)))
    expect_true(is.ComplexLazyTensor(step(xc_i)))
    
    obj <- step(x_i)
    expect_equal(length(obj$args), 1)
    expect_equal(length(obj$data), 1)
    bool_grep_formula <- grep("Step\\(A0x.*i\\)", obj$formula)
    expect_equal(bool_grep_formula, 1)
    
    obj <- step(Pm(2))
    expect_null(obj$args)
    expect_null(obj$data)
    bool_grep_formula <- grep("Step\\(IntCst\\(.*\\)\\)", obj$formula)
    expect_equal(bool_grep_formula, 1)
})


test_that("relu", {
    
    # check that base operation is still working
    expect_equal(relu(3), 3)
    expect_equal(relu(1:10), 1:10)
    expect_equal(relu(-9:10), c(rep(0, 10), 1:10))
    expect_equal(
        relu(matrix(-9:10, 10, 2)), 
        matrix(c(rep(0, 10), 1:10), 10, 2))
    
    # basic example
    D <- 3
    M <- 100
    N <- 150
    x <- matrix(runif(M * D), M, D)
    x_i <- LazyTensor(x, index = 'i')
    
    # check results, formulas & classes
    expect_false(is.LazyTensor(relu(2)))
    expect_true(is.LazyTensor(relu(x_i)))
    expect_true(is.LazyTensor(relu(x_i + 5)))
    
    obj <- relu(x_i)
    expect_equal(length(obj$args), 1)
    expect_equal(length(obj$data), 1)
    bool_grep_formula <- grep("ReLU\\(A0x.*i\\)", obj$formula)
    expect_equal(bool_grep_formula, 1)
    
    obj <- relu(LazyTensor(4))
    expect_null(obj$args)
    expect_null(obj$data)
    bool_grep_formula <- grep("ReLU\\(IntCst\\(.*\\)\\)", obj$formula)
    expect_equal(bool_grep_formula, 1)
})


test_that("clamp", {
    # basic example
    D <- 3
    M <- 100
    N <- 150
    P <- 200
    w <- matrix(runif(P * 7), P, 7)
    x <- matrix(runif(M * D), M, D)
    y <- matrix(runif(N * D), N, D)
    z <- matrix(runif(P * D), P, D)
    w_i <- LazyTensor(w, index = 'i')
    x_i <- LazyTensor(x, index = 'i')
    y_j <- LazyTensor(y, index = 'j')
    z_i <- LazyTensor(z, index = 'i')
    
    # complex 
    xc_i <- LazyTensor(x, index = 'i', is_complex = TRUE)
    yc_j <- LazyTensor(y, index = 'j', is_complex = TRUE)
    zc_i <- LazyTensor(z, index = 'i', is_complex = TRUE)
    
    # check formulas, args & classes
    obj <-  clamp(x_i, y_j, z_i)
    expect_true(is.LazyTensor(obj))
    expect_equal(length(obj$args), 3)
    expect_equal(length(obj$data), 3)
    bool_grep_formula <- grep("Clamp\\(A0x.*i,A0x.*j,A0x.*i\\)", obj$formula)
    expect_equal(bool_grep_formula, 1)
    
    obj <-  clamp(x_i, y_j, 3)
    expect_true(is.LazyTensor(obj))
    expect_equal(length(obj$args), 2)
    expect_equal(length(obj$data), 2)
    bool_grep_formula <- grep("Clamp\\(A0x.*i,A0x.*j,IntCst\\(3\\)\\)", obj$formula)
    expect_equal(bool_grep_formula, 1)
    
    obj <-  clamp(2, y_j, 3)
    expect_true(is.LazyTensor(obj))
    expect_equal(length(obj$args), 1)
    expect_equal(length(obj$data), 1)
    bool_grep_formula <- grep("Clamp\\(IntCst\\(2\\),A0x.*j,IntCst\\(3\\)\\)", obj$formula)
    expect_equal(bool_grep_formula, 1)
    
    obj <-  clamp(x_i, 2, 3)
    expect_true(is.LazyTensor(obj))
    expect_equal(length(obj$args), 1)
    expect_equal(length(obj$data), 1)
    bool_grep_formula <- grep("ClampInt\\(A0x.*i,2,3\\)", obj$formula)
    expect_equal(bool_grep_formula, 1)
    
    # errors
    expect_error(
        clamp(x_i, y_j, w_i),
        paste("Operation `Clamp` expects inputs of the same dimension or dimension 1.",
              " Received 3, 3 and 7.", sep = ""
        ),
        fixed = TRUE
    )
    
    expect_error(
        clamp(xc_i, yc_j, zc_i),
        paste0("`x`, `a` and `b` input arguments cannot be ComplexLazyTensors."),
        fixed = TRUE
    )
    
    expect_error(
        clamp(xc_i, y_j, z_i),
        paste0("`x`, `a` and `b` input arguments cannot be ComplexLazyTensors."),
        fixed = TRUE
    )
    
    expect_error(
        clamp(x_i, yc_j, z_i),
        paste0("`x`, `a` and `b` input arguments cannot be ComplexLazyTensors."),
        fixed = TRUE
    )
    
    expect_error(
        clamp(x_i, y_j, zc_i),
        paste0("`x`, `a` and `b` input arguments cannot be ComplexLazyTensors."),
        fixed = TRUE
    )
    
})


test_that("clampint", {
    # basic example
    D <- 3
    M <- 100
    N <- 150
    P <- 200
    x <- matrix(runif(M * D), M, D)
    y <- matrix(runif(N * D), N, D)
    z <- matrix(runif(P * D), P, D)
    x_i <- LazyTensor(x, index = 'i')
    y_j <- LazyTensor(y, index = 'j')
    z_i <- LazyTensor(z, index = 'i')
    xc_i <- LazyTensor(x, index = 'i', is_complex = TRUE)
    
    # check formulas, args & classes
    obj <-  clampint(x_i, 6, 8)
    expect_equal(length(obj$args), 1)
    expect_equal(length(obj$data), 1)
    expect_s3_class(obj, "LazyTensor")
    bool_grep_formula <- grep("ClampInt\\(A0x.*i,6,8\\)", obj$formula) 
    expect_equal(bool_grep_formula, 1)
    
    obj <-  clampint(1, 6, 8)
    expect_null(obj$args)
    expect_null(obj$data)
    expect_s3_class(obj, "LazyTensor")
    bool_grep_formula <- grep("ClampInt\\(IntCst\\(1\\),6,8\\)", obj$formula) 
    expect_equal(bool_grep_formula, 1)
    
    
    # errors
    expect_error(
        clampint(x_i, y_j, 8),
        paste("`clampint(x, y, z)` expects integer arguments for `y` and `z`.",
              " Use clamp(x, y, z) for different `y` and `z` types.", sep = ""
        ),
        fixed = TRUE
    )
    
    expect_error(
        clampint(x_i, y_j, z_i),
        paste("`clampint(x, y, z)` expects integer arguments for `y` and `z`.",
              " Use clamp(x, y, z) for different `y` and `z` types.", sep = ""
        ),
        fixed = TRUE
    )
    
    expect_error(
        clampint(xc_i, 2, 8),
        paste0("`x` cannot be a ComplexLazyTensor."),
        fixed = TRUE
    )
    
})


test_that("ifelse", {
    
    # check that base operation is still working
    expect_equal(ifelse(TRUE, 10, 1), 10)
    expect_equal(ifelse(FALSE, 10, 1), 1)
    expect_equal(ifelse(c(TRUE, FALSE), 10, 1), c(10, 1))
    expect_equal(
        ifelse(
            matrix(c(TRUE, FALSE, TRUE, FALSE), 2, 2), 
            matrix(10, 2, 2),
            matrix(1, 2, 2)),
        matrix(c(10, 1, 10, 1), 2, 2))
    
    # basic example
    D <- 3
    M <- 100
    N <- 150
    P <- 200
    w <- matrix(runif(P * 7), P, 7)
    x <- matrix(runif(M * D), M, D)
    y <- matrix(runif(N * D), N, D)
    z <- matrix(runif(P * D), P, D)
    w_i <- LazyTensor(w, index = 'i')
    x_i <- LazyTensor(x, index = 'i')
    y_j <- LazyTensor(y, index = 'j')
    z_i <- LazyTensor(z, index = 'i')
    
    # complex 
    xc_i <- LazyTensor(x, index = 'i', is_complex = TRUE)
    yc_j <- LazyTensor(y, index = 'j', is_complex = TRUE)
    zc_i <- LazyTensor(z, index = 'i', is_complex = TRUE)
    
    # default if-else R function
    a <- 9
    expect_equal(ifelse(a >= 0, sqrt(a), NA), 3)
    
    # check formulas, args & classes
    obj <-  ifelse(x_i, y_j, z_i)
    expect_equal(length(obj$args), 3)
    expect_equal(length(obj$data), 3)
    bool_grep_formula <- grep("IfElse\\(A0x.*i,A0x.*j,A0x.*i\\)", obj$formula)
    expect_equal(bool_grep_formula, 1)
    expect_s3_class(obj, "LazyTensor")
    
    obj <-  ifelse(x_i, y_j, 3)
    expect_equal(length(obj$args), 2)
    expect_equal(length(obj$data), 2)
    bool_grep_formula <- grep("IfElse\\(A0x.*i,A0x.*j,IntCst\\(3\\)\\)", obj$formula)
    expect_equal(bool_grep_formula, 1)
    expect_s3_class(obj, "LazyTensor")
    
    obj <-  ifelse(x_i, 2, 3)
    expect_equal(length(obj$args), 1)
    expect_equal(length(obj$data), 1)
    bool_grep_formula <- grep("IfElse\\(A0x.*i,IntCst\\(2\\),IntCst\\(3\\)\\)", obj$formula)
    expect_equal(bool_grep_formula, 1)
    expect_s3_class(obj, "LazyTensor")
    
    # errors
    expect_error(
        ifelse(x_i, y_j, w_i),
        paste(
            "Operation `IfElse` expects inputs of the same dimension or ", 
            "dimension 1.",
            " Received 3, 3 and 7.", sep = ""
        ),
        fixed = TRUE
    )
    
    expect_error(
        ifelse(xc_i, y_j, z_i),
        paste0(
            "`test`, `yes` and `no` input arguments cannot be ",
            "ComplexLazyTensors."),
        fixed = TRUE
    )
    
    expect_error(
        ifelse(x_i, yc_j, zc_i),
        paste0(
            "`test`, `yes` and `no` input arguments cannot be ",
            "ComplexLazyTensors."),
        fixed = TRUE
    )
    
})


test_that("mod", {
    # basic example
    D <- 3
    M <- 100
    N <- 150
    P <- 200
    w <- matrix(runif(P * 7), P, 7)
    x <- matrix(runif(M * D), M, D)
    y <- matrix(runif(N * D), N, D)
    z <- matrix(runif(P * D), P, D)
    w_i <- LazyTensor(w, index = 'i')
    x_i <- LazyTensor(x, index = 'i')
    y_j <- LazyTensor(y, index = 'j')
    z_i <- LazyTensor(z, index = 'i')
    
    # complex
    xc_i <- LazyTensor(x, index = 'i', is_complex = TRUE)
    yc_j <- LazyTensor(y, index = 'j', is_complex = TRUE)
    zc_i <- LazyTensor(z, index = 'i', is_complex = TRUE)
    
    # check formulas, args & classes
    obj <-  mod(x_i, y_j, z_i)
    expect_equal(length(obj$args), 3)
    expect_equal(length(obj$data), 3)
    bool_grep_formula <- grep("Mod\\(A0x.*i,A0x.*j,A0x.*i\\)", obj$formula)
    expect_equal(bool_grep_formula, 1)
    expect_s3_class(obj, "LazyTensor")
    
    obj <-  mod(xc_i, yc_j, zc_i)
    expect_equal(length(obj$args), 3)
    expect_equal(length(obj$data), 3)
    bool_grep_formula <- grep("Mod\\(A0x.*i,A0x.*j,A0x.*i\\)", obj$formula)
    expect_equal(bool_grep_formula, 1)
    expect_s3_class(obj, "ComplexLazyTensor")
    
    obj <-  mod(x_i, yc_j, zc_i)
    expect_equal(length(obj$args), 3)
    expect_equal(length(obj$data), 3)
    bool_grep_formula <- grep("Mod\\(A0x.*i,A0x.*j,A0x.*i\\)", obj$formula)
    expect_equal(bool_grep_formula, 1)
    expect_s3_class(obj, "ComplexLazyTensor")
    
    obj <-  mod(xc_i, y_j, z_i)
    expect_equal(length(obj$args), 3)
    expect_equal(length(obj$data), 3)
    bool_grep_formula <- grep("Mod\\(A0x.*i,A0x.*j,A0x.*i\\)", obj$formula)
    expect_equal(bool_grep_formula, 1)
    expect_s3_class(obj, "ComplexLazyTensor")
    
    obj <-  mod(x_i, y_j, 3)
    expect_equal(length(obj$args), 2)
    expect_equal(length(obj$data), 2)
    bool_grep_formula <- grep("Mod\\(A0x.*i,A0x.*j,IntCst\\(3\\)\\)", obj$formula)
    expect_equal(bool_grep_formula, 1)
    expect_s3_class(obj, "LazyTensor")
    
    obj <-  mod(x_i, 2, 3)
    expect_equal(length(obj$args), 1)
    expect_equal(length(obj$data), 1)
    bool_grep_formula <- grep("Mod\\(A0x.*i,IntCst\\(2\\),IntCst\\(3\\)\\)", obj$formula)
    expect_equal(bool_grep_formula, 1)
    expect_s3_class(obj, "LazyTensor")
    
    obj <-  mod(x_i, 2)
    expect_equal(length(obj$args), 1)
    expect_equal(length(obj$data), 1)
    bool_grep_formula <- grep("Mod\\(A0x.*i,IntCst\\(2\\),IntCst\\(0\\)\\)",
                              obj$formula)
    expect_equal(bool_grep_formula, 1)
    expect_s3_class(obj, "LazyTensor")
    
    # errors
    expect_error(
        mod(x_i, y_j, w_i),
        paste(
            "Operation `Mod` expects inputs of the same dimension or dimension 1.",
            " Received 3, 3 and 7.", sep = ""
        ),
        fixed = TRUE
    )
    
})


# TEST SIMPLE NORM AND DISTANCE OPERATIONS =====================================


test_that("sqnorm2", {
    # basic example
    D <- 3
    M <- 100
    N <- 150
    x <- matrix(runif(M * D), M, D)
    x_i <- LazyTensor(x, index = 'i')
    xc_i <- LazyTensor(x, index = 'i', is_complex = TRUE)
    
    # check results, formulas & classes
    expect_true(is.LazyTensor(sqnorm2(2)))
    expect_true(is.LazyTensor(sqnorm2(x_i)))
    expect_true(is.LazyTensor(sqnorm2(xc_i)))
    expect_false(is.ComplexLazyTensor(sqnorm2(xc_i)))
    
    obj <- sqnorm2(x_i)
    expect_equal(length(obj$args), 1)
    expect_equal(length(obj$data), 1)
    bool_grep_formula <- grep("SqNorm2\\(A0x.*i\\)", obj$formula)
    expect_equal(bool_grep_formula, 1)
})


test_that("norm2", {
    # basic example
    D <- 3
    M <- 100
    N <- 150
    x <- matrix(runif(M * D), M, D)
    x_i <- LazyTensor(x, index = 'i')
    xc_i <- LazyTensor(x, index = 'i', is_complex = TRUE)
    
    # check results, formulas & classes
    expect_true(is.LazyTensor(norm2(2)))
    expect_true(is.LazyTensor(norm2(x_i)))
    expect_true(is.LazyTensor(norm2(xc_i)))
    expect_false(is.ComplexLazyTensor(norm2(xc_i)))
    
    obj <- norm2(x_i)
    expect_equal(length(obj$args), 1)
    expect_equal(length(obj$data), 1)
    bool_grep_formula <- grep("Norm2\\(A0x.*i\\)", obj$formula)
    expect_equal(bool_grep_formula, 1)
})


test_that("normalize", {
    # basic example
    D <- 3
    M <- 100
    N <- 150
    x <- matrix(runif(M * D), M, D)
    x_i <- LazyTensor(x, index = 'i')
    xc_i <- LazyTensor(x, index = 'i', is_complex = TRUE)
    
    # check results, formulas & classes
    expect_true(is.LazyTensor(normalize(2)))
    expect_true(is.LazyTensor(normalize(x_i)))
    expect_true(is.ComplexLazyTensor(normalize(xc_i)))
    
    obj <- normalize(x_i)
    expect_equal(length(obj$args), 1)
    expect_equal(length(obj$data), 1)
    bool_grep_formula <- grep("Normalize\\(A0x.*i\\)", obj$formula)
    expect_equal(bool_grep_formula, 1)
})


test_that("sqdist", {
    # basic example
    D <- 3
    M <- 100
    N <- 150
    P <- 200
    E <- 7
    
    x <- matrix(runif(M * D), M, D)
    y <- matrix(runif(N * D), N, D)
    z <- matrix(runif(P * D), P, D)
    t <- matrix(runif(N * E), N, E)
    
    x_i <- LazyTensor(x, index = 'i')
    y_j <- LazyTensor(y, index = 'j')
    z_i <- LazyTensor(z, index = 'i')
    t_j <- LazyTensor(t, index = 'j')
    
    # complex
    xc_i <- LazyTensor(x, index = 'i', is_complex = TRUE)
    yc_j <- LazyTensor(y, index = 'j', is_complex = TRUE)
    
    
    # check results, formulas & classes
    expect_true(is.LazyTensor(sqdist(2, 3)))
    expect_true(is.LazyTensor(sqdist(x_i, y_j)))
    expect_false(is.ComplexLazyTensor(sqdist(xc_i, yc_j)))
    expect_true(is.LazyTensor(sqdist(xc_i, yc_j)))
    expect_false(is.ComplexLazyTensor(sqdist(x_i, yc_j)))
    expect_true(is.LazyTensor(sqdist(x_i, yc_j)))
    expect_false(is.ComplexLazyTensor(sqdist(xc_i, y_j)))
    expect_true(is.LazyTensor(sqdist(xc_i, y_j)))
    
    
    obj <- sqdist(x_i, y_j)
    expect_equal(length(obj$args), 2)
    expect_equal(length(obj$data), 2)
    bool_grep_formula <- grep("SqDist\\(A0x.*i,A0x.*j\\)", obj$formula)
    expect_equal(bool_grep_formula, 1)
    
    obj <- sqdist(x_i, 3)
    expect_equal(length(obj$args), 1)
    expect_equal(length(obj$data), 1)
    bool_grep_formula <- grep("SqDist\\(A0x.*i,IntCst\\(3\\)\\)", obj$formula)
    expect_equal(bool_grep_formula, 1)
    
    # errors
    expect_error(
        sqdist(x_i, t_j),
        paste(
            "Operation `SqDist` expects inputs of the same dimension or dimension 1.",
            " Received 3 and 7.", sep = ""
        ),
        fixed = TRUE
    )
})


test_that("weightedsqnorm", {
    # basic example
    D <- 3
    M <- 100
    x <- matrix(runif(M * D), M, D)
    s <- matrix(runif(M * D), M, D)
    x_i <- LazyTensor(x, index = 'i')
    s_j <- LazyTensor(s, index = 'j')
    xc_i <- LazyTensor(x, index = 'i', is_complex = TRUE)
    sc_j <- LazyTensor(s, index = 'j', is_complex = TRUE)
    
    # check results, formulas & classes
    expect_true(is.LazyTensor(weightedsqnorm(2, 3)))
    expect_true(is.LazyTensor(weightedsqnorm(x_i, s_j)))
    expect_false(is.ComplexLazyTensor(weightedsqnorm(xc_i, sc_j)))
    expect_true(is.LazyTensor(weightedsqnorm(xc_i, sc_j)))
    expect_false(is.ComplexLazyTensor(weightedsqnorm(x_i, sc_j)))
    expect_true(is.LazyTensor(weightedsqnorm(x_i, sc_j)))
    expect_false(is.ComplexLazyTensor(weightedsqnorm(xc_i, s_j)))
    expect_true(is.LazyTensor(weightedsqnorm(xc_i, s_j)))
    
    obj <- weightedsqnorm(x_i, s_j)
    expect_equal(length(obj$args), 2)
    expect_equal(length(obj$data), 2)
    bool_grep_formula <- grep("WeightedSqNorm\\(A0x.*j,A0x.*i\\)", obj$formula)
    expect_equal(bool_grep_formula, 1)
    
    obj <- weightedsqnorm(x_i, 3)
    expect_equal(length(obj$args), 1)
    expect_equal(length(obj$data), 1)
    bool_grep_formula <- grep("WeightedSqNorm\\(IntCst\\(3\\),A0x.*i\\)", obj$formula)
    expect_equal(bool_grep_formula, 1)
})


test_that("weightedsqdist", {
    
    D <- 3
    M <- 100
    
    x <- matrix(runif(M * D), M, D)
    y <- matrix(runif(M * D), M, D)
    s <- matrix(runif(M * D), M, D)
    
    x_i <- LazyTensor(x, index = 'i')
    y_j <- LazyTensor(y, index = 'j')
    s_i <- LazyTensor(s, index = 'i')
    
    # complex
    xc_i <- LazyTensor(x, index = 'i', is_complex = TRUE)
    yc_j <- LazyTensor(y, index = 'j', is_complex = TRUE)
    sc_i <- LazyTensor(s, index = 'i', is_complex = TRUE)
    
    # check results, formulas & classes
    expect_true(is.LazyTensor(weightedsqdist(2, 3, 1)))
    expect_true(is.LazyTensor(weightedsqdist(x_i, y_j, s_i)))
    expect_false(is.ComplexLazyTensor(weightedsqdist(xc_i, yc_j, sc_i)))
    expect_true(is.LazyTensor(weightedsqdist(xc_i, yc_j, sc_i)))
    expect_warning(tmp <- is.ComplexLazyTensor(weightedsqdist(x_i, yc_j, sc_i)))
    expect_false(tmp)
    expect_warning(tmp <- is.LazyTensor(weightedsqdist(x_i, yc_j, sc_i)))
    expect_true(tmp)
    expect_warning(tmp <- is.ComplexLazyTensor(weightedsqdist(xc_i, y_j, s_i)))
    expect_false(tmp)
    expect_warning(tmp <- is.LazyTensor(weightedsqdist(xc_i, y_j, s_i)))
    expect_true(tmp)
    
    obj <- weightedsqdist(x_i, y_j, s_i)
    expect_equal(length(obj$args), 3)
    expect_equal(length(obj$data), 3)
    bool_grep_formula <- grep("WeightedSqNorm\\(A0x.*i,A0x.*i-A0x.*j\\)", 
                              obj$formula)
    expect_equal(bool_grep_formula, 1)
    
    obj <- weightedsqdist(x_i, 2, s_i)
    expect_equal(length(obj$args), 2)
    expect_equal(length(obj$data), 2)
    bool_grep_formula <- grep("WeightedSqNorm\\(A0x.*i,A0x.*i-IntCst\\(2\\)\\)", 
                              obj$formula)
    expect_equal(bool_grep_formula, 1)
    
    obj <- weightedsqdist(x_i, y_j, 3)
    expect_equal(length(obj$args), 2)
    expect_equal(length(obj$data), 2)
    bool_grep_formula <- grep("WeightedSqNorm\\(IntCst\\(3\\),A0x.*i-A0x.*j\\)", 
                              obj$formula)
    expect_equal(bool_grep_formula, 1)
    
    obj <- weightedsqdist(x_i, 2, 3)
    expect_equal(length(obj$args), 1)
    expect_equal(length(obj$data), 1)
    bool_grep_formula <- grep(
        "WeightedSqNorm\\(IntCst\\(3\\),A0x.*i-IntCst\\(2\\)\\)", 
        obj$formula
    )
    expect_equal(bool_grep_formula, 1)
    
})




# TEST COMPLEX FUNCTIONS =======================================================


test_that("Re", {
    
    # check that base operation is still working
    expect_equal(Re(1), 1)
    expect_equal(Re(1i), 0)
    expect_equal(Re(1 + 1i), 1)
    expect_equal(Re(10 + 1i), 10)
    expect_equal(Re(c(10 + 1i, 5 + 1i)), c(10, 5))
    expect_equal(Re(matrix(1+1i, 2, 2)), matrix(1, 2, 2))
    
    # basic example
    D <- 3
    M <- 100
    x <- matrix(runif(M * D), M, D)
    z <- matrix(1i^ (-6:5), nrow = 4)
    
    x_i <- LazyTensor(x, index = 'i')
    
    # complex
    z_i <- LazyTensor(z, index = 'i', is_complex = TRUE)
    
    # check formulas, args & classes
    expect_true(Re(2) == 2)
    expect_true(Re(2 + 1i) == 2)
    
    
    obj <-  Re(z_i)
    expect_equal(length(obj$args), 1)
    expect_equal(length(obj$data), 1)
    bool_grep_formula <- grep("ComplexReal\\(A0x.*i\\)", obj$formula)
    expect_equal(bool_grep_formula, 1)
    expect_s3_class(obj, "LazyTensor")
    
    expect_error(
        Re(x_i), 
        "`Re` cannot be applied to a LazyTensor. See `?Re` for compatible types.",
        fixed = TRUE
    )
    
})


test_that("Im", {
    
    # check that base operation is still working
    expect_equal(Im(1), 0)
    expect_equal(Im(1i), 1)
    expect_equal(Im(1 + 1i), 1)
    expect_equal(Im(10 + 10i), 10)
    expect_equal(Im(c(10 + 10i, 5 + 5i)), c(10, 5))
    expect_equal(Im(matrix(1+1i, 2, 2)), matrix(1, 2, 2))
    
    # basic example
    D <- 3
    M <- 100
    
    x <- matrix(runif(M * D), M, D)
    z <- matrix(1i^ (-6:5), nrow = 4)
    
    x_i <- LazyTensor(x, index = 'i')
    
    # complex
    z_i <- LazyTensor(z, index = 'i', is_complex = TRUE)
    
    # check formulas & classes
    expect_true(Im(2) == 0)
    expect_true(Im(2+1i) == 1)
    
    obj <-  Im(z_i)
    expect_equal(length(obj$args), 1)
    expect_equal(length(obj$data), 1)
    bool_grep_formula <- grep("ComplexImag\\(A0x.*i\\)", obj$formula)
    expect_equal(bool_grep_formula, 1)
    expect_s3_class(obj, "LazyTensor")
    
    expect_error(
        Im(x_i), 
        "`Im` cannot be applied to a LazyTensor. See `?Im` for compatible types.",
        fixed = TRUE
    )
    
})


test_that("Arg", {
    
    # check that base operation is still working
    expect_equal(Arg(1), 0)
    expect_equal(Arg(1i), 3.141593 / 2, tolerance = 1e-5)
    expect_equal(Arg(1 + 1i), 3.141593 / 4, tolerance = 1e-5)
    expect_equal(
        Arg(c(1 + 1i, 0 + 1i)), c(3.141593 / 4, 3.141593 / 2), tolerance = 1e-5)
    expect_equal(
        Arg(matrix(1+1i, 2, 2)), matrix(3.141593 / 4, 2, 2), tolerance = 1e-5)
    
    # basic example
    D <- 3
    M <- 100
    x <- matrix(runif(M * D), M, D)
    z <- matrix(1i^ (-6:5), nrow = 4)
    
    x_i <- LazyTensor(x, index = 'i')
    
    # complex
    z_i <- LazyTensor(z, index = 'i', is_complex = TRUE)
    
    # check formulas & classes
    expect_true(Arg(2) == 0)
    expect_true(round(Arg(pi*1i),2) == 1.57)
    
    obj <- Arg(z_i)
    expect_equal(length(obj$args), 1)
    expect_equal(length(obj$data), 1)
    bool_grep_formula <- grep("ComplexAngle\\(A0x.*i\\)", obj$formula)
    expect_equal(bool_grep_formula, 1)
    expect_s3_class(obj, "LazyTensor")
    
    # error
    expect_error(
        Arg(x_i), 
        "`Arg` cannot be applied to a LazyTensor. See `?Arg` for compatible types.",
        fixed = TRUE
    )
    
})


test_that("real2complex", {
    # basic example
    D <- 3
    M <- 100
    x <- matrix(runif(M * D), M, D)
    # LazyTensor
    x_i <- LazyTensor(x, index = 'i')
    # complex
    xc_i <- LazyTensor(x, index = 'i', is_complex = TRUE)
    
    # check formulas & classes
    expect_warning(obj <-  real2complex(x_i))
    expect_equal(length(obj$args), 1)
    expect_equal(length(obj$data), 1)
    expect_equal(obj$dimres, x_i$dimres)
    bool_grep_formula <- grep("Real2Complex\\(A0x.*i\\)", obj$formula)
    expect_equal(bool_grep_formula, 1)
    expect_s3_class(obj, "ComplexLazyTensor")
    
    expect_warning(obj <-  real2complex(Pm(2)))
    expect_null(obj$args)
    expect_null(obj$data)
    expect_equal(obj$dimres, 1)
    bool_grep_formula <- grep("Real2Complex\\(IntCst\\(2\\)\\)", obj$formula)
    expect_equal(bool_grep_formula, 1)
    expect_s3_class(obj, "ComplexLazyTensor")
    
    # errors
    expect_error(real2complex(xc_i), 
                 "`real2complex` cannot be applied to a ComplexLazyTensor.",
                 fixed = TRUE)
    
    # Should always produce a warning
    expect_warning(real2complex(x_i))
    expect_warning(real2complex(Pm(3)))
    
})


test_that("imag2complex", {
    # basic example
    D <- 3
    M <- 100
    x <- matrix(runif(M * D), M, D)
    x_i <- LazyTensor(x, index = 'i')
    xc_i <- LazyTensor(x, index = 'i', is_complex = TRUE)
    
    # check formulas & classes
    obj <-  imag2complex(x_i)
    expect_equal(length(obj$args), 1)
    expect_equal(length(obj$data), 1)
    expect_equal(obj$dimres, x_i$dimres)
    bool_grep_formula <- grep("Imag2Complex\\(A0x.*i\\)", obj$formula)
    expect_equal(bool_grep_formula, 1)
    expect_s3_class(obj, "ComplexLazyTensor")
    
    obj <-  imag2complex(Pm(2))
    expect_null(obj$args)
    expect_null(obj$data)
    expect_equal(obj$dimres, 1)
    bool_grep_formula <- grep("Imag2Complex\\(IntCst\\(2\\)\\)", obj$formula)
    expect_equal(bool_grep_formula, 1)
    expect_s3_class(obj, "ComplexLazyTensor")
    
    expect_error(imag2complex(xc_i), 
                 "`imag2complex` cannot be applied to a ComplexLazyTensor.",
                 fixed = TRUE)
    
})


test_that("exp1j", {
    # basic example
    D <- 3
    M <- 100
    x <- matrix(runif(M * D), M, D)
    x_i <- LazyTensor(x, index = 'i')
    xc_i <- LazyTensor(x, index = 'i', is_complex = TRUE)
    
    
    # check formulas & classes
    obj <-  exp1j(x_i)
    expect_equal(length(obj$args), 1)
    expect_equal(length(obj$data), 1)
    expect_equal(obj$dimres, x_i$dimres)
    bool_grep_formula <- grep("ComplexExp1j\\(A0x.*i\\)", obj$formula)
    expect_equal(bool_grep_formula, 1)
    expect_s3_class(obj, "ComplexLazyTensor")
    
    expect_error(exp1j(xc_i), 
                 "`exp1j` cannot be applied to a ComplexLazyTensor.",
                 fixed = TRUE)
    
})


test_that("Conj", {
    
    # check that base operation is still working
    expect_equal(Conj(10), 10)
    expect_equal(Conj(10i), -10i)
    expect_equal(Conj(10+10i), 10-10i)
    expect_equal(Conj(c(10+10i,2)), c(10-10i, 2))
    expect_equal(
        Conj(matrix(10-10i, 2, 2)), matrix(10+10i, 2, 2))
    
    # basic example
    D <- 3
    M <- 100
    N <- 150
    x <- matrix(runif(M * D), M, D)
    x_i <- LazyTensor(x, index = 'i')
    
    z <- matrix(2i^(-6:5), nrow = 4) # complex 4x3 matrix
    z_i <- LazyTensor(z, index = 'i', is_complex = TRUE)  # ComplexLazyTensor
    
    # check results, formulas & classes
    obj <- Conj(z_i)
    expect_s3_class(obj, "ComplexLazyTensor")
    expect_equal(length(obj$args), 1)
    expect_equal(length(obj$data), 1)
    bool_grep_formula <- grep("Conj\\(A0x.*i\\)", obj$formula)
    expect_equal(bool_grep_formula, 1)
    
    obj <- Conj(1 + 2i)
    expect_equal(obj, 1 - 2i)
    expect_type(obj, "complex")
    
    # errors
    expect_error(
        Conj(x_i),
        "`Conj` cannot be applied to a LazyTensor. See `?Conj` for compatible types.",
        fixed = TRUE
    )
})


test_that("Mod", {
    
    # check that base operation is still working
    expect_equal(Mod(10), 10)
    expect_equal(Mod(10i), 10)
    expect_equal(Mod(1+1i), sqrt(2))
    expect_equal(Mod(c(1+1i, 2)), c(sqrt(2), 2))
    expect_equal(
        Mod(matrix(1+1i, 2, 2)), matrix(sqrt(2), 2, 2))
    
    # basic example
    D <- 3
    M <- 100
    x <- matrix(runif(M * D), M, D)
    x_i <- LazyTensor(x, index = 'i')
    
    z <- matrix(2i^(-6:5), nrow = 4) # complex 4x3 matrix
    z_i <- LazyTensor(z, index = 'i', is_complex = TRUE)  # ComplexLazyTensor
    
    # check results, formulas & classes
    obj <- Mod(z_i)
    expect_equal(length(obj$args), 1)
    expect_equal(length(obj$data), 1)
    expect_equal(obj$dimres, 3)
    expect_true(is.LazyTensor(obj))
    expect_false(is.ComplexLazyTensor(obj))
    bool_grep_formula <- grep("ComplexAbs\\(A0x.*i\\)", obj$formula)
    expect_equal(bool_grep_formula, 1)
})



# TEST CONSTANT AND PADDING/CONCATENATION OPERATIONS ===========================


test_that("elem", {
    # basic example
    int <- 5
    Pm_int <- LazyTensor(int)
    
    x <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows, 3 columns
    x_i <- LazyTensor(x, index = 'i')   # LazyTensor from matrix x, indexed by 'i'
    
    m <- 2
    
    # check formulas, args & classes
    obj <- elem(x_i, m)
    expect_true(is.LazyTensor(obj))
    bool_grep_formula <- grep("Elem\\(A0x.*i,2\\)", obj$formula)
    expect_equal(bool_grep_formula, 1)
    
    obj <- elem(Pm_int, 0)
    expect_true(is.LazyTensor(obj))
    bool_grep_formula <- grep("Elem\\(IntCst\\(5\\),0\\)", obj$formula)
    expect_equal(bool_grep_formula, 1)
    expect_null <- obj$args
    expect_null <- obj$data
    expect_equal(obj$dimres, 1)
    
    # errors
    expect_error(elem(x_i, 3.14),
                 "`m` input argument should be an integer.",
                 fixed = TRUE)
    expect_error(elem(x_i, 4),
                 "Index `m` is out of bounds. Should be in [0, 3).",
                 fixed = TRUE)
    expect_error(elem(x_i, -1),
                 "Index `m` is out of bounds. Should be in [0, 3).",
                 fixed = TRUE)
    
})


test_that("elemT", {
    
    # basic example
    x <- 3.14              # arbitrary value
    Pm_x <- LazyTensor(x)  # creating scalar parameter LazyTensor from x
    
    int <- 4
    Pm_int <- LazyTensor(int)
    
    m <- 2
    n <- 3
    
    # Complex single-value LazyTensor just for test purposes
    z <- 2 + 3i # arbitrary complex value
    Pm_z <- LazyTensor(z)
    
    # Matrix LazyTensor just for error purposes
    y <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows, 3 columns
    y_i <- LazyTensor(y, index = 'i')   # LazyTensor from matrix x, indexed by 'i'
    
    # check formulas, args & classes
    obj <- elemT(Pm_x, m, n)
    expect_true(is.LazyTensor(obj))
    bool_grep_formula <- grep("ElemT\\(A0x.*NA,3,2\\)", obj$formula)
    expect_equal(bool_grep_formula, 1)
    
    obj <- elemT(Pm_z, m, n)
    expect_true(is.LazyTensor(obj))
    expect_true(is.ComplexLazyTensor(obj))
    bool_grep_formula <- grep("ElemT\\(A0x.*NA,3,2\\)", obj$formula)
    expect_equal(bool_grep_formula, 1)
    
    obj <- elemT(Pm_int, 5, 7)
    expect_true(is.LazyTensor(obj))
    bool_grep_formula <- grep("ElemT\\(IntCst\\(4\\),7,5\\)", obj$formula)
    expect_equal(bool_grep_formula, 1)
    expect_null <- obj$args
    expect_null <- obj$data
    expect_equal(obj$dimres, 1)
    
    # errors
    expect_error(elemT(y_i + y_i, m, n),
                 paste("`x` input argument should be a `LazyTensor`", 
                       " encoding a single value.", sep = ""),
                 fixed = TRUE)
    expect_error(elemT(Pm_x, 3.14, 12),
                 "`m` input argument should be an integer.",
                 fixed = TRUE)
    expect_error(elemT(Pm_x, m, 3.14),
                 "`n` input argument should be an integer.",
                 fixed = TRUE)
    expect_error(elemT(Pm_x, -1, n),
                 "Index `m` is out of bounds. Should be in [0, 3).",
                 fixed = TRUE)
    expect_error(elemT(Pm_x, 3, n),
                 "Index `m` is out of bounds. Should be in [0, 3).",
                 fixed = TRUE)
    
})


test_that("extract", {
    # basic example
    x <- matrix(runif(150 * 5), 150, 5) # arbitrary R matrix, 150 rows, 5 columns
    x_i <- LazyTensor(x, index = 'i')   # LazyTensor from matrix x, indexed by 'i'
    
    int <- 2
    Pm_int <- LazyTensor(int)
    
    v <- c(1, 2, 3, 1, 5)
    Pm_v <- LazyTensor(v)
    
    m <- 1
    d <- 3
    
    # check formulas, args & classes
    obj <- extract(x_i, m, d)
    expect_true(is.LazyTensor(obj))
    bool_grep_formula <- grep("Extract\\(A0x.*i,1,3\\)", obj$formula)
    expect_equal(bool_grep_formula, 1)
    
    obj <- extract(Pm_v, 0, d)
    expect_true(is.LazyTensor(obj))
    bool_grep_formula <- grep("Extract\\(A0x.*NA,0,3\\)", obj$formula)
    expect_equal(bool_grep_formula, 1)
    
    obj <- extract(Pm_int, 0, 1)
    expect_true(is.LazyTensor(obj))
    bool_grep_formula <- grep("Extract\\(IntCst\\(2\\),0,1\\)", obj$formula)
    expect_equal(bool_grep_formula, 1)
    expect_null <- obj$args
    expect_null <- obj$data
    expect_equal(obj$dimres, 1)
    
    # errors
    expect_error(extract(x_i, 3.14, d),
                 "`m` input argument should be an integer.",
                 fixed = TRUE)
    expect_error(extract(x_i, m, 3.14),
                 "`d` input argument should be an integer.",
                 fixed = TRUE)
    expect_error(extract(x_i, 7, d),
                 "Index `m` is out of bounds. Should be in [0, 5).",
                 fixed = TRUE)
    expect_error(extract(x_i, -1, d),
                 "Index `m` is out of bounds. Should be in [0, 5).",
                 fixed = TRUE)
    expect_error(extract(x_i, 4, d),
                 paste("Slice dimension is out of bounds. Input `d` should be ",
                       "in [0, 5-m] where `m` is the starting index.", sep = ""),
                 fixed = TRUE)
    expect_error(extract(x_i, 2, -1),
                 paste("Slice dimension is out of bounds. Input `d` should be ",
                       "in [0, 5-m] where `m` is the starting index.", sep = ""),
                 fixed = TRUE)
})


test_that("extractT", {
    # basic example
    x <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows, 3 columns
    x_i <- LazyTensor(x, index = 'i')   # LazyTensor from matrix x, indexed by 'i'
    
    v <- c(1, 2, 3, 1, 5)
    Pm_v <- LazyTensor(v)
    
    int <- 2
    Pm_int <- LazyTensor(int)
    
    m <- 1
    d <- 8
    
    # check formulas, args & classes
    obj <- extractT(x_i, m, d)
    expect_true(is.LazyTensor(obj))
    bool_grep_formula <- grep("ExtractT\\(A0x.*i,1,8\\)", obj$formula)
    expect_equal(bool_grep_formula, 1)
    
    obj <- extractT(Pm_v, 0, d)
    expect_true(is.LazyTensor(obj))
    bool_grep_formula <- grep("ExtractT\\(A0x.*NA,0,8\\)", obj$formula)
    expect_equal(bool_grep_formula, 1)
    
    obj <- extractT(Pm_int, 0, d)
    expect_true(is.LazyTensor(obj))
    bool_grep_formula <- grep("ExtractT\\(IntCst\\(2\\),0,8\\)", obj$formula)
    expect_equal(bool_grep_formula, 1)
    expect_null <- obj$args
    expect_null <- obj$data
    expect_equal(obj$dimres, 1)
    
    # errors
    expect_error(extractT(x_i, 3.14, d),
                 "`m` input argument should be an integer.",
                 fixed = TRUE)
    expect_error(extractT(x_i, m, 3.14),
                 "`d` input argument should be an integer.",
                 fixed = TRUE)
    expect_error(extractT(x_i, m, -1),
                 paste("Input `d` is out of bounds. Should be at least equal",
                       " to `x` inner dimension, which is 3.",
                       sep = ""),
                 fixed = TRUE)
    expect_error(extractT(x_i, -2, d),
                 "Index `m` is out of bounds. Should be in [0, `d`).",
                 fixed = TRUE)
    expect_error(extractT(x_i, 8, 8),
                 "Index `m` is out of bounds. Should be in [0, `d`).",
                 fixed = TRUE)
    expect_error(extractT(x_i, 7, d),
                 paste("Slice dimension is out of bounds: `d` - `m`",
                       " should be strictly greater than `x`", 
                       " inner dimension, which is 3.",
                       sep = ""),
                 fixed = TRUE)
})


test_that("concat", {
    # toy example
    x <- matrix(1, 1, 3)
    y <- matrix(2, 1, 3)
    x_i <- LazyTensor(x, index = 'i')
    y_j <- LazyTensor(y, index = 'j')
    
    obj <- concat(x_i, y_j)
    expect_true(is.LazyTensor(obj))
    bool_grep_formula <- grep("Concat\\(A0x.*i,A0x.*j\\)", obj$formula)
    expect_equal(bool_grep_formula, 1)
    
    res <- sum(concat(x_i,y_j), "i")
    expected_res <- rep(1:2, each = 3)
    expect_equal(as.vector(res), expected_res)
    
    # basic example
    x <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows, 3 columns
    y <- matrix(runif(250 * 3), 250, 3) # arbitrary R matrix, 250 rows, 3 columns
    x_i <- LazyTensor(x, index = 'i')   # LazyTensor from matrix x, indexed by 'i'
    y_j <- LazyTensor(y, index = 'j')   # LazyTensor from matrix x, indexed by 'j'
    
    int <- 5
    Pm_int <- LazyTensor(int)
    
    # check formulas, args & classes
    obj <- concat(x_i, y_j)
    expect_true(is.LazyTensor(obj))
    bool_grep_formula <- grep("Concat\\(A0x.*i,A0x.*j\\)", obj$formula)
    expect_equal(bool_grep_formula, 1)
    
    obj <- concat(x_i, Pm_int)
    expect_true(is.LazyTensor(obj))
    bool_grep_formula <- grep("Concat\\(A0x.*i,IntCst\\(5\\)\\)", obj$formula)
    expect_equal(bool_grep_formula, 1)
    expect_equal(length(obj$args), 1)
    expect_equal(length(obj$data), 1)
    expect_equal(obj$dimres, 4)
})


test_that("one_hot", {
    # basic example
    x <- matrix(runif(150 * 1), 150, 1) # arbitrary R matrix, 150 rows, 1 column
    y <- matrix(runif(250 * 3), 250, 3) # arbitrary R matrix, 250 rows, 3 columns
    x_i <- LazyTensor(x, index = 'i')   # LazyTensor from matrix x, indexed by 'i'
    y_j <- LazyTensor(y, index = 'j')   # LazyTensor from matrix x, indexed by 'j'
    v <- runif(150, min = 0, max = 3.14)
    LT_v <- LazyTensor(v) # parameter vector
    LT_s <- LazyTensor(13) # parameter scalar
    LT_dec <- LazyTensor(7.14) # parameter decimal scalar
    
    z <- matrix(1i^ (-6:5), nrow = 4)                     # complex 4x3 matrix
    z_i <- LazyTensor(z, index = 'i', is_complex = TRUE)  # ComplexLazyTensor
    
    D <- 7
    # check formulas, args & classes
    obj <- one_hot(LT_s, D)
    expect_true(is.LazyTensor(obj))
    bool_grep_formula <- grep("OneHot\\(IntCst\\(13\\),7\\)", obj$formula)
    expect_equal(bool_grep_formula, 1)
    expect_null <- obj$args
    expect_null <- obj$data
    expect_equal(obj$dimres, 7)
    
    obj <- one_hot(LT_dec, D)
    expect_true(is.LazyTensor(obj))
    bool_grep_formula <- grep("OneHot\\(A0x.*NA,7\\)", obj$formula)
    expect_equal(bool_grep_formula, 1)
    expect_equal(length(obj$args), 1)
    expect_equal(length(obj$data), 1)
    expect_equal(obj$dimres, 7)
    
    # errors
    expect_error(one_hot(x_i, D),
                 "One-hot encoding is only supported for scalar formulas.",
                 fixed = TRUE)
    expect_error(one_hot(y_j, D),
                 "One-hot encoding is only supported for scalar formulas.",
                 fixed = TRUE)
    expect_error(one_hot(LT_v, D),
                 "One-hot encoding is only supported for scalar formulas.",
                 fixed = TRUE)
    expect_error(one_hot(z_i, D),
                 paste("`one_hot` operation can only be applied to `LazyTensor`,",
                       " not `ComplexLazyTensor`", sep = ""),
                 fixed = TRUE)
})



# TEST ELEMENTARY DOT PRODUCT OPERATIONS =======================================


test_that("matvecmult", {
    # basic example
    m <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows, 3 columns
    bad_m <- matrix(c(4, 3), ncol = 1)  # arbitrary R matrix, 2 rows, 1 column
    v <- runif(3, 0, 1)                 # arbitrary R vector of length 3
    bad_v <- runif(150, 0, 1)           # arbitrary R vector of length 150
    m_i <- LazyTensor(m, index = 'i')   # LazyTensor from matrix m, indexed by 'i'
    bad_m_i <- LazyTensor(bad_m, 'i')
    Pm_v <- LazyTensor(v)               # parameter vector LazyTensor from v
    Pm_bad_v <- LazyTensor(bad_v)       # parameter vector LazyTensor from bad_v
    Pm_one <- LazyTensor(c(3.14))       # parameter vector of length 1
    
    # check formulas, args & classes
    obj <- matvecmult(m_i, Pm_v)
    expect_true(is.LazyTensor(obj))
    bool_grep_formula <- grep("MatVecMult\\(A0x.*i,A0x.*NA\\)",
                              obj$formula)
    expect_equal(bool_grep_formula, 1)
    
    obj_Pm_one <- matvecmult(m_i, Pm_one)
    expect_true(is.LazyTensor(obj_Pm_one))
    bool_grep_formula <- grep("MatVecMult\\(A0x.*i,A0x.*NA\\)",
                              obj_Pm_one$formula)
    expect_equal(bool_grep_formula, 1)
    
    # check dimres
    expect_equal(obj$dimres, 1)
    expect_equal(obj_Pm_one$dimres, 3)
    
    # errors
    expect_error(matvecmult(m_i, Pm_bad_v),
                 paste("`m` and `v` should have the same inner dimension or",
                       " `v` should be of dimension 1.", sep = ""),
                 fixed = TRUE)
    expect_error(matvecmult(Pm_v, Pm_v),
                 paste("`m` input argument should be a `LazyTensor` encoding", 
                       " a matrix defined with `Vi()` or `Vj()`.", sep = ""),
                 fixed = TRUE)
    expect_error(matvecmult(m_i, m_i),
                 paste("`v` input argument should be a `LazyTensor` encoding", 
                       " a vector defined with `Pm()`.", sep = ""),
                 fixed = TRUE)
    expect_error(matvecmult(bad_m_i, Pm_v),
                 paste("`m` and `v` should have the same inner dimension or",
                       " `v` should be of dimension 1.", sep = ""),
                 fixed = TRUE)
    
})


test_that("vecmatmult", {
    # basic example
    m <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows, 3 columns
    bad_m <- matrix(c(4, 3), ncol = 1)  # arbitrary R matrix, 2 rows, 1 column
    v <- runif(3, 0, 1)                 # arbitrary R vector of length 3
    bad_v <- runif(150, 0, 1)           # arbitrary R vector of length 150
    m_i <- LazyTensor(m, index = 'i')   # LazyTensor from matrix m, indexed by 'i'
    bad_m_i <- LazyTensor(bad_m, 'i')
    Pm_v <- LazyTensor(v)               # parameter vector LazyTensor from v
    Pm_bad_v <- LazyTensor(bad_v)       # parameter vector LazyTensor from bad_v
    Pm_one <- LazyTensor(c(3.14))       # parameter vector of length 1
    
    # check formulas, args & classes
    obj <- vecmatmult(Pm_v, m_i)
    expect_true(is.LazyTensor(obj))
    bool_grep_formula <- grep("VecMatMult\\(A0x.*NA,A0x.*i\\)",
                              obj$formula)
    expect_equal(bool_grep_formula, 1)
    
    obj_Pm_one <- vecmatmult(Pm_one, m_i)
    expect_true(is.LazyTensor(obj_Pm_one))
    bool_grep_formula <- grep("VecMatMult\\(A0x.*NA,A0x.*i\\)",
                              obj_Pm_one$formula)
    expect_equal(bool_grep_formula, 1)
    
    # check dimres
    expect_equal(obj$dimres, 1)
    expect_equal(obj_Pm_one$dimres, 3)
    
    # errors
    expect_error(vecmatmult(Pm_bad_v, m_i),
                 paste("`v` and `m` should have the same inner dimension or",
                       " `v` should be of dimension 1.", sep = ""),
                 fixed = TRUE)
    expect_error(vecmatmult(m_i, m_i),
                 paste("`v` input argument should be a `LazyTensor` encoding", 
                       " a vector defined with `Pm()`.", sep = ""),
                 fixed = TRUE)
    
    expect_error(vecmatmult(Pm_v, Pm_v),
                 paste("`m` input argument should be a `LazyTensor` encoding", 
                       " a matrix defined with `Vi()` or `Vj()`.", sep = ""),
                 fixed = TRUE)
    expect_error(vecmatmult(Pm_v, bad_m_i),
                 paste("`v` and `m` should have the same inner dimension or",
                       " `v` should be of dimension 1.", sep = ""),
                 fixed = TRUE)
    
})


test_that("tensorprod", {
    x <- matrix(c(1, 2, 3), 2, 3)
    x_i <- LazyTensor(x, index = 'i')  
    xc_i <- LazyTensor(x, index = 'i', is_complex = TRUE)  
    y <- matrix(c(1, 1, 1), 2, 3)
    y_i <- LazyTensor(y, index = 'i')
    yc_i <- LazyTensor(y, index = 'i', is_complex = TRUE) 
    
    expect_true(is.ComplexLazyTensor(tensorprod(xc_i, yc_i)))
    expect_true(is.ComplexLazyTensor(tensorprod(xc_i, y_i)))
    expect_true(is.ComplexLazyTensor(tensorprod(xc_i, y_i)))
    
    obj <- tensorprod(x_i, y_i)    # symbolic (4, 9) matrix. 
    expect_true(is.LazyTensor(obj))
    expect_false(is.ComplexLazyTensor(obj))
    expect_equal(obj$dimres, 9)
    bool_grep_formula <- grep("TensorProd\\(A0x.*i,A0x.*i\\)", obj$formula)
    expect_equal(bool_grep_formula, 1)
})


# TEST REDUCTIONS ==============================================================


test_that("reduction.LazyTensor", {
    D <- 3
    M <- 100
    N <- 150
    x <- matrix(runif(M * D), M, D)
    x_i <- LazyTensor(x, index = 'i')
    opstr <- "Sum"
    
    res <- reduction.LazyTensor(x_i, opstr, "i")
    
    # check results, formulas & classes
    expect_false(is.LazyTensor(res))
    expect_true(is.matrix(res))
    
    # with an optional argument
    opstr <- "KMin"
    K <- 2
    res <- reduction.LazyTensor(x_i, opstr, "i", opt_arg = K)
    checkmate::expect_array(res, d = 3)
    
    # errors
    expect_error(reduction.LazyTensor(3, opstr, "i"),
                 "`x` input should be a LazyTensor or a ComplexLazyTensor.", 
                 fixed=TRUE)
    
    expect_error(reduction.LazyTensor(x, opstr, "i"),
                 "`x` input should be a LazyTensor or a ComplexLazyTensor.", 
                 fixed=TRUE)
    
    expect_error(
        reduction.LazyTensor(x_i, opstr, "b"),
        "`index` input argument should be a character, either 'i' or 'j'.", 
        fixed=TRUE
    )
    
    expect_error(reduction.LazyTensor(x_i, 2, "i"),
                 "`opstr` input should be a string text.", 
                 fixed=TRUE)
    
})


test_that("sum", {
    
    # check that base operation is still working
    expect_equal(sum(1), 1)
    expect_equal(sum(1:10), 10*11/2)
    expect_equal(sum(c(1:10, NA)), as.integer(NA))
    expect_equal(sum(c(1:10, NA), na.rm = TRUE), 10*11/2)
    expect_equal(sum(matrix(1:10, 2, 5)), 10*11/2)
    
    # basic example
    x <- matrix(c(1, 2, 3), 2, 3)
    x_i <- LazyTensor(x, index = 'i')
    x_j <- LazyTensor(x, index = 'j')
    
    z <- matrix((1+2i)^(-6:5), nrow = 4)
    z_i <- LazyTensor(z, index = 'i', is_complex = TRUE)
    
    v <- c(1, 2, 3)
    Pm_v <- Pm(v)
    
    complex_vect <- c(1 + 2i, 1 + 2i, 1 + 2i)
    Pm_complex <- Pm(complex_vect)
    
    # check classes and results
    res <- sum(x_i, index = "i")
    expect_false(is.LazyTensor(res))
    expect_equal(dim(res), c(1, 3))
    expect_true(is.matrix(res))
    expected_res <- c(sum(x[, 1]), sum(x[, 2]), sum(x[, 3]))
    expect_true(sum(abs(res - expected_res)) < 1E-5)
    
    res <- sum(x_i, "j")
    expect_false(is.LazyTensor(res))
    expect_true(is.matrix(res))
    expect_equal(dim(res), c(2, 3))
    expected_res <- x
    expect_true(sum(abs(res - expected_res)) < 1E-5)
    
    res <- sum(x_j, "j")
    expect_false(is.LazyTensor(res))
    expect_equal(dim(res), c(1, 3))
    expect_true(is.matrix(res))
    expected_res <- c(sum(x[, 1]), sum(x[, 2]), sum(x[, 3]))
    expect_true(sum(abs(res - expected_res)) < 1E-5)
    
    res <- sum(x_j, "i")
    expect_false(is.LazyTensor(res))
    expect_true(is.matrix(res))
    expect_equal(dim(res), c(2, 3))
    expected_res <- x
    expect_true(sum(abs(res - expected_res)) < 1E-5)
    
    res <- sum(z_i, "i")
    expect_false(is.LazyTensor(res))
    expect_true(is.matrix(res))
    expect_equal(dim(res), c(1, 6))
    z_var <- z_i$data[[1]]
    expected_res <- c(sum(z_var[, 1]), sum(z_var[, 2]), sum(z_var[, 3]), 
                      sum(z_var[, 4]), sum(z_var[, 5]), sum(z_var[, 6]))
    expect_true(sum(abs(res - expected_res)) < 1E-5)
    
    res <- sum(Pm_v)
    expect_true(is.LazyTensor(res))
    
    res <- sum(Pm_complex)
    expect_true(is.ComplexLazyTensor(res))
    
    
    # more complex examples
    x <- matrix(runif(150 * 3), 150, 3)
    y <- matrix(runif(150 * 3), 150, 3)
    x_i <- LazyTensor(x, index = 'i')
    y_j <- LazyTensor(y, index = 'j')
    K <- sum(x_i - y_j)
    res <- sum(K, index = "i")
    expected_res <- apply(
        t(sapply(
            1:nrow(x), function(id_x) 
                sapply(1:nrow(y), function(id_y) sum(x[id_x,] - y[id_y,]))
        )), 2, sum)
    expect_equal(as.vector(res), expected_res, tolerance = 1E-5)
    
    # check formulae
    bool_grep_formula <- grep("Sum\\(A0x.*NA\\)", sum(Pm_v)$formula)
    expect_equal(bool_grep_formula, 1)
    
    bool_grep_formula <- grep("ComplexSum\\(A0x.*NA\\)", sum(Pm_complex)$formula)
    expect_equal(bool_grep_formula, 1)
    
    # errors
    expect_error(sum(x_i, "b"),
                 paste("`index` input argument should be a character,",
                       " either 'i' or 'j', or NA.", sep = ""), 
                 fixed = TRUE)
    
    expect_error(sum(x_i, 2),
                 paste("`index` input argument should be a character,",
                       " either 'i' or 'j', or NA.", sep = ""), 
                 fixed = TRUE)
})


test_that("sum_reduction", {
    D <- 3
    M <- 100
    N <- 150
    x <- matrix(runif(M * D), M, D)
    x_i <- LazyTensor(x, index = 'i')
    
    res <- sum_reduction(x_i, "i")
    expect_false(is.LazyTensor(res))
    expect_true(is.matrix(res))
    
    expect_equal(as.vector(res), apply(x, 2, sum), tolerance = 1e-5)
    
    # errors
    expect_error(sum_reduction(x_i, "b"),
                 paste("`index` input argument should be a character,",
                       " either 'i' or 'j', or NA.", sep = ""), 
                 fixed = TRUE)
    
    expect_error(sum_reduction(x_i, 2),
                 paste("`index` input argument should be a character,",
                       " either 'i' or 'j', or NA.", sep = ""), 
                 fixed = TRUE)
    
    expect_error(sum_reduction(x, "i"),
                 "`x` input should be a LazyTensor or a ComplexLazyTensor.", 
                 fixed = TRUE)
    
    expect_error(sum_reduction(3, "i"),
                 "`x` input should be a LazyTensor or a ComplexLazyTensor.", 
                 fixed = TRUE)
    
})


test_that("min", {
    
    # check that base operation is still working
    expect_equal(min(1), 1)
    expect_equal(min(1:10), 1)
    expect_equal(min(c(1:10, NA)), as.integer(NA))
    expect_equal(min(c(1:10, NA), na.rm = TRUE), 1)
    expect_equal(min(matrix(1:10, 2, 5)), 1)
    
    # basic example
    x <- matrix(c(1, 2, 3), 2, 3)
    y <- matrix(c(3, 4, 5), 2, 3)
    x_i <- LazyTensor(x, index = 'i')
    y_j <- LazyTensor(y, index = 'j')
    
    z <- matrix((1+2i)^(-6:5), nrow = 4)
    z_i <- LazyTensor(z, index = 'i', is_complex = TRUE)
    
    v <- c(1, 2, 3)
    Pm_v <- Pm(v)
    
    # check results, formulas & classes
    expect_equal(min(3), 3)
    expect_false(is.LazyTensor(min(x)))
    expect_true(is.LazyTensor(min(x_i)))
    expect_true(is.ComplexLazyTensor(min(z_i)))
    expect_true(is.LazyTensor(min(Pm_v)))
    expect_true(is.ComplexLazyTensor(min(z_i)))
    
    # reduction
    res <- min(x_i, "i")
    expect_false(is.LazyTensor(res))
    expect_equal(dim(res), c(1, 3))
    expect_true(is.matrix(res))
    expected_res <- c(min(x[, 1]), min(x[, 2]), min(x[, 3]))
    expect_true(sum(abs(res - expected_res)) < 1E-5)
    
    res <- min(x_i, "j")
    expect_false(is.LazyTensor(res))
    expect_equal(dim(res), c(2, 3))
    expect_true(is.matrix(res))
    expected_res <- x
    expect_true(sum(abs(res - expected_res)) < 1E-5)
    
    res <- min(Pm_v, "j")
    expect_false(is.LazyTensor(res))
    expect_equal(dim(res), c(1, 3))
    expect_true(is.matrix(res))
    expected_res <- v
    expect_true(sum(abs(res - expected_res)) < 1E-5)
    
    # checks when there is no reduction
    obj <- min(x_i)
    expect_true(is.LazyTensor(obj))
    bool_grep_formula <- grep("Min\\(A0x.*i\\)", obj$formula)
    expect_equal(bool_grep_formula, 1)
    
    # errors
    expect_error(min(x_i, "b"),
                 paste("`index` input argument should be a character,",
                       " either 'i' or 'j', or NA.", sep = ""),
                 fixed = TRUE)
    
    expect_error(min(x_i, 4),
                 paste("`index` input argument should be a character,",
                       " either 'i' or 'j', or NA.", sep = ""),
                 fixed = TRUE)
    
})


test_that("min_reduction", {
    x <- matrix(c(1, 2, 3), 2, 3)
    x_i <- LazyTensor(x, index = 'i')
    
    res <- min_reduction(x_i, "i")
    # check class
    expect_false(is.LazyTensor(res))
    expect_true(is.matrix(res))
    
    # errors
    expect_error(min_reduction(3, "i"),
                 "`x` input should be a LazyTensor or a ComplexLazyTensor.",
                 fixed = TRUE)
    
    expect_error(min_reduction(x, "i"),
                 "`x` input should be a LazyTensor or a ComplexLazyTensor.",
                 fixed = TRUE)
    
    expect_error(
        min_reduction(x_i, "b"),
        "`index` input argument should be a character, either 'i' or 'j'.",
        fixed = TRUE
    )
    
    expect_error(
        min_reduction(x_i, 3),
        "`index` input argument should be a character, either 'i' or 'j'.",
        fixed = TRUE
    )
    
})


test_that("argmin", {
    x <- matrix(c(1, 2, 3), 2, 3)
    x_i <- LazyTensor(x, index = 'i')
    xc_i <- LazyTensor(x, index = "i", is_complex = TRUE)
    
    
    # check results, formulas & classes
    expect_true(is.LazyTensor(argmin(3)))
    
    res <- argmin(x_i, "i")
    expect_false(is.LazyTensor(res))
    expect_true(is.matrix(res))
    expected_res <- c(which.min(x[, 1]) - 1, # subtract 1 because
                      which.min(x[, 2]) - 1, # indices start at zero
                      which.min(x[, 3]) - 1) # in KeOps...
    expect_true(sum(abs(res - expected_res)) < 1E-5)
    
    obj <- argmin(x_i)
    expect_true(is.LazyTensor(obj))
    bool_grep_formula <- grep("ArgMin\\(A0x.*i\\)", obj$formula)
    expect_equal(bool_grep_formula, 1)
    
    # errors
    expect_error(argmin(3, "i"),
                 "`x` input should be a LazyTensor or a ComplexLazyTensor.",
                 fixed = TRUE)
    
    expect_error(argmin(x, "i"),
                 "`x` input should be a LazyTensor or a ComplexLazyTensor.",
                 fixed = TRUE)
    
    expect_error(argmin(x_i, "b"),
                 paste("`index` input argument should be a character,",
                       " either 'i' or 'j', or NA.", sep = ""),
                 fixed = TRUE)
    
    expect_error(argmin(x_i, 3),
                 paste("`index` input argument should be a character,",
                       " either 'i' or 'j', or NA.", sep = ""),
                 fixed = TRUE)
})


test_that("argmin_reduction", {
    x <- matrix(c(1, 2, 3), 2, 3)
    x_i <- LazyTensor(x, index = 'i')
    
    res <- argmin_reduction(x_i, "i")
    # check results, formulas & classes
    expect_false(is.LazyTensor(res))
    expect_true(is.matrix(res))
    
    # errors
    expect_error(argmin_reduction(3, "i"),
                 "`x` input should be a LazyTensor or a ComplexLazyTensor.",
                 fixed = TRUE)
    
    expect_error(
        argmin_reduction(x_i, 3),
        "`index` input argument should be a character, either 'i' or 'j'.",
        fixed = TRUE
    )
})


test_that("min_argmin", {
    x <- matrix(runif(12), 4, 3)
    x_i <- LazyTensor(x, index = 'i')
    
    res <- min_argmin(x_i, "i")
    expect_false(is.LazyTensor(res))
    checkmate::expect_list(res, len = 2)
    expect_equal(dim(res[[1]]), c(1, 3))
    expect_equal(dim(res[[2]]), c(1, 3))
    expected_res <- list(
        apply(x, 2, min),
        apply(x, 2, which.min) - 1 # subtract 1 because index start at 0 in KeOps
    )
    expect_equal(lapply(res, as.vector), expected_res, tolerance = 1e-5)
    
    res <- min_argmin(x_i, "j")
    expect_false(is.LazyTensor(res))
    checkmate::expect_list(res, len = 2)
    expect_equal(dim(res[[1]]), c(4, 3))
    expect_equal(dim(res[[2]]), c(4, 3))
    expected_res <- list(x, matrix(0, 4, 3))
    expect_equal(res, expected_res, tolerance = 1e-5)
    
    # errors
    expect_error(min_argmin(3, "i"),
                 "`x` input should be a LazyTensor or a ComplexLazyTensor.",
                 fixed = TRUE)
    
    expect_error(
        min_argmin(x_i, "b"),
        "`index` input argument should be a character, either 'i' or 'j'.",
        fixed = TRUE)
})



test_that("min_argmin_reduction", {
    x <- matrix(c(1, 2, 3), 2, 3)
    x_i <- LazyTensor(x, index = 'i')
    
    res <- min_argmin_reduction(x_i, "i")
    
    # check results, formulas & classes
    expect_false(is.LazyTensor(res))
    checkmate::expect_list(res, len = 2)
    expect_equal(dim(res[[1]]), c(1, 3))
    expect_equal(dim(res[[2]]), c(1, 3))
    expected_res <- list(
        apply(x, 2, min),
        apply(x, 2, which.min) - 1 # subtract 1 because index start at 0 in KeOps
    )
    expect_equal(lapply(res, as.vector), expected_res, tolerance = 1e-5)
    
    # errors
    expect_error(min_argmin_reduction(3, "i"),
                 "`x` input should be a LazyTensor or a ComplexLazyTensor.",
                 fixed = TRUE)
    
    expect_error(
        min_argmin_reduction(x_i, "b"),
        "`index` input argument should be a character, either 'i' or 'j'.",
        fixed = TRUE
    )
})


test_that("max", {
    
    # check that base operation is still working
    expect_equal(max(1), 1)
    expect_equal(max(1:10), 10)
    expect_equal(max(c(1:10, NA)), as.integer(NA))
    expect_equal(max(c(1:10, NA), na.rm = TRUE), 10)
    expect_equal(max(matrix(1:10, 2, 5)), 10)
    
    # basic example
    x <- matrix(c(1, 2, 3), 2, 3)
    y <- matrix(c(3, 4, 5), 2, 3)
    x_i <- LazyTensor(x, index = 'i')
    y_j <- LazyTensor(y, index = 'j')
    
    z <- matrix((1+2i)^(-6:5), nrow = 4)
    z_i <- LazyTensor(z, index = 'i', is_complex = TRUE)
    
    v <- c(1, 2, 3)
    Pm_v <- Pm(v)
    
    # check results, formulas & classes
    expect_equal(max(1, 2, 3), 3)
    expect_false(is.LazyTensor(max(x)))
    expect_true(is.LazyTensor(max(x_i)))
    expect_true(is.ComplexLazyTensor(max(z_i)))
    expect_true(is.LazyTensor(max(Pm_v)))
    expect_true(is.ComplexLazyTensor(max(z_i)))
    
    # reduction
    res <- max(x_i, "i")
    expect_false(is.LazyTensor(res))
    expect_equal(dim(res), c(1, 3))
    expect_true(is.matrix(res))
    expected_res <- c(max(x[, 1]), max(x[, 2]), max(x[, 3]))
    expect_true(sum(abs(res - expected_res)) < 1E-5)
    
    res <- max(x_i, "j")
    expect_false(is.LazyTensor(res))
    expect_equal(dim(res), c(2, 3))
    expect_true(is.matrix(res))
    expected_res <- x
    expect_true(sum(abs(res - expected_res)) < 1E-5)
    
    res <- max(Pm_v, "j")
    expect_false(is.LazyTensor(res))
    expect_equal(dim(res), c(1, 3))
    expect_true(is.matrix(res))
    expected_res <- v
    expect_true(sum(abs(res - expected_res)) < 1E-5)
    
    # when there is no reduction
    obj <- max(x_i)
    expect_true(is.LazyTensor(obj))
    bool_grep_formula <- grep("Max\\(A0x.*i\\)", obj$formula)
    expect_equal(bool_grep_formula, 1)
    
    # errors
    expect_error(max(x_i, "b"),
                 paste("`index` input argument should be a character,",
                       " either 'i' or 'j', or NA.", sep = ""),
                 fixed = TRUE)
    
    expect_error(max(x_i, 4),
                 paste("`index` input argument should be a character,",
                       " either 'i' or 'j', or NA.", sep = ""),
                 fixed = TRUE)
})


test_that("max_reduction", {
    x <- matrix(c(1, 2, 3), 2, 3)
    x_i <- LazyTensor(x, index = 'i')
    
    res <- max_reduction(x_i, "i")
    # check results, formulas & classes
    expect_false(is.LazyTensor(res))
    expect_true(is.matrix(res))
    
    # errors
    expect_error(
        max_reduction(x_i, "b"),
        "`index` input argument should be a character, either 'i' or 'j'.",
        fixed = TRUE
    )
    
    expect_error(max_reduction(x, "i"),
                 "`x` input should be a LazyTensor or a ComplexLazyTensor.",
                 fixed = TRUE)
})


test_that("argmax", {
    x <- matrix(c(1, 2, 3), 2, 3)
    x_i <- LazyTensor(x, index = 'i')
    
    # check results, formulas & classes
    res <- argmax(x_i, "i")
    expect_false(is.LazyTensor(res))
    expect_true(is.matrix(res))
    expected_res <- c(which.max(x[, 1]) - 1, # add 1 because
                      which.max(x[, 2]) - 1, # indices start at zero
                      which.max(x[, 3]) - 1) # in KeOps...
    expect_true(sum(abs(res - expected_res)) < 1E-5)
    
    expect_true(is.LazyTensor(argmax(3)))
    expect_false(is.LazyTensor(res))
    expect_true(is.matrix(res))
    
    obj <- argmax(x_i)
    expect_true(is.LazyTensor(obj))
    bool_grep_formula <- grep("ArgMax\\(A0x.*i\\)", obj$formula)
    expect_equal(bool_grep_formula, 1)
    
    # errors
    expect_error(argmax(3, "i"),
                 "`x` input should be a LazyTensor or a ComplexLazyTensor.",
                 fixed = TRUE)
    
    expect_error(argmax(x, "i"),
                 "`x` input should be a LazyTensor or a ComplexLazyTensor.",
                 fixed = TRUE)
    
    expect_error(argmax(x_i, "b"),
                 paste("`index` input argument should be a character,",
                       " either 'i' or 'j', or NA.", sep = ""),
                 fixed = TRUE)
    
    expect_error(argmax(x_i, 3),
                 paste("`index` input argument should be a character,",
                       " either 'i' or 'j', or NA.", sep = ""),
                 fixed = TRUE)
})


test_that("argmax_reduction", {
    x <- matrix(c(1, 2, 3), 2, 3)
    x_i <- LazyTensor(x, index = 'i')
    
    res <- argmax_reduction(x_i, "i")
    # check results, formulas & classes
    expect_false(is.LazyTensor(res))
    expect_true(is.matrix(res))
    
    # errors
    expect_error(argmax_reduction(3, "i"),
                 "`x` input should be a LazyTensor or a ComplexLazyTensor.",
                 fixed = TRUE)
    
    expect_error(
        argmax_reduction(x_i, 3),
        "`index` input argument should be a character, either 'i' or 'j'.",
        fixed = TRUE
    )
})


test_that("max_argmax", {
    x <- matrix(runif(12), 4, 3)
    x_i <- LazyTensor(x, index = 'i')
    
    res <- max_argmax(x_i, "i")
    expect_false(is.LazyTensor(res))
    checkmate::expect_list(res, len = 2)
    expect_equal(dim(res[[1]]), c(1, 3))
    expect_equal(dim(res[[2]]), c(1, 3))
    expected_res <- list(
        apply(x, 2, max),
        apply(x, 2, which.max) - 1 # subtract 1 because index start at 0 in KeOps
    )
    expect_equal(lapply(res, as.vector), expected_res, tolerance = 1e-5)
    
    res <- max_argmax(x_i, "j")
    expect_false(is.LazyTensor(res))
    checkmate::expect_list(res, len = 2)
    expect_equal(dim(res[[1]]), c(4, 3))
    expect_equal(dim(res[[2]]), c(4, 3))
    expected_res <- list(x, matrix(0, 4, 3))
    expect_equal(res, expected_res, tolerance = 1e-5)
    
    # errors
    expect_error(max_argmax(3, "i"),
                 "`x` input should be a LazyTensor or a ComplexLazyTensor.",
                 fixed = TRUE)
    
    expect_error(
        max_argmax(x_i, "b"),
        "`index` input argument should be a character, either 'i' or 'j'.",
        fixed = TRUE
    )
})



test_that("max_argmax_reduction", {
    x <- matrix(runif(12), 4, 3)
    x_i <- LazyTensor(x, index = 'i')
    
    res <- max_argmax_reduction(x_i, "i")
    # check results, formulas & classes
    expect_false(is.LazyTensor(res))
    checkmate::expect_list(res, len = 2)
    expect_equal(dim(res[[1]]), c(1, 3))
    expect_equal(dim(res[[2]]), c(1, 3))
    expected_res <- list(
        apply(x, 2, max),
        apply(x, 2, which.max) - 1 # subtract 1 because index start at 0 in KeOps
    )
    expect_equal(lapply(res, as.vector), expected_res, tolerance = 1e-5)
    
    # errors
    expect_error(
        max_argmax_reduction(3, "i"),
        "`x` input should be a LazyTensor or a ComplexLazyTensor.",
        fixed = TRUE
    )
    
    expect_error(
        max_argmax_reduction(x_i, "b"),
        "`index` input argument should be a character, either 'i' or 'j'.",
        fixed = TRUE
    )
    
})


test_that("Kmin", {
    
    x <- matrix(c(1, 2, 3), 2, 3, byrow = TRUE)
    x_i <- LazyTensor(x, index = 'i')
    
    res <- Kmin(x_i, 2, "i")
    expect_false(is.LazyTensor(res))
    checkmate::expect_array(res, d = 3)
    expect_equal(dim(res), c(1, 2, 3))
    expected_res <- array(x, dim = c(1,2,3))
    expect_equal(res, expected_res, tolerance = 1e-5)
    
    res <- Kmin(x_i, 2, "j")
    expect_false(is.LazyTensor(res))
    checkmate::expect_array(res, d = 3)
    expect_equal(dim(res), c(2, 2, 3))
    expected_res <- array(0, dim = c(2,2,3))
    expected_res[1,,] <- matrix(c(1:3, rep(Inf, 3)), 2, 3, byrow = TRUE)
    expected_res[2,,] <- matrix(c(1:3, rep(Inf, 3)), 2, 3, byrow = TRUE)
    expect_equal(res, expected_res, tolerance = 1e-5)
    
    # basic example
    w <- matrix(c(2, 4, 6, 3, 2, 8, 9, 1, 3), 3, 3)
    w_i <- LazyTensor(w, index = 'i')
    x <- matrix(c(1, 2, 3), 2, 3)
    x_i <- LazyTensor(x, index = 'i')
    y <- matrix(c(5, 1, 3), 2, 3)
    y_j <- LazyTensor(y, index = 'j')
    
    K <- 2
    
    # check formulas, args & classes
    S_ij <- sum((x_i - y_j)^2)
    res <- Kmin(S_ij, K, "i")
    expect_false(is.LazyTensor(res))
    checkmate::expect_matrix(res)
    
    # check very simple results
    res <- Kmin(x_i, 1, 'j')
    expect_equal(res, array(x, dim = c(2, 1, 3)), tolerance = 1e-5)
    # ---
    res <- Kmin(x_i, 1, 'i')
    expect_equal(
        res, array(c(min(x[, 1]), min(x[, 2]), min(x[, 3])), dim = c(1, 1, 3)))
    # ---
    res <- Kmin(w_i, K, 'i')
    expect_false(is.LazyTensor(res))
    checkmate::expect_array(res, d = 3)
    expect_equal(dim(res), c(1, 2, 3))
    K_1 <- c(min(w[, 1]),
             min(w[, 2]),
             min(w[, 3]))
    K_2 <- c(min(w[, 1][-which.min(w[, 1])]),
             min(w[, 2][-which.min(w[, 2])]),
             min(w[, 3][-which.min(w[, 3])]))
    expected_res <- array(
        matrix(c(K_1, K_2), 2, 3, byrow = TRUE), dim = c(1, 2, 3)
    )
    expect_equal(res, expected_res, tolerance = 1e-5)
    
    # errors
    expect_error(
        Kmin(x_i, 3.14, "i"),
        "`K` input argument should be an integer.",
        fixed = TRUE
    )
    expect_error(
        Kmin(x_i, K, "k"),
        "`index` input argument should be a character, either 'i' or 'j'.",
        fixed = TRUE
    )
    
})


test_that("Kmin_reduction", {
    
    x <- matrix(c(1, 2, 3), 2, 3, byrow = TRUE)
    x_i <- LazyTensor(x, index = 'i')
    
    res <- Kmin_reduction(x_i, 2, "i")
    expect_false(is.LazyTensor(res))
    checkmate::expect_array(res, d = 3)
    expect_equal(dim(res), c(1, 2, 3))
    expected_res <- array(x, dim = c(1,2,3))
    expect_equal(res, expected_res, tolerance = 1e-5)
    
    res <- Kmin_reduction(x_i, 2, "j")
    expect_false(is.LazyTensor(res))
    checkmate::expect_array(res, d = 3)
    expect_equal(dim(res), c(2, 2, 3))
    expected_res <- array(0, dim = c(2,2,3))
    expected_res[1,,] <- matrix(c(1:3, rep(Inf, 3)), 2, 3, byrow = TRUE)
    expected_res[2,,] <- matrix(c(1:3, rep(Inf, 3)), 2, 3, byrow = TRUE)
    expect_equal(res, expected_res, tolerance = 1e-5)
    
    
    x <- matrix(runif(150 * 3), 150, 3) 
    x_i <- LazyTensor(x, index = 'i') 
    y <- matrix(runif(100 * 3), 100, 3)
    y_j <- LazyTensor(y, index = 'j')
    
    K <- 2
    
    # check formulas, args & classes
    S_ij <- sum((x_i - y_j)^2)
    res <- Kmin_reduction(S_ij, K, "i")
    expect_false(is.LazyTensor(res))
    expect_true(is.matrix(res))
    
    # errors
    expect_error(
        Kmin_reduction(x_i, 3.14, "i"),
        "`K` input argument should be an integer.",
        fixed = TRUE
    )
    expect_error(
        Kmin_reduction(x_i, K, "k"),
        "`index` input argument should be a character, either 'i' or 'j'.",
        fixed = TRUE
    )
    
})


test_that("argKmin", {
    
    x <- matrix(c(1, 2, 3), 2, 3, byrow = TRUE)
    x_i <- LazyTensor(x, index = 'i')
    
    res <- argKmin(x_i, 2, "i")
    expect_false(is.LazyTensor(res))
    checkmate::expect_array(res, d = 3)
    expect_equal(dim(res), c(1, 2, 3))
    expected_res <- array(matrix(c(0,1), 2, 3), dim = c(1,2,3))
    expect_equal(res, expected_res, tolerance = 1e-5)
    
    res <- argKmin(x_i, 2, "j")
    expect_false(is.LazyTensor(res))
    checkmate::expect_array(res, d = 3)
    expect_equal(dim(res), c(2, 2, 3))
    expected_res <- array(0, dim = c(2,2,3))
    expect_equal(res, expected_res, tolerance = 1e-5)
    
    
    w <- matrix(c(2, 4, 6, 3, 2, 8, 9, 1, 3), 3, 3)
    w_i <- LazyTensor(w, index = 'i')
    x <- matrix(runif(150 * 3), 150, 3) 
    x_i <- LazyTensor(x, index = 'i') 
    y <- matrix(runif(100 * 3), 100, 3)
    y_j <- LazyTensor(y, index = 'j')
    
    K <- 2
    
    # check formulas, args & classes
    S_ij <- sum((x_i - y_j)^2)
    res <- argKmin(S_ij, K, "i")
    expect_false(is.LazyTensor(res))
    expect_true(is.matrix(res))
    
    # check result for simple example
    res <- argKmin(w_i, K, 'i')
    # grab index of min values
    arg_K_1 <- c(which.min(w[, 1]) - 1,
                 which.min(w[, 2]) - 1,
                 which.min(w[, 3]) - 1)
    # replace min values by huge ones
    w[, 1][which.min(w[, 1])] <- 100
    w[, 2][which.min(w[, 2])] <- 100
    w[, 3][which.min(w[, 3])] <- 100
    # grab index of (second) min values
    arg_K_2 <- c(which.min(w[, 1]) - 1,
                 which.min(w[, 2]) - 1,
                 which.min(w[, 3]) - 1)
    
    expected_res <- array(
        matrix(c(arg_K_1, arg_K_2), 2, 3, byrow = TRUE), 
        dim = c(1, 2, 3)
    )
    expect_equal(res, expected_res, tolerance = 1e-5)
    
    # errors
    expect_error(
        argKmin(x_i, 3.14, "i"),
        "`K` input argument should be an integer.",
        fixed = TRUE
    )
    expect_error(
        argKmin(x_i, K, "k"),
        "`index` input argument should be a character, either 'i' or 'j'.",
        fixed = TRUE
    )
    
})


test_that("argKmin_reduction", {
    
    x <- matrix(c(1, 2, 3), 2, 3, byrow = TRUE)
    x_i <- LazyTensor(x, index = 'i')
    
    res <- argKmin_reduction(x_i, 2, "i")
    expect_false(is.LazyTensor(res))
    checkmate::expect_array(res, d = 3)
    expect_equal(dim(res), c(1, 2, 3))
    expected_res <- array(matrix(c(0,1), 2, 3), dim = c(1,2,3))
    expect_equal(res, expected_res, tolerance = 1e-5)
    
    res <- argKmin_reduction(x_i, 2, "j")
    expect_false(is.LazyTensor(res))
    checkmate::expect_array(res, d = 3)
    expect_equal(dim(res), c(2, 2, 3))
    expected_res <- array(0, dim = c(2,2,3))
    expect_equal(res, expected_res, tolerance = 1e-5)
    
    
    
    x <- matrix(runif(150 * 3), 150, 3) 
    x_i <- LazyTensor(x, index = 'i') 
    y <- matrix(runif(100 * 3), 100, 3)
    y_j <- LazyTensor(y, index = 'j')
    
    K <- 2
    
    # check formulas, args & classes
    S_ij <- sum((x_i - y_j)^2)
    res <- argKmin_reduction(S_ij, K, "i")
    expect_false(is.LazyTensor(res))
    expect_true(is.matrix(res))
    
    # errors
    expect_error(
        argKmin_reduction(x_i, 3.14, "i"),
        "`K` input argument should be an integer.",
        fixed = TRUE
    )
    expect_error(
        argKmin_reduction(x_i, K, "k"),
        "`index` input argument should be a character, either 'i' or 'j'.",
        fixed = TRUE
    )
    
})


test_that("Kmin_argKmin", {
    
    x <- matrix(c(1, 2, 3), 2, 3, byrow = TRUE)
    x_i <- LazyTensor(x, index = 'i')
    
    res <- Kmin_argKmin(x_i, 2, "i")
    expect_false(is.LazyTensor(res))
    checkmate::expect_list(res, len = 2)
    expect_equal(dim(res[[1]]), c(1, 2, 3))
    expect_equal(dim(res[[2]]), c(1, 2, 3))
    expected_res <- list(
        array(x, dim = c(1,2,3)),
        array(matrix(c(0,1), 2, 3), dim = c(1,2,3))
    )
    expect_equal(res, expected_res, tolerance = 1e-5)
    
    res <- Kmin_argKmin(x_i, 2, "j")
    expect_false(is.LazyTensor(res))
    checkmate::expect_list(res, len = 2)
    expect_equal(dim(res[[1]]), c(2, 2, 3))
    expect_equal(dim(res[[2]]), c(2, 2, 3))
    expected_res <- list(
        array(0, dim = c(2,2,3)),
        array(matrix(0, 2, 3), dim = c(2,2,3))
    )
    expected_res[[1]][1,,] <- matrix(c(1:3, rep(Inf, 3)), 2, 3, byrow = TRUE)
    expected_res[[1]][2,,] <- matrix(c(1:3, rep(Inf, 3)), 2, 3, byrow = TRUE)
    expect_equal(res, expected_res, tolerance = 1e-5)
    
    
    
    w <- matrix(c(2, 4, 6, 3, 2, 8, 9, 1, 3), 3, 3)
    w_i <- LazyTensor(w, index = 'i')
    x <- matrix(runif(150 * 3), 150, 3) 
    x_i <- LazyTensor(x, index = 'i') 
    y <- matrix(runif(100 * 3), 100, 3)
    y_j <- LazyTensor(y, index = 'j')
    
    K <- 2
    
    # check formulas, args & classes
    S_ij <- sum((x_i - y_j)^2)
    res <- Kmin_argKmin(S_ij, K, "i")
    expect_false(is.LazyTensor(res))
    checkmate::expect_list(res, len = 2)
    
    # check results for simple example
    res <- Kmin_argKmin(w_i, K, 'i')
    # grab min values for K = 1, 2
    Kmin_K_1 <- c(min(w[, 1]),
                  min(w[, 2]),
                  min(w[, 3]))
    Kmin_K_2 <- c(min(w[, 1][-which.min(w[, 1])]),
                  min(w[, 2][-which.min(w[, 2])]),
                  min(w[, 3][-which.min(w[, 3])]))
    # grab index of min values
    argKmin_K_1 <- c(which.min(w[, 1]) - 1,
                     which.min(w[, 2]) - 1,
                     which.min(w[, 3]) - 1)
    # replace min values by huge ones
    w[, 1][which.min(w[, 1])] <- 100
    w[, 2][which.min(w[, 2])] <- 100
    w[, 3][which.min(w[, 3])] <- 100
    # grab index of (second) min values
    argKmin_K_2 <- c(which.min(w[, 1]) - 1,
                     which.min(w[, 2]) - 1,
                     which.min(w[, 3]) - 1)
    # expected result
    expected_res <- list(
        array(
            matrix(c(Kmin_K_1, Kmin_K_2), 2, 3, byrow = TRUE), dim = c(1, 2, 3)
        ),
        array(
            matrix(c(argKmin_K_1, argKmin_K_2), 2, 3, byrow = TRUE), 
            dim = c(1, 2, 3)
        )
    )
    expect_equal(res, expected_res, tolerance = 1e-5)
    
    # errors
    expect_error(
        Kmin_argKmin(x_i, 3.14, "i"),
        "`K` input argument should be an integer.",
        fixed = TRUE
    )
    expect_error(
        Kmin_argKmin(x_i, K, "k"),
        "`index` input argument should be a character, either 'i' or 'j'.",
        fixed = TRUE
    )
    
})


test_that("Kmin_argKmin_reduction", {
    
    x <- matrix(c(1, 2, 3), 2, 3, byrow = TRUE)
    x_i <- LazyTensor(x, index = 'i')
    
    res <- Kmin_argKmin_reduction(x_i, 2, "i")
    expect_false(is.LazyTensor(res))
    checkmate::expect_list(res, len = 2)
    expect_equal(dim(res[[1]]), c(1, 2, 3))
    expect_equal(dim(res[[2]]), c(1, 2, 3))
    expected_res <- list(
        array(x, dim = c(1,2,3)),
        array(matrix(c(0,1), 2, 3), dim = c(1,2,3))
    )
    expect_equal(res, expected_res, tolerance = 1e-5)
    
    res <- Kmin_argKmin_reduction(x_i, 2, "j")
    expect_false(is.LazyTensor(res))
    checkmate::expect_list(res, len = 2)
    expect_equal(dim(res[[1]]), c(2, 2, 3))
    expect_equal(dim(res[[2]]), c(2, 2, 3))
    expected_res <- list(
        array(0, dim = c(2,2,3)),
        array(matrix(0, 2, 3), dim = c(2,2,3))
    )
    expected_res[[1]][1,,] <- matrix(c(1:3, rep(Inf, 3)), 2, 3, byrow = TRUE)
    expected_res[[1]][2,,] <- matrix(c(1:3, rep(Inf, 3)), 2, 3, byrow = TRUE)
    expect_equal(res, expected_res, tolerance = 1e-5)
    
    
    
    
    x <- matrix(runif(150 * 3), 150, 3) 
    x_i <- LazyTensor(x, index = 'i') 
    y <- matrix(runif(100 * 3), 100, 3)
    y_j <- LazyTensor(y, index = 'j')
    
    K <- 2
    
    # check formulas, args & classes
    S_ij <- sum((x_i - y_j)^2)
    res <- Kmin_argKmin_reduction(S_ij, K, "i")
    expect_false(is.LazyTensor(res))
    checkmate::expect_list(res, len = 2)
    
    # errors
    expect_error(
        Kmin_argKmin_reduction(x_i, 3.14, "i"),
        "`K` input argument should be an integer.",
        fixed = TRUE
    )
    
    expect_error(
        Kmin_argKmin_reduction(x_i, K, "k"),
        "`index` input argument should be a character, either 'i' or 'j'.",
        fixed = TRUE
    )
    
})


test_that("logsumexp", {
    x <- matrix(runif(12), 4, 3)
    y <- matrix(runif(15), 5, 3)
    w <- matrix(c(1., 1., 1.), 5, 3)
    
    x_i <- LazyTensor(x, index = 'i')
    y_j <- LazyTensor(y, index = 'j')
    w_j <- LazyTensor(w, index = 'j')
    
    V_ij <- x_i - y_j
    S_ij <- sum(V_ij^2)
    
    expect_error(logsumexp(sum(V_ij), 'i', w_j))
    
    res <- logsumexp(S_ij, 'i')
    expect_false(is.LazyTensor(res))
    expect_true(is.matrix(res))
    expect_equal(dim(res), c(5, 1))
    expect_true(is.matrix(res))
    expected_res <- apply(
        t(sapply(
            1:nrow(x), function(id_x) 
                sapply(1:nrow(y), function(id_y) sum((x[id_x,] - y[id_y,])^2))
        )), 2, function(vec) return(log(sum(exp(vec)))))
    expect_equal(as.vector(res), expected_res, tolerance = 1e-5)
    
    # errors
    expect_error(
        logsumexp(S_ij, '9'),
        "`index` input argument should be a character, either 'i' or 'j'.",
        fixed = TRUE
    )
    
})

test_that("logsumexp_reduction", {
    x <- matrix(runif(150 * 3), 150, 3) 
    x_i <- LazyTensor(x, index = 'i') 
    y <- matrix(runif(100 * 3), 100, 3)
    y_j <- LazyTensor(y, index = 'j')
    w <- matrix(runif(100 * 3), 100, 3) # weight LazyTensor
    w_j <- LazyTensor(w, index = 'j')
    
    S_ij <- sum( (x_i - y_j)^2 )
    # check formulas, args & classes
    expect_error(logsumexp(sum(V_ij), 'i', w_j))
    
    res <- logsumexp(S_ij, 'i')
    expect_false(is.LazyTensor(res))
    expect_true(is.matrix(res))
    expect_equal(dim(res), c(100, 1))
    expect_true(is.matrix(res))
    expected_res <- apply(
        t(sapply(
            1:nrow(x), function(id_x) 
                sapply(1:nrow(y), function(id_y) sum((x[id_x,] - y[id_y,])^2))
        )), 2, function(vec) return(log(sum(exp(vec)))))
    expect_equal(as.vector(res), expected_res, tolerance = 1e-5)
    
    
    # errors
    expect_error(
        logsumexp_reduction(S_ij, '9'),
        "`index` input argument should be a character, either 'i' or 'j'.",
        fixed = TRUE
    )
    
})


test_that("sumsoftmaxweight", {
    
    ## basic example (reduction on 'i' -> gives back weights)
    x <- matrix(runif(12), 4, 3)
    y <- matrix(runif(15), 5, 3)
    x_i <- LazyTensor(x, index = 'i') 
    y_j <- LazyTensor(y, index = 'j')
    
    # with weights equal to 1
    w <- matrix(1, 5, 3)
    w_j <- LazyTensor(w, index = 'j')
    
    V_ij <- x_i - y_j
    S_ij <- sum(V_ij^2)
    
    res <- sumsoftmaxweight(S_ij, 'i', w_j)
    expect_false(is.LazyTensor(res))
    expect_true(is.matrix(res))
    expect_equal(res, w, tolerance = 1e-5)
    
    # with random weights
    w <- matrix(runif(15), 5, 3)
    w_j <- LazyTensor(w, index = 'j')
    
    V_ij <- x_i - y_j
    S_ij <- sum(V_ij^2)
    
    res <- sumsoftmaxweight(S_ij, 'i', w_j)
    expect_false(is.LazyTensor(res))
    expect_true(is.matrix(res))
    expect_equal(res, w, tolerance = 1e-5)
    
    ## basic example (reduction on 'j')
    x <- matrix(runif(12), 4, 3)
    y <- matrix(runif(15), 5, 3)
    x_i <- LazyTensor(x, index = 'i') 
    y_j <- LazyTensor(y, index = 'j')
    
    # with weights equal to 1
    w <- matrix(1, 5, 3)
    w_j <- LazyTensor(w, index = 'j')
    
    V_ij <- x_i - y_j
    S_ij <- sum(V_ij^2)
    
    res <- sumsoftmaxweight(S_ij, 'j', w_j)
    expect_false(is.LazyTensor(res))
    expect_true(is.matrix(res))
    expect_equal(res, matrix(1, 4, 3))
    
    # with random weights
    w <- matrix(runif(15), 5, 3)
    w_j <- LazyTensor(w, index = 'j')
    
    V_ij <- x_i - y_j
    S_ij <- sum(V_ij^2)
    
    res <- sumsoftmaxweight(S_ij, 'j', w_j)
    expect_false(is.LazyTensor(res))
    expect_true(is.matrix(res))
    
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
    expected_res <- exp_S_val %*% w / matrix(apply(exp_S_val, 1, sum), 4, 3)
    expect_equal(res, expected_res, tolerance = 1e-5)
    
    # more complicated example
    x <- matrix(runif(150 * 3), 150, 3)
    x_i <- LazyTensor(x, index = 'i')
    y <- matrix(runif(100 * 3), 100, 3)
    y_j <- LazyTensor(y, index = 'j')

    V_ij <- x_i - y_j   # weight matrix
    S_ij = sum(V_ij^2)

    res <- sumsoftmaxweight(S_ij, 'i', V_ij)
    expect_false(is.LazyTensor(res))
    expect_true(is.matrix(res))
    # TODO: compare results with R code
    
    res <- sumsoftmaxweight(S_ij, 'j', V_ij)
    expect_false(is.LazyTensor(res))
    expect_true(is.matrix(res))
    # TODO: compare results with R code
    
    # errors
    expect_error(
        sumsoftmaxweight(S_ij, '9', V_ij),
        "`index` input argument should be a character, either 'i' or 'j'.",
        fixed = TRUE
    )
    
})


test_that("sumsoftmaxweight_reduction", {
    x <- matrix(runif(150 * 3), 150, 3) 
    x_i <- LazyTensor(x, index = 'i') 
    y <- matrix(runif(150 * 3), 150, 3)
    y_j <- LazyTensor(y, index = 'j')
    
    V_ij <- x_i - y_j
    S_ij <- sum(V_ij^2)
    
    # check formulas, args & classes
    res <- sumsoftmaxweight_reduction(S_ij, 'i', V_ij)
    expect_false(is.LazyTensor(res))
    expect_true(is.matrix(res))
    
    # errors
    expect_error(
        sumsoftmaxweight_reduction(S_ij, '9', V_ij),
        "`index` input argument should be a character, either 'i' or 'j'.",
        fixed = TRUE
    )
    
})


test_that("grad", {
    # data
    nx <- 10
    ny <- 15
    x <- matrix(runif(nx * 3), nrow = nx, ncol = 3)
    y <- matrix(runif(ny * 3), nrow = ny, ncol = 3)
    eta <- matrix(1, nrow = nx, ncol = 1)
    
    # LazyTensors
    x_i <- Vi(x)
    y_j <- Vj(y)
    eta_i <- Vi(eta)
    
    res1 <- grad(sqnorm2(x_i - y_j), eta_i, "Sum", var = x_i, "j") 
    expect_true(is.matrix(res1))
    expect_equal(dim(res1), c(nx, 3))
    expected_res <- expected_res <- sapply(1:nx, function(i) {
        tmp <- sapply(1:ny, function(j) {
            return(2 * (x[i, ] - y[j, ]))
        })
        return(apply(tmp,1,sum))
    })
    expect_true(sum(abs(res1 - t(expected_res))) < 1E-4)
    
    res2 <- grad(sqnorm2(x_i - y_j), eta_i, "Sum", var = 0, "j")
    expect_true(is.matrix(res2))
    expect_equal(dim(res2), c(nx, 3))
    expect_true(sum(abs(res1 - res2)) < 1E-4)
    
    eta <- matrix(1, nrow = ny, ncol = 1)
    eta_j <- Vj(eta)
    
    res3 <- grad(sqnorm2(y_j - x_i), eta_j, "Sum", var = x_i, "i") 
    expect_true(is.matrix(res3))
    expect_equal(dim(res3), c(ny, 3))
    expected_res <- expected_res <- sapply(1:ny, function(i) {
        tmp <- sapply(1:nx, function(j) {
            return(2 * (y[i, ] - x[j, ]))
        })
        return(apply(tmp, 1, sum))
    })
    expect_true(sum(abs(res3 - t(expected_res))) < 1E-4)
    
    # errors
    expect_error(
        grad(sqnorm2(x_i - y_j), eta_j, "Sum", var = 0, "j"),
        paste0("`gradin` input argument should be indexed by 'i'"), 
        fixed = TRUE
    )
    
    bad_eta <- matrix(1, nrow = nx + 2, ncol = 1)
    bad_eta_i <- Vi(bad_eta)
    
    expect_error(
        grad(sqnorm2(x_i - y_j), bad_eta_i, "Sum", var = 0, "j"),
        paste0(
            "`gradin` input argument should be a LazyTensor encoding a matrix ",
            "of shape (10,1)."
        ), 
        fixed = TRUE
    )
    
    expect_error(
        grad(sqnorm2(x_i - y_j), eta_j, 2, var = 0, "j"),
        paste0("`opstr` input should be a string text corresponding", 
               " to a reduction formula."), 
        fixed = TRUE
    )
    
    expect_error(
        grad(sqnorm2(x_i - y_j), eta_i, "Sum", var = 0,  1),
        paste0("`index` input argument should be a character, either 'i' or 'j'."), 
        fixed = TRUE
    )
    
    expect_error(
        grad(sqnorm2(x_i - y_j), eta_i, "Sum", var = 0,  "i"),
        paste0("`index` input argument should be 'j'."), 
        fixed = TRUE
    )
    
})
