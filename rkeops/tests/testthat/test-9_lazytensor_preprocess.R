skip_if_no_python()
skip_if_no_keopscore()
skip_if_no_pykeops()

# TEST LAZYTENSOR CONFIGURATION ================================================

# Tests for LazyTensor related functions
#
# Use of regular expressions to check formulas and arguments
# since we use pointer addresses as unique variable id


test_that("LazyTensor", {
    # basic examples
    D <- 3
    M <- 100
    x <- matrix(runif(M * D), M, D)
    u <- runif(M, min = 0, max = 3.14)
    x_i <- LazyTensor(x, index = 'i')
    x_j <- LazyTensor(x, index = 'j')
    Pm_u <- LazyTensor(u) # parameter vector
    Pm_int <- LazyTensor(D) # parameter scalar
    Pm_dec <- LazyTensor(3.14)
    
    z <- matrix(1i^ (-6:5), nrow = 4)                     # complex 4x3 matrix
    z_i <- LazyTensor(z, index = 'i', is_complex = TRUE)  # ComplexLazyTensor
    still_good_z_Vi <- LazyTensor(z, index = 'i') # without specifying 
    # "is_complex = TRUE": should work as well.
    cplx <- 3 + 2i
    Pm_cplx <- LazyTensor(cplx)
    
    # check classes
    classes <- c(class(x_i), class(x_j), class(Pm_u),
                 class(Pm_int), class(Pm_dec))
    k <- length(classes)
    expect_equal(classes, rep("LazyTensor", k))
    
    expect_true(is.ComplexLazyTensor(z_i))
    expect_true(is.ComplexLazyTensor(still_good_z_Vi))
    expect_true(is.ComplexLazyTensor(Pm_cplx))
    
    # check formulae
    bool_grep_x_i <- grep("A0x.*i", x_i$formula)
    expect_equal(bool_grep_x_i, 1)
    bool_grep_x_j <- grep("A0x.*j", x_j$formula)
    expect_equal(bool_grep_x_j, 1)
    bool_grep_Pm_u <- grep("A0x.*NA", Pm_u$formula)
    expect_equal(bool_grep_Pm_u, 1)
    bool_grep_Pm_int <- grep("IntCst\\(3\\)", Pm_int$formula)
    expect_equal(bool_grep_Pm_int, 1)
    bool_grep_Pm_dec <- grep("A0x.*NA", Pm_dec$formula)
    expect_equal(bool_grep_Pm_dec, 1)
    bool_grep_z_i <- grep("A0x.*i", z_i$formula)
    expect_equal(bool_grep_z_i, 1)
    bool_grep_Pm_cplx <- grep("A0x.*NA", Pm_cplx$formula)
    expect_equal(bool_grep_Pm_cplx, 1)
    
    # check args
    bool_grep_x_i <- grep("A0x.*i=Vi\\(3\\)", x_i$args)
    expect_equal(bool_grep_x_i, 1)
    bool_grep_x_j <- grep("A0x.*j=Vj\\(3\\)", x_j$args)
    expect_equal(bool_grep_x_j, 1)
    bool_grep_Pm_u <- grep("A0x.*NA=Pm\\(100\\)", Pm_u$args)
    expect_equal(bool_grep_Pm_u, 1)
    expect_null(Pm_int$args)
    bool_grep_Pm_dec <- grep("A0x.*NA=Pm\\(1\\)", Pm_dec$args)
    expect_equal(bool_grep_Pm_dec, 1)
    bool_grep_z_i <- grep("A0x.*i=Vi\\(6\\)", z_i$args)
    expect_equal(bool_grep_z_i, 1)
    bool_grep_Pm_cplx <- grep("A0x.*NA=Pm\\(2\\)", Pm_cplx$args)
    expect_equal(bool_grep_Pm_cplx, 1)
    
    # check data
    expect_true(is.matrix(x_i$data[[1]]))
    expect_true(is.matrix(x_j$data[[1]]))
    expect_true(is.vector(Pm_u$data[[1]]))
    expect_null(Pm_int$data)
    expect_equal(Pm_dec$data[[1]], 3.14)
    expect_true(is.matrix(z_i$data[[1]]))
    expect_true(is.matrix(Pm_cplx$data[[1]]))
    
    # check dimres
    expect_equal(D, x_i$dimres)
    expect_equal(D, x_j$dimres)
    expect_equal(M, Pm_u$dimres)
    expect_equal(1, Pm_int$dimres)
    expect_equal(3, z_i$dimres)
    expect_equal(3, still_good_z_Vi$dimres)
    expect_equal(1, Pm_cplx$dimres)
    
    # check complex data
    Z <- z_i$data[[1]]
    # The number of column of Z is twice the number of column of z.
    expect_equal(ncol(Z), ncol(z) * 2)
    
    # The first column of Z is the real part of the first column of z;
    # the second column of Z is the imaginary part of the first column of z;
    # the third column of Z is the real part of the second column of z;
    # and so on.
    expect_equal(Z[, 1], Re(z[, 1]))
    expect_equal(Z[, 2], Im(z[, 1]))
    expect_equal(Z[, 3], Re(z[, 2]))
    
    # same principle with a single parameter complex value
    data_Pm_cplx <- Pm_cplx$data[[1]]
    expect_equal(ncol(data_Pm_cplx), 2)
    
    expect_equal(data_Pm_cplx[, 1], Re(cplx))
    expect_equal(data_Pm_cplx[, 2], Im(cplx))
    
    # errors
    expect_error(LazyTensor(x_i), 
                 "Input `x` is already a LazyTensor.", 
                 fixed = TRUE)
    expect_error(LazyTensor("x"), 
                 paste("`x` input argument should be a matrix, a vector",
                       "a scalar or a complex value.",
                       sep = ""),
                 fixed = TRUE)
    expect_error(LazyTensor(x), 
                 "missing `index` argument.", 
                 fixed = TRUE)
    expect_error(LazyTensor(u, index = "i"), 
                 "`index` must be NA with a vector or a single value.", 
                 fixed = TRUE)
    expect_error(LazyTensor(D, index = "i"), 
                 "`index` must be NA with a vector or a single value.", 
                 fixed = TRUE)
})



test_that("Vi", {
    D <- 3
    M <- 100
    x <- matrix(runif(M * D), M, D)
    u <- runif(M, min = 0, max = 3.14)
    z <- matrix(1i^ (-6:5), nrow = 4) # complex 4x3 matrix
    
    x_i <- LazyTensor(x, index = 'i')
    x_Vi <- Vi(x)
    
    # ComplexLazyTensor
    z_i <- LazyTensor(z, index = 'i', is_complex = TRUE)
    z_Vi <- Vi(z, is_complex = TRUE)
    still_good_z_Vi <- Vi(z) # without specifying 
    # "is_complex = TRUE": should work as well.
    
    # check classes
    expect_true(is.LazyTensor(x_Vi))
    expect_true(is.ComplexLazyTensor(z_Vi))
    expect_true(is.ComplexLazyTensor(still_good_z_Vi))
    
    # check formulae
    bool_grep_i <- grep("A0x.*i", x_Vi$formula)
    expect_equal(bool_grep_i, 1)
    bool_grep_zi <- grep("A0x.*i", z_Vi$formula)
    expect_equal(bool_grep_zi, 1)
    
    # check args
    expect_equal(x_i$args, x_Vi$args)
    expect_equal(z_i$args, z_Vi$args)
    expect_equal(z_i$args, still_good_z_Vi$args)
    
    # check dimres
    expect_equal(D, x_Vi$dimres)
    expect_equal(D, z_Vi$dimres)
    
    # errors
    expect_error(Vi(x_i), 
                 "`x` must be a matrix.", 
                 fixed = TRUE)
    expect_error(Vi(u), 
                 "`x` must be a matrix.", 
                 fixed = TRUE)
    expect_error(Vi(3), 
                 "`x` must be a matrix.", 
                 fixed = TRUE)
    expect_error(Vi("3"), 
                 "`x` must be a matrix.", 
                 fixed = TRUE)
    
})


test_that("Vj", {
    D <- 3
    M <- 100
    x <- matrix(runif(M * D), M, D)
    u <- runif(M, min = 0, max = 3.14)
    z <- matrix(1i^ (-6:5), nrow = 4) # complex 4x3 matrix
    
    x_j <- LazyTensor(x, index = 'j')
    x_Vj <- Vj(x)
    
    # ComplexLazyTensor
    z_j <- LazyTensor(z, index = 'j', is_complex = TRUE)
    z_Vj <- Vj(z, is_complex = TRUE)
    still_good_z_Vj <- Vj(z) # without specifying 
    # "is_complex = TRUE": should work as well.
    
    # check classes
    expect_true(is.LazyTensor(x_Vj))
    expect_true(is.ComplexLazyTensor(z_Vj))
    expect_true(is.ComplexLazyTensor(still_good_z_Vj))
    
    # check formulae
    bool_grep_j <- grep("A0x.*j", x_Vj$formula)
    expect_equal(bool_grep_j, 1)
    bool_grep_zj <- grep("A0x.*j", z_Vj$formula)
    expect_equal(bool_grep_zj, 1)
    
    # check arguments 
    expect_equal(x_j$args, x_Vj$args)
    expect_equal(z_j$args, z_Vj$args)
    expect_equal(z_j$args, still_good_z_Vj$args)
    expect_true(is.LazyTensor(x_Vj))
    expect_true(is.ComplexLazyTensor(z_Vj))
    expect_true(is.ComplexLazyTensor(still_good_z_Vj))
    
    # check dimres
    expect_equal(D, x_Vj$dimres)
    expect_equal(D, z_Vj$dimres)
    
    # errors
    expect_error(Vj(x_j), 
                 "`x` must be a matrix.", 
                 fixed = TRUE)
    expect_error(Vj(u), 
                 "`x` must be a matrix.", 
                 fixed = TRUE)
    expect_error(Vj(3), 
                 "`x` must be a matrix.", 
                 fixed = TRUE)
    expect_error(Vj("3"), 
                 "`x` must be a matrix.", 
                 fixed = TRUE)
    
})


test_that("Pm", {
    D <- 3
    M <- 100
    x <- matrix(runif(M * D), M, D)
    u <- c(2, 3, 5, 7, 11, 13)
    z <- rep(1i^(-6:5), 4) # complex vector
    
    int_LT <- LazyTensor(D)
    int_Pm <- Pm(D)
    dec <- 7.2
    dec_LT <- LazyTensor(dec)
    dec_Pm <- Pm(dec)
    u_LT <- LazyTensor(u)
    u_Pm <- Pm(u)
    z_LT <- LazyTensor(z)
    z_Pm <- Pm(z)
    x_i <- LazyTensor(x, index = 'i')
    
    # check classes
    expect_true(is.LazyTensor(int_Pm))
    expect_true(is.LazyTensor(dec_Pm))
    expect_true(is.LazyTensor(u_Pm))
    expect_true(is.ComplexLazyTensor(z_Pm))
    
    # check formulae
    bool_grep_int_Pm <- grep("IntCst\\(3\\)", int_Pm$formula)
    expect_equal(bool_grep_int_Pm, 1)
    bool_grep_dec_Pm <- grep("A0x.*NA", dec_Pm$formula)
    expect_equal(bool_grep_dec_Pm, 1)
    bool_grep_u_Pm <- grep("A0x.*NA", u_Pm$formula)
    expect_equal(bool_grep_u_Pm, 1)
    bool_grep_Z_LT <- grep("A0x.*NA", z_LT$formula)
    expect_equal(bool_grep_Z_LT, 1)
    
    # check arguments
    expect_null(int_Pm$args)
    expect_true(dec_LT$args == dec_Pm$args)
    expect_true(u_LT$args == u_Pm$args)
    expect_true(z_LT$args == z_Pm$args)
    
    # check data
    expect_null(int_Pm$data[[1]])
    expect_equal(dec_LT$data[[1]], dec_Pm$data[[1]])
    expect_equal(u_LT$data[[1]], u_Pm$data[[1]])
    expect_equal(z_LT$data[[1]], z_Pm$data[[1]])
    
    # check dimres
    expect_equal(1, int_Pm$dimres)
    expect_equal(1, dec_Pm$dimres)
    expect_equal(6, u_Pm$dimres)
    expect_equal(length(z), z_Pm$dimres)
    
    # errors
    expect_error(Pm(x_i), 
                 "`x` input is already a LazyTensor.", 
                 fixed = TRUE)
    expect_error(Pm(x), 
                 "`x` input must be a vector or a single value.", 
                 fixed = TRUE)
    expect_error(Pm("a"), 
                 "`x` input must be a vector or a single value.", 
                 fixed = TRUE)
    
})



# TEST LAZYTENSOR *NARYOP ======================================================


test_that("unaryop.LazyTensor", {
    D <- 3
    M <- 100
    x <- matrix(runif(M * D), M, D)
    x_i <- LazyTensor(x, index = 'i')
    z <- matrix(1i^ (-6:5), nrow = 4) # complex 4x3 matrix
    # ComplexLazyTensor
    z_j <- LazyTensor(z, index = 'j', is_complex = TRUE)
    
    Pm_int <- LazyTensor(D)
    Pm_dec <- LazyTensor(3.14)
    Pm_v <- LazyTensor(c(2, 3, 5, 7, 11, 13))
    
    # check formulas, args, dimres and classes for several unary operations
    obj <- unaryop.LazyTensor(x_i, "Square")
    bool_grep_formula <- grep("Square\\(A0x.*i\\)", obj$formula)
    expect_equal(bool_grep_formula, 1)
    bool_grep_args <- grep("A0x.*i=Vi\\(3\\)", obj$args[1])
    expect_equal(bool_grep_args, 1)
    expect_equal(obj$dimres, x_i$dimres)
    expect_true(is.LazyTensor(obj))
    
    obj <- unaryop.LazyTensor(z_j, "Square")
    expect_true(is.ComplexLazyTensor(obj))
    
    obj <- unaryop.LazyTensor(z_j, "ComplexAbs", res_type = "LazyTensor")
    expect_false(is.ComplexLazyTensor(obj))
    expect_true(is.LazyTensor(obj))
    
    obj <- exp(Pm_int)
    bool_grep_formula <- grep("Exp\\(IntCst\\(3\\)\\)", obj$formula)
    expect_null <- obj$args
    expect_null <- obj$data
    expect_equal(obj$dimres, 1)
    expect_true(is.LazyTensor(obj))
    
    obj <- log(Pm_dec)
    bool_grep_formula <- grep("Log\\(A0x.*NA\\)", obj$formula)
    expect_equal(bool_grep_formula, 1)
    bool_grep_args <- grep("A0x.*NA=Pm\\(1\\)", obj$args[1])
    expect_equal(bool_grep_args, 1)
    expect_equal(obj$data[[1]], 3.14)
    expect_equal(obj$dimres, 1)
    expect_true(is.LazyTensor(obj))
    
    # check dimres for some special cases
    una_x <- unaryop.LazyTensor(x_i, "Minus")   # symbolic matrix
    expect_equal(D, una_x$dimres)
    una2_x <- unaryop.LazyTensor(x_i, "Norm2",
                                 res_type = "LazyTensor",
                                 dim_res = 1)
    expect_equal(1, una2_x$dimres)
    una3_x <- unaryop.LazyTensor(x_i, "Norm2",
                                 res_type = "LazyTensor",
                                 dim_res = 1.0)
    expect_equal(1, una3_x$dimres)
    
    # errors
    expect_error(
        unaryop.LazyTensor(x, "Square"), 
        paste("`x` input argument should be a LazyTensor, a vector or a scalar.",
              "\nIf you want to use a matrix, convert it to LazyTensor first.",
              sep = ""),
        fixed = TRUE
    )
    expect_error(unaryop.LazyTensor(x_i, "Norm2", dim_res = 3.14), 
                 paste(
                     "If not NA, `dim_res` input argument should be an integer. ",
                     "Received 3.14.",
                     sep = ""), 
                 fixed = TRUE)
})


test_that("binaryop.LazyTensor", {
    # basic example
    D <- 3
    E <- 7
    M <- 100
    N <- 150
    x <- matrix(runif(M * D), M, D)
    y <- matrix(runif(N * D), N, D)
    z <- matrix(runif(N * E), N, E)
    x_i <- LazyTensor(x, index = 'i')
    x_j <- LazyTensor(x, index = 'j')
    y_i <- LazyTensor(y, index = 'i')
    y_j <- LazyTensor(y, index = 'j')
    z_j <- LazyTensor(z, index = 'j')
    # ComplexLazyTensor
    xc_i <- LazyTensor(x, index = 'i', is_complex = TRUE)
    
    # check formulas, args, dimres and classes for several binary operations
    obj <-  binaryop.LazyTensor(x_i, y_j, "Sum")
    bool_grep_formula <- grep("Sum\\(A0x.*i,A0x.*j\\)", obj$formula)
    expect_equal(bool_grep_formula, 1)
    expect_equal(length(obj$args), 2)
    bool_grep_args1 <- grep("A0x.*i=Vi\\(3\\)", obj$args[1])
    expect_equal(bool_grep_args1, 1)
    bool_grep_args2 <- grep("A0x.*j=Vj\\(3\\)", obj$args[2])
    expect_equal(bool_grep_args2, 1)
    expect_s3_class(obj, "LazyTensor")
    
    obj <-  binaryop.LazyTensor(x_i, y_j, "-", is_operator = TRUE)
    bool_grep_formula <- grep("A0x.*i-A0x.*j", obj$formula)
    bool_grep_args1 <- grep("A0x.*i=Vi\\(3\\)", obj$args[1])
    bool_grep_args2 <- grep("A0x.*j=Vj\\(3\\)", obj$args[2])
    expect_equal(bool_grep_formula, 1)
    expect_equal(bool_grep_args1, 1)
    expect_equal(bool_grep_args2, 1)
    expect_s3_class(obj, "LazyTensor")
    
    obj <-  binaryop.LazyTensor(x_i, 3.9, "Powf")
    bool_grep_formula <- grep("Powf\\(A0x.*i,A0x.*NA\\)", obj$formula)
    bool_grep_args1 <- grep("A0x.*i=Vi\\(3\\)", obj$args[1])
    expect_equal(bool_grep_formula, 1)
    expect_equal(bool_grep_args1, 1)
    expect_equal(length(obj$args), 2)
    expect_s3_class(obj, "LazyTensor")
    
    obj <-  binaryop.LazyTensor(x_i, 3, "+", is_operator = TRUE)
    bool_grep_formula <- grep("A0x.*i\\+IntCst\\(3\\)", obj$formula)
    expect_equal(bool_grep_formula, 1)
    expect_equal(length(obj$args), 1) # no args added for IntCst
    bool_grep_args <- grep("A0x.*i=Vi\\(3\\)", obj$args)
    expect_equal(length(obj$data), 1) # no data added for IntCst
    expect_equal(obj$dimres, 3)
    expect_s3_class(obj, "LazyTensor")
    
    obj <-  binaryop.LazyTensor(x_i, x_i, "+", is_operator = TRUE)
    expect_equal(length(obj$args), 1)
    expect_equal(length(obj$data), 1)
    expect_true(is.LazyTensor(obj))
    
    obj <-  binaryop.LazyTensor(xc_i, x_i, "+", is_operator = TRUE)
    expect_equal(length(obj$args), 2)
    expect_equal(length(obj$data), 2)
    expect_true(is.ComplexLazyTensor(obj))
    
    # check dimres for some special cases
    bin1 <- binaryop.LazyTensor(xc_i, x_i, "Add", is_operator = TRUE)
    expect_equal(D, bin1$dimres)
    
    bin2 <- binaryop.LazyTensor(x_i, y_j, "SqDist",
                                res_type = "LazyTensor",
                                dim_res = 1)
    expect_equal(1, bin2$dimres)
    
    # check particular cases with duplicate inputs
    v <- c(2, 7, 8)
    Pm_v <- Pm(v)
    
    obj <- x_i + y_j + Pm_v + x_j + y_j + x_i + Pm_v
    expect_equal(length(obj$args), 4)
    expect_equal(length(obj$data), 4)
    expect_equal(obj$data[[1]], obj$data[[4]])
    
    obj <- x_i + 3 + Pm_v + x_j + 7 + x_i + Pm_v
    expect_equal(length(obj$args), 3)
    expect_equal(length(obj$data), 3)
    
    # errors
    expect_error(
        binaryop.LazyTensor(x, y_j, "+"), 
        paste(
            "`x` input argument should be a LazyTensor, a vector or a scalar.",
            "\nIf you want to use a matrix, convert it to LazyTensor first.",
            sep = ""
        ), 
        fixed = TRUE
    )
    
    expect_error(
        binaryop.LazyTensor(x_i, z_j, "|", is_operator = TRUE, 
                            dim_check_type = "same"),
        "Operation `|` expects inputs of the same dimension. Received 3 and 7.",
        fixed = TRUE
    )
    
    expect_error(
        binaryop.LazyTensor(x_i, y_j, "SqDist", res_type = "LazyTensor",
                            dim_res = 3.14), 
        paste(
            "If not NA, `dim_res` input argument should be an integer. ",
            "Received 3.14.", sep = ""
        ), 
        fixed = TRUE
    )
    
})


test_that("ternaryop.LazyTensor", {
    # basic example
    D <- 3
    M <- 10
    N <- 15
    P <- 20
    w <- matrix(runif(M * D), M, D)
    x <- matrix(runif(M * D), M, D)
    y <- matrix(runif(N * D), N, D)
    z <- matrix(runif(P * D), P, D)
    w_j <- LazyTensor(w, index = 'j')
    x_i <- LazyTensor(x, index = 'i')
    x_j <- LazyTensor(x, index = 'j')
    y_i <- LazyTensor(y, index = 'i')
    y_j <- LazyTensor(y, index = 'j')
    z_i <- LazyTensor(z, index = 'i')
    
    # check formulas, args, dimres and classes for several ternary operations
    obj <-  ternaryop.LazyTensor(x_i, y_j, z_i, "Clamp")
    bool_grep_formula <- grep("Clamp\\(A0x.*i,A0x.*j,A0x.*i\\)", obj$formula)
    bool_grep_args1 <- grep("A0x.*i=Vi\\(3\\)", obj$args[1])
    bool_grep_args2 <- grep("A0x.*j=Vj\\(3\\)", obj$args[2])
    bool_grep_args3 <- grep("A0x.*i=Vi\\(3\\)", obj$args[3])
    expect_equal(bool_grep_formula, 1)
    expect_equal(length(obj$args), 3)
    expect_equal(bool_grep_args1, 1)
    expect_equal(bool_grep_args2, 1)
    expect_equal(bool_grep_args3, 1)
    expect_s3_class(obj, "LazyTensor")
    
    # with an IntCst
    obj <-  ternaryop.LazyTensor(x_i, w_j, 7, "IfElse")
    bool_grep_formula <- grep("IfElse\\(A0x.*i,A0x.*j,IntCst\\(7\\)\\)", obj$formula)
    expect_equal(bool_grep_formula, 1)
    expect_equal(length(obj$args), 2) # no args added for IntCst input
    bool_grep_args1 <- grep("A0x.*i=Vi\\(3\\)", obj$args[1])
    bool_grep_args2 <- grep("A0x.*j=Vj\\(3\\)", obj$args[2])
    expect_equal(bool_grep_args1, 1)
    expect_equal(bool_grep_args2, 1)
    expect_equal(length(obj$data), 2) # no data added for IntCst input
    expect_s3_class(obj, "LazyTensor")
    
    obj <-  ternaryop.LazyTensor(4, y_j, z_i, "Clamp")
    bool_grep_formula <- grep("Clamp\\(IntCst\\(4\\),A0x.*j,A0x.*i\\)", obj$formula)
    expect_equal(bool_grep_formula, 1)
    expect_equal(length(obj$args), 2) # no args added for IntCst input
    bool_grep_args1 <- grep("A0x.*j=Vj\\(3\\)", obj$args[1])
    bool_grep_args2 <- grep("A0x.*i=Vi\\(3\\)", obj$args[2])
    expect_equal(bool_grep_args1, 1)
    expect_equal(bool_grep_args2, 1)
    expect_equal(length(obj$data), 2) # no data added for IntCst input
    expect_true(is.LazyTensor(obj))
    
    obj <-  ternaryop.LazyTensor(x_i, x_i, x_i, "Clamp")
    expect_equal(length(obj$args), 1)
    expect_equal(length(obj$data), 1)
    expect_true(is.LazyTensor(obj))
    
    obj <-  ternaryop.LazyTensor(x_i, x_i, z_i, "Clamp")
    expect_equal(length(obj$args), 2)
    expect_equal(length(obj$data), 2)
    expect_true(is.LazyTensor(obj))
    
    # check dimres
    tern1 <- ternaryop.LazyTensor(4, y_j, z_i, "Clamp")
    expect_equal(D, tern1$dimres)
    
    # check particular cases with duplicate inputs
    obj <- clamp(x_i, x_j, x_i)
    expect_equal(length(obj$args), 2)
    expect_equal(length(obj$data), 2)
    expect_equal(obj$data[[1]], obj$data[[2]])
    
    obj <- ifelse(x_i, 3, x_j)
    expect_equal(length(obj$args), 2)
    expect_equal(length(obj$data), 2)
    
    # errors
    expect_error(
        ternaryop.LazyTensor(x_i, y_j, z, "Clamp"), 
        paste(
            "`", "z", 
            "` input argument should be a LazyTensor, a ComplexLazyTensor,",
            " a vector or a scalar.",
            "\nIf you want to use a matrix, convert it to LazyTensor first.", 
            sep = ""
        ), 
        fixed = TRUE
    )
    
    expect_error(
        ternaryop.LazyTensor(x, y_j, z, "Clamp"), 
        paste(
            "`", "x", 
            "` input argument should be a LazyTensor, a ComplexLazyTensor,",
            " a vector or a scalar.",
            "\nIf you want to use a matrix, convert it to LazyTensor first.", 
            sep = ""
        ), 
        fixed = TRUE
    )
    
    expect_error(
        ternaryop.LazyTensor(4, y_j, z_i, "Clamp", dim_res = 3.14), 
        paste(
            "If not NA, `dim_res` input argument should be an integer. ",
            "Received 3.14.",
            sep = ""
        ), 
        fixed = TRUE
    )
    
})



# TEST TYPE CHECKING ===========================================================


test_that("is.LazyTensor", {
    # basic example
    D <- 3
    M <- 100
    x <- matrix(runif(M * D), M, D)
    x_i <- LazyTensor(x, index = 'i')
    p <- LazyTensor(runif(3, 0, 1)) # fixed vector parameter across indices
    l <- LazyTensor(314)            # fixed scalar parameter across indices
    z <- matrix(1i^(-6:5), nrow = 4)                     # complex 4x3 matrix
    z_i <- LazyTensor(z, index = 'i', is_complex = TRUE) # ComplexLazyTensor
    
    # check results
    expect_true(is.LazyTensor(x_i))
    expect_true(is.LazyTensor(p))
    expect_true(is.LazyTensor(l))
    expect_false(is.LazyTensor(x))
    expect_false(is.LazyTensor(D))
    expect_true(is.LazyTensor(z_i)) # a ComplexLazyTensor is a LazyTensor
})


test_that("is.ComplexLazyTensor", {
    # basic example
    D <- 3
    M <- 100
    x <- matrix(runif(M * D), M, D)
    x_i <- LazyTensor(x, index = 'i')
    p <- LazyTensor(runif(3, 0, 1)) # fixed vector parameter across indices
    l <- LazyTensor(314)            # fixed scalar parameter across indices
    z <- matrix(1i^(-6:5), nrow = 4)                     # complex 4x3 matrix
    z_i <- LazyTensor(z, index = 'i', is_complex = TRUE) # ComplexLazyTensor
    
    # check results
    expect_false(is.ComplexLazyTensor(x_i))
    expect_false(is.ComplexLazyTensor(p))
    expect_false(is.ComplexLazyTensor(l))
    expect_true(is.ComplexLazyTensor(z_i))
})


test_that("is.LazyParameter", {
    # basic example
    scal <- 3.14
    cplx <- 2 + 3i
    v <- rep(3, 10)
    x <- matrix(runif(100 * 3), 100, 3)
    
    scal_LT <- LazyTensor(scal)
    cplx_LT <- LazyTensor(cplx)
    v_LT <- LazyTensor(v)
    x_i <- LazyTensor(x, index = 'i')
    
    # check results
    expect_true(is.LazyParameter(scal_LT))
    expect_false(is.LazyParameter(cplx_LT))
    expect_false(is.LazyParameter(v_LT))
    expect_false(is.LazyParameter(x_i))
    
    # errors
    expect_error(is.LazyParameter(x), 
                 "`x` input must be a LazyTensor.",
                 fixed = TRUE
    )
    expect_error(is.LazyParameter(7), 
                 "`x` input must be a LazyTensor.",
                 fixed = TRUE
    )
})


test_that("is.ComplexLazyParameter", {
    # basic example
    scal <- 3.14
    cplx <- 2 + 3i
    v <- rep(3 + 7i, 10)
    z <- matrix(2 + 1i^ (-6:5), nrow = 4)
    x <- matrix(runif(100 * 3), 100, 3)
    
    scal_LT <- LazyTensor(scal)
    cplx_LT <- LazyTensor(cplx)
    v_LT <- LazyTensor(v)
    z_i <- LazyTensor(z, index = 'i')
    x_i <- LazyTensor(x, index = 'i')
    
    # check results
    expect_false(is.ComplexLazyParameter(scal_LT))
    expect_true(is.ComplexLazyParameter(cplx_LT))
    expect_false(is.ComplexLazyParameter(v_LT))
    expect_false(is.ComplexLazyParameter(x_i))
    
    # errors
    expect_error(is.ComplexLazyParameter(x), 
                 "`x` input must be a LazyTensor or a ComplexLazyTensor.",
                 fixed = TRUE
    )
    expect_error(is.ComplexLazyParameter(7), 
                 "`x` input must be a LazyTensor or a ComplexLazyTensor.",
                 fixed = TRUE
    )
    
})


test_that("is.LazyVector", {
    # basic example
    scal <- 3.14
    cplx <- 2 + 3i
    v <- rep(3, 10)
    x <- matrix(runif(100 * 3), 100, 3)
    
    scal_LT <- LazyTensor(scal)
    cplx_LT <- LazyTensor(cplx)
    v_LT <- LazyTensor(v)
    x_i <- LazyTensor(x, index = 'i')
    
    # check results
    expect_true(is.LazyVector(scal_LT))
    expect_true(is.LazyVector(cplx_LT))
    expect_true(is.LazyVector(v_LT))
    expect_false(is.LazyVector(x_i))
    
    # errors
    expect_error(is.LazyVector(v), 
                 "`x` input must be a LazyTensor or a ComplexLazyTensor.",
                 fixed = TRUE
    )
    expect_error(is.LazyVector(7), 
                 "`x` input must be a LazyTensor or a ComplexLazyTensor.",
                 fixed = TRUE
    )
})


test_that("is.LazyMatrix", {
    # basic example
    scal <- 3.14
    cplx <- 2 + 3i
    v <- rep(3, 10)
    x <- matrix(runif(100 * 3), 100, 3)
    
    scal_LT <- LazyTensor(scal)
    cplx_LT <- LazyTensor(cplx)
    v_LT <- LazyTensor(v)
    x_i <- LazyTensor(x, index = 'i')
    
    # check results
    expect_false(is.LazyMatrix(scal_LT))
    expect_false(is.LazyMatrix(cplx_LT))
    expect_false(is.LazyMatrix(v_LT))
    expect_true(is.LazyMatrix(x_i))
    
    # errors
    expect_error(is.LazyMatrix(x), 
                 "`x` input must be a LazyTensor or a ComplexLazyTensor.",
                 fixed = TRUE
    )
    expect_error(is.LazyMatrix(7), 
                 "`x` input must be a LazyTensor or a ComplexLazyTensor.",
                 fixed = TRUE
    )
})

test_that("is.int", {
    # basic example
    A <- 3
    B <- 3.0
    C <- 3.14
    D <- rep(3, 10)
    E <- 2 + 3i
    
    # check results
    expect_true(is.int(A))
    expect_true(is.int(B))
    expect_false(is.int(C))
    expect_false(is.int(D))
    expect_false(is.int(E))
    expect_false(is.int(LazyTensor(4)))
})



# TEST GLOBAL CHECKS ===========================================================


# Test get and check dimensions
test_that("get_inner_dim", {
    # basic example
    D <- 3
    E <- 7
    M <- 100
    N <- 150
    x <- matrix(runif(M * D), M, D)
    y <- matrix(runif(N * D), N, D)
    t <- matrix(runif(N * E), N, E)
    x_i <- LazyTensor(x, index = 'i')
    y_j <- LazyTensor(y, index = 'j')
    t_j <- LazyTensor(t, index = 'j')
    p <- LazyTensor(runif(3, 0, 1)) # fixed vector parameter across indices
    l <- LazyTensor(314)            # fixed scalar parameter across indices
    z <- matrix(1i^(-6:5), nrow = 4)                     # complex 4x3 matrix
    z_i <- LazyTensor(z, index = 'i', is_complex = TRUE) # ComplexLazyTensor
    
    # check results
    expect_equal(get_inner_dim(x_i), 3)
    expect_equal(get_inner_dim(y_j), 3)
    expect_equal(get_inner_dim(t_j), 7)
    expect_equal(get_inner_dim(p), 3)
    expect_equal(get_inner_dim(l), 1)
    expect_equal(get_inner_dim(z_i), 3)
    
    # errors
    expect_error(
        get_inner_dim(x),
        "`x` input argument should be a LazyTensor or a ComplexLazyTensor.",
        fixed = TRUE
    )
})


test_that("check_inner_dim", {
    # basic example
    D <- 3
    E <- 7
    M <- 100
    N <- 150
    x <- matrix(runif(M * D), M, D)
    y <- matrix(runif(N * D), N, D)
    t <- matrix(runif(N * E), N, E)
    x_i <- LazyTensor(x, index = 'i')
    y_j <- LazyTensor(y, index = 'j')
    t_j <- LazyTensor(t, index = 'j')
    p <- LazyTensor(runif(3, 0, 1)) # fixed vector parameter across indices
    l <- LazyTensor(314)            # fixed scalar parameter across indices
    
    z <- matrix(1i^(-6:5), nrow = 4) # complex 4x3 matrix
    z_i <- LazyTensor(z, index = 'i', is_complex = TRUE) # ComplexLazyTensor
    
    # check results
    ## with two inputs
    expect_true(check_inner_dim(x_i, y_j, check_type = "sameor1"))
    expect_false(check_inner_dim(x_i, t_j, check_type = "sameor1"))
    expect_true(check_inner_dim(x_i, p, check_type = "sameor1"))
    expect_true(check_inner_dim(x_i, l, check_type = "sameor1"))
    
    expect_true(check_inner_dim(x_i, y_j, check_type = "same"))
    expect_false(check_inner_dim(x_i, l, check_type = "same"))
    
    ## with three inputs
    expect_true(check_inner_dim(x_i, p, z_i, check_type = "sameor1"))
    expect_true(check_inner_dim(x_i, l, z_i, check_type = "sameor1"))
    
    expect_true(check_inner_dim(x_i, y_j, z_i, check_type = "same"))
    expect_false(check_inner_dim(x_i, t_j, z_i, check_type = "same"))
    
    # errors
    expect_error(
        check_inner_dim(x, y_j),
        "Input arguments should be of class 'LazyTensor' or 'ComplexLazyTensor'.",
        fixed = TRUE
    )
    expect_error(
        check_inner_dim(x_i, y_j, z),
        "Input arguments should be of class 'LazyTensor' or 'ComplexLazyTensor'.",
        fixed = TRUE
    )
})


test_that("check_index", {
    expect_type(check_index("i"), "logical")
    
    expect_true(check_index("i"))
    expect_true(check_index("j"))
    
    expect_false(check_index(5))
    expect_false(check_index("n"))
})


test_that("index_to_int", {
    checkmate::expect_integerish(index_to_int("i"))
    
    expect_equal(index_to_int("i"), 0)
    expect_equal(index_to_int("j"), 1)
})


# TEST REDUCTION-RELATED PREPROCESS FUNCTIONS ==================================

test_that("identifier", {
    # basic examples
    D <- 3
    E <- 7
    M <- 100
    N <- 150
    x <- matrix(runif(M * D), M, D)
    y <- matrix(runif(N * D), N, D)
    x_i <- LazyTensor(x, index = 'i')
    y_j <- LazyTensor(y, index = 'j')
    p <- LazyTensor(runif(3, 0, 1)) # fixed vector parameter across indices
    z <- matrix(1i^(-6:5), nrow = 4)                     # complex 4x3 matrix
    z_i <- LazyTensor(z, index = 'i', is_complex = TRUE) # ComplexLazyTensor
    
    # check results
    bool_grep_args_i <- grep("A0x.*i", identifier(x_i$args))
    expect_equal(bool_grep_args_i, 1)
    bool_grep_args_j <- grep("A0x.*j", identifier(y_j$args))
    expect_equal(bool_grep_args_j, 1)
    bool_grep_args_NA <- grep("A0x.*NA", identifier(p$args))
    expect_equal(bool_grep_args_NA, 1)
    bool_grep_args_z_i <- grep("A0x.*i", identifier(z_i$args))
    expect_equal(bool_grep_args_z_i, 1)
    
    # errors
    expect_error(identifier(x_i),
                 "`arg` input argument should be a character string.",
                 fixed = TRUE)
})


test_that("fix_variables", {
    # basic example
    D <- 3
    E <- 7
    M <- 100
    N <- 150
    x <- matrix(runif(M * D), M, D)
    y <- matrix(runif(N * D), N, D)
    x_i <- LazyTensor(x, index = 'i')
    x_j <- LazyTensor(x, index = 'j')
    y_j <- LazyTensor(y, index = 'j')
    p <- LazyTensor(runif(3, 0, 1)) # fixed vector parameter across indices
    l <- LazyTensor(314)            # fixed scalar parameter across indices
    
    # weird formulae for verification purpose
    fix_expr1 <- fix_variables(x_i + y_j + x_i + x_j)
    fix_expr2 <- fix_variables(exp(sqdist(x_i, y_j)^l) - x_i*x_j)
    fix_expr3 <- fix_variables(norm2(y_j) + (p|x_i)*l)
    fix_expr4 <- fix_variables(clamp(x_i, x_j, y_j))
    
    # check classes
    expect_true(is.LazyTensor(fix_variables(l)))
    expect_true(is.LazyTensor(fix_variables(x_i)))
    expect_true(is.LazyTensor(fix_expr1))
    expect_true(is.LazyTensor(fix_expr2))
    expect_true(is.LazyTensor(fix_expr3))
    expect_true(is.LazyTensor(fix_expr4))
    
    # check formulae
    expect_equal(fix_variables(l)$formula,
                 "IntCst(314)")
    expect_equal(fix_variables(l + l*l)$formula,
                 "IntCst(314)+IntCst(314)*IntCst(314)")
    expect_equal(fix_variables(x_i)$formula,
                 "V0")
    expect_equal(fix_expr1$formula,
                 "V0+V1+V0+V2")
    expect_equal(fix_expr2$formula,
                 "Exp(Powf(SqDist(V0,V1),IntCst(314)))-V0*V2")
    expect_equal(fix_expr3$formula,
                 "Norm2(V0)+(V1|V2)*IntCst(314)")
    expect_equal(fix_expr4$formula,
                 "Clamp(V0,V1,V2)")
    
    # check args
    expect_equal(fix_variables(x_i)$args, "V0=Vi(3)")
    expect_equal(fix_expr2$args[1], "V0=Vi(3)")
    expect_equal(fix_expr2$args[2], "V1=Vj(3)")
    expect_equal(fix_expr2$args[3], "V2=Vj(3)")
    expect_equal(fix_expr3$args[1], "V0=Vj(3)")
    expect_equal(fix_expr3$args[2], "V1=Pm(3)")
    expect_equal(fix_expr3$args[3], "V2=Vi(3)")
    
    # errors
    expect_error(fix_variables(x),
                 "`x` input must be a LazyTensor or a ComplexLazyTensor.",
                 fixed = TRUE)
})


test_that("fix_op_reduction", {
    
    op <- "Sum"
    expected_res <- op
    expect_warning(res <- fix_op_reduction(op))
    expect_equal(res, expected_res)
    expect_warning(res <- fix_op_reduction(op, with_weight = TRUE))
    expect_equal(res, expected_res)
    
    op <- "Argmin"
    expected_res <- op
    expect_warning(res <- fix_op_reduction(op))
    expect_equal(res, expected_res)
    expect_warning(res <- fix_op_reduction(op, with_weight = TRUE))
    expect_equal(res, expected_res)
    
    op <- "Min_Argmin"
    expected_res <- op
    expect_warning(res <- fix_op_reduction(op))
    expect_equal(res, expected_res)
    expect_warning(res <- fix_op_reduction(op, with_weight = TRUE))
    expect_equal(res, expected_res)
    
    op <- "LogSumExp"
    expected_res <- "Max_SumShiftExp"
    expect_warning(res <- fix_op_reduction(op))
    expect_equal(res, expected_res)
    expected_res <- "Max_SumShiftExpWeight"
    expect_warning(res <- fix_op_reduction(op, with_weight = TRUE))
    expect_equal(res, expected_res)
    
    op <- "SumSoftMaxWeight"
    expected_res <- "Max_SumShiftExpWeight"
    expect_warning(res <- fix_op_reduction(op))
    expect_equal(res, expected_res)
    expect_warning(res <- fix_op_reduction(op, with_weight = TRUE))
    expect_equal(res, expected_res)
    
})


test_that("preprocess_reduction", {
    
    x <- matrix(runif(10), 5, 2)
    x_i <- LazyTensor(x, index = "i")
    opstr <- "Sum"
    index <- "i"
    opt_arg <- NULL
    
    op <- preprocess_reduction(x_i, opstr, index, opt_arg)
    expect_equal(
        op(),
        list(
            formula = "Sum_Reduction(V0,0)", args = "V0=Vi(2)",
            args_info = list(
                args = "V0=Vi(2)", var_name = "V0", var_type = "Vi", 
                var_pos = 0, var_dim = 2, decl = "dim"), 
            sum_scheme = "auto"
        )
    )
    res <- op(list(x))
    expected_res <- apply(x, 2, sum)
    expect_equal(as.vector(res), expected_res, tolerance = 1e-5)
    
    
    x <- matrix(runif(10), 5, 2)
    x_i <- LazyTensor(x, index = "i")
    opstr <- "Min_ArgMin"
    index <- "i"
    opt_arg <- NULL
    op <- preprocess_reduction(x_i, opstr, index, opt_arg)
    expect_equal(
        op(),
        list(
            formula = "Min_ArgMin_Reduction(V0,0)", args = "V0=Vi(2)",
            args_info = list(
                args = "V0=Vi(2)", var_name = "V0", var_type = "Vi", 
                var_pos = 0, var_dim = 2, decl = "dim"), 
            sum_scheme = "auto"
        )
    )
    res <- op(list(x))
    expected_res <- list(
        apply(x, 2, min),
        apply(x, 2, which.min) - 1
    )
    expect_equal(lapply(res, as.vector), expected_res, tolerance = 1e-5)
})

test_that("cplx_warning", {
    expect_warning(cplx_warning(TRUE))
    
    cplx_warning(FALSE) # should not produce warning
})
