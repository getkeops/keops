context("LazyTensor operations")


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
  out_i <- LazyTensor(x, index = 'i')
  out_j <- LazyTensor(x, index = 'j')
  out_u <- LazyTensor(u) # parameter vector
  out_D <- LazyTensor(D) # parameter scalar
  
  z <- matrix(1i^ (-6:5), nrow = 4)                     # create a complex 4x3 matrix
  z_i <- LazyTensor(z, index = 'i', is_complex = TRUE)  # create a ComplexLazyTensor
  still_good_z_Vi <- LazyTensor(z, index = 'i') # without specifying "is_complex = TRUE": should work as well.
  
  # check the object class
  classes <- c(class(out_i), class(out_j), class(out_u), class(out_D))
  k <- length(classes)
  expect_equal(classes, rep("LazyTensor", k))
  
  complex_class <- c(class(z_i), class(still_good_z_Vi))
  expect_equal(complex_class, rep("ComplexLazyTensor", 2))
  
  # check object formula and args
  bool_grep_i <- grep("A0x.*i", out_i$formula)
  expect_equal(bool_grep_i, 1)
  bool_grep_j <- grep("A0x.*j", out_j$formula)
  expect_equal(bool_grep_j, 1)
  bool_grep_NA <- grep("A0x.*NA", out_u$formula)
  expect_equal(bool_grep_NA, 1)
  bool_grep_zi <- grep("A0x.*i", z_i$formula)
  expect_equal(bool_grep_zi, 1)
  bool_grep_Pm <- grep("IntCst\\(3\\)=Pm\\(1\\)", out_D$args)
  expect_equal(bool_grep_Pm, 1)
  bool_grep_zi_args <- grep("A0x.*i=Vi\\(6\\)", z_i$args)
  expect_equal(bool_grep_zi_args, 1)
  
  # errors
  expect_error(LazyTensor("x"), 
               "`x` input argument should be a matrix, a vector or a scalar.", 
               fixed = TRUE)
  expect_error(LazyTensor(x), 
               "missing `index` argument.", 
               fixed = TRUE)
  expect_error(LazyTensor(u, index = "i"), 
               "`index` must be NA with a vector or a scalar value.", 
               fixed = TRUE)
  expect_error(LazyTensor(D, index = "i"), 
               "`index` must be NA with a vector or a scalar value.", 
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
  still_good_z_Vi <- Vi(z) # without specifying "is_complex = TRUE": should work as well.
  
  # check arguments 
  expect_true(x_i$args == x_Vi$args)
  expect_true(z_i$args == z_Vi$args)
  expect_true(z_i$args == still_good_z_Vi$args)
  expect_true(class(x_Vi) == "LazyTensor")
  expect_true(class(z_Vi) == "ComplexLazyTensor")
  expect_true(class(still_good_z_Vi) == "ComplexLazyTensor")
  
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
  x_Vj<- Vj(x)
  
  # ComplexLazyTensor
  z_j <- LazyTensor(z, index = 'j', is_complex = TRUE)
  z_Vj <- Vj(z, is_complex = TRUE)
  still_good_z_Vj <- Vj(z) # without specifying "is_complex = TRUE": should work as well.
  
  # check arguments 
  expect_true(x_j$args == x_Vj$args)
  expect_true(z_j$args == z_Vj$args)
  expect_true(z_j$args == still_good_z_Vj$args)
  expect_true(class(x_Vj) == "LazyTensor")
  expect_true(class(z_Vj) == "ComplexLazyTensor")
  expect_true(class(still_good_z_Vj) == "ComplexLazyTensor")
  
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
  u <- runif(M, min = 0, max = 3.14)
  z <- rep(1i^(-6:5), 4) # complex vector
  
  D_LT <- LazyTensor(D)
  D_Pm <- Pm(D)
  z_LT <- LazyTensor(z)
  z_Pm <- Pm(z)
  # check arguments 
  expect_true(D_LT$args == D_Pm$args)
  expect_true(class(D_Pm) == "LazyTensor")
  
  
  u_LT <- LazyTensor(u)
  u_Pm <- Pm(u)
  # check arguments 
  expect_true(u_LT$args == u_Pm$args)
  expect_true(class(u_Pm) == "LazyTensor")
  
  # errors
  expect_error(Pm(x_i), 
               "`x` input is already a LazyTensor.", 
               fixed = TRUE)
  expect_error(Pm(x), 
               "`x` input must be a scalar or a vector.", 
               fixed = TRUE)
  expect_error(Pm("a"), 
               "`x` input must be a scalar or a vector.", 
               fixed = TRUE)
  
})


test_that("unaryop.LazyTensor", {
  D <- 3
  M <- 100
  x <- matrix(runif(M * D), M, D)
  x_i <- LazyTensor(x, index = 'i')
  
  # check formulas, args & classes
  obj <- unaryop.LazyTensor(x_i, "Square")
  bool_grep_formula <- grep("Square\\(A0x.*i\\)", obj$formula)
  bool_grep_args <- grep("A0x.*i=Vi\\(3\\)", obj$args[1])
  expect_equal(bool_grep_formula, 1)
  expect_equal(bool_grep_args, 1)
  expect_is(obj, "LazyTensor")
  
  # errors
  expect_error(unaryop.LazyTensor(x, "Square"), 
               paste("`x` input argument should be a LazyTensor, a vector or a scalar.",
                     "\nIf you want to use a matrix, convert it to LazyTensor first.", sep = ""), 
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
  y_j <- LazyTensor(y, index = 'j')
  z_j <- LazyTensor(z, index = 'j')
  
  # check formulas, args & classes
  obj <-  binaryop.LazyTensor(x_i, y_j, "Sum")
  bool_grep_formula <- grep("Sum\\(A0x.*i,A0x.*j\\)", obj$formula)
  bool_grep_args1 <- grep("A0x.*i=Vi\\(3\\)", obj$args[1])
  bool_grep_args2 <- grep("A0x.*j=Vj\\(3\\)", obj$args[2])
  expect_equal(bool_grep_formula, 1)
  expect_equal(length(obj$args), 2)
  expect_equal(bool_grep_args1, 1)
  expect_equal(bool_grep_args2, 1)
  expect_is(obj, "LazyTensor")
  
  obj <-  binaryop.LazyTensor(x_i, y_j, "-", is_operator = TRUE)
  bool_grep_formula <- grep("A0x.*i-A0x.*j", obj$formula)
  bool_grep_args1 <- grep("A0x.*i=Vi\\(3\\)", obj$args[1])
  bool_grep_args2 <- grep("A0x.*j=Vj\\(3\\)", obj$args[2])
  expect_equal(bool_grep_formula, 1)
  expect_equal(bool_grep_args1, 1)
  expect_equal(bool_grep_args2, 1)
  expect_is(obj, "LazyTensor")
  
  obj <-  binaryop.LazyTensor(x_i, 3, "Powf")
  bool_grep_formula <- grep("Powf\\(A0x.*i,A0x.*NA\\)", obj$formula)
  bool_grep_args1 <- grep("A0x.*i=Vi\\(3\\)", obj$args[1])
  expect_equal(bool_grep_formula, 1)
  expect_equal(length(obj$args), 2)
  expect_is(obj, "LazyTensor")
  
  # errors
  expect_error(binaryop.LazyTensor(x, y_j, "+"), 
               paste("`x` input argument should be a LazyTensor, a vector or a scalar.",
                     "\nIf you want to use a matrix, convert it to LazyTensor first.", sep = ""), 
               fixed = TRUE)
  
  expect_error(binaryop.LazyTensor(x_i, z_j, "|", is_operator = TRUE, dim_check_type = "same"),
               "Operation `|` expects inputs of the same dimension. Received 3 and 7.",
               fixed = TRUE)
})


test_that("ternaryop.LazyTensor", {
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
  
  # check formulas, args & classes
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
  expect_is(obj, "LazyTensor")
  
  obj <-  ternaryop.LazyTensor(4, y_j, z_i, "Clamp")
  bool_grep_args <- grep("IntCst\\(4\\)=Pm\\(1\\)", obj$args[1])
  expect_equal(bool_grep_args, 1)
  
  # errors
  expect_error(ternaryop.LazyTensor(x_i, y_j, z, "Clamp"), 
               paste("`", "z", "` input argument should be a LazyTensor, a vector or a scalar.",
                     "\nIf you want to use a matrix, convert it to LazyTensor first.", sep = ""), 
               fixed = TRUE)
  expect_error(ternaryop.LazyTensor(x, y_j, z, "Clamp"), 
               paste("`", "x", "` input argument should be a LazyTensor, a vector or a scalar.",
                     "\nIf you want to use a matrix, convert it to LazyTensor first.", sep = ""), 
               fixed = TRUE)
})


# TEST PREPROCESS ==============================================================


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
  expect_false(is.LazyTensor(z_i))
})


# Test get and check dimensions
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
  expect_error(get_inner_dim(x),
               "`x` input argument should be a LazyTensor or a ComplexLazyTensor.",
               fixed = TRUE)
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
  expect_error(check_inner_dim(x, y_j),
               "Input arguments should be of class 'LazyTensor' or 'ComplexLazyTensor'.",
               fixed = TRUE)
  expect_error(check_inner_dim(x_i, y_j, z),
               "Input arguments should be of class 'LazyTensor' or 'ComplexLazyTensor'.",
               fixed = TRUE)
})



# TEST OPERATIONS ==============================================================


test_that("+", {
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
  
  expect_true(class(x_i + y_j) == "LazyTensor")
  expect_true(class(x_i + xc_i) == "ComplexLazyTensor")
  expect_true(class(xc_i + x_i) == "ComplexLazyTensor")
  expect_true(class(xc_i + yc_j) == "ComplexLazyTensor")
  
  obj <- x_i + y_j
  bool_grep_formula <- grep("A0x.*i\\+A0x.*j", obj$formula)
  expect_equal(bool_grep_formula, 1)
  
  obj <-  x_i + 3
  bool_grep_formula <- grep("A0x.*i\\+IntCst\\(3\\)", obj$formula)
  expect_equal(bool_grep_formula, 1)
  
  obj <-  3.14 + x_i
  bool_grep_formula <- grep("A0x.*NA\\+A0x.*i", obj$formula)
  expect_equal(bool_grep_formula, 1)
  
  # errors
  expect_error(x_i + z_j,
               "Operation `+` expects inputs of the same dimension or dimension 1. Received 3 and 7.",
               fixed = TRUE)
})


test_that("-", {
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
  
  expect_true(class(x_i - y_j) == "LazyTensor")
  expect_true(class(x_i - xc_i) == "ComplexLazyTensor")
  expect_true(class(xc_i - yc_j) == "ComplexLazyTensor")
  
  obj <- x_i - y_j
  bool_grep_formula <- grep("A0x.*i-A0x.*j", obj$formula)
  expect_equal(bool_grep_formula, 1)
  
  obj <-  x_i - 3
  bool_grep_formula <- grep("A0x.*i-+IntCst\\(3\\)", obj$formula)
  expect_equal(bool_grep_formula, 1)
  
  obj <-  -x_i
  bool_grep_formula <- grep("Minus\\(A0x.*i\\)", obj$formula)
  expect_equal(bool_grep_formula, 1)
  
  obj <-  3.14 - x_i
  bool_grep_formula <- grep("A0x.*NA-A0x.*i", obj$formula)
  expect_equal(bool_grep_formula, 1)
  
  # errors
  expect_error(x_i - z_j,
               "Operation `-` expects inputs of the same dimension or dimension 1. Received 3 and 7.",
               fixed = TRUE)
})


test_that("*", {
  # basic example
  D <- 3
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
  
  expect_true(class(x_i * y_j) == "LazyTensor")
  expect_true(class(x_i * yc_j) == "ComplexLazyTensor")
  expect_true(class(xc_i * y_j) == "ComplexLazyTensor")
  expect_true(class(xc_i * yc_j) == "ComplexLazyTensor")
  
  obj <- x_i * y_j
  bool_grep_formula <- grep("A0x.*i\\*A0x.*j", obj$formula)
  expect_equal(bool_grep_formula, 1)
  
  obj <-  x_i * 3
  bool_grep_formula <- grep("A0x.*i\\*+IntCst\\(3\\)", obj$formula)
  expect_equal(bool_grep_formula, 1)
  
  obj <-  3.14 * x_i
  bool_grep_formula <- grep("A0x.*NA\\*A0x.*i", obj$formula)
  expect_equal(bool_grep_formula, 1)
  
  # errors
  expect_error(x_i * z_j,
               "Operation `*` expects inputs of the same dimension or dimension 1. Received 3 and 7.",
               fixed = TRUE)
})


test_that("/", {
  # basic example
  D <- 3
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
  
  expect_true(class(x_i / y_j) == "LazyTensor")
  expect_true(class(x_i / yc_j) == "ComplexLazyTensor")
  expect_true(class(xc_i / y_j) == "ComplexLazyTensor")
  expect_true(class(xc_i / yc_j) == "ComplexLazyTensor")
  
  obj <- x_i / y_j
  bool_grep_formula <- grep("A0x.*i/A0x.*j", obj$formula)
  expect_equal(bool_grep_formula, 1)
  
  obj <-  x_i / 3
  bool_grep_formula <- grep("A0x.*i/IntCst\\(3\\)", obj$formula)
  expect_equal(bool_grep_formula, 1)
  
  obj <-  3.14 / x_i
  bool_grep_formula <- grep("A0x.*NA/A0x.*i", obj$formula)
  expect_equal(bool_grep_formula, 1)
  
  # errors
  expect_error(x_i / z_j,
               "Operation `/` expects inputs of the same dimension or dimension 1. Received 3 and 7.",
               fixed = TRUE)
})


test_that("^", {
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
  
  expect_is(x_i^y_j, "LazyTensor")
  expect_is(x_i^3, "LazyTensor")
  expect_is(x_i^yc_j, "ComplexLazyTensor")
  expect_is(xc_i^y_j, "ComplexLazyTensor")
  expect_is(xc_i^yc_j, "ComplexLazyTensor")
  expect_is(xc_i^2, "ComplexLazyTensor")
  expect_is(xc_i^3, "ComplexLazyTensor")
  expect_is(xc_i^0.5, "ComplexLazyTensor")
  
  
  obj <- x_i^y_j
  bool_grep_formula <- grep("Powf\\(A0x.*i,A0x.*j\\)", obj$formula)
  expect_equal(bool_grep_formula, 1)
  
  obj <-  x_i^3
  bool_grep_formula <- grep("Pow\\(A0x.*i,3\\)", obj$formula)
  expect_equal(bool_grep_formula, 1)
  
  obj <-  x_i^(-0.5)
  bool_grep_formula <- grep("Rsqrt\\(A0x.*i\\)", obj$formula)
  expect_equal(bool_grep_formula, 1)
  
  obj <-  x_i^(0.5)
  bool_grep_formula <- grep("Sqrt\\(A0x.*i\\)", obj$formula)
  expect_equal(bool_grep_formula, 1)
  
  obj <-  y_j^2
  bool_grep_formula <- grep("Square\\(A0x.*j\\)", obj$formula)
  expect_equal(bool_grep_formula, 1)
  
  obj <-  x_i^0.314
  bool_grep_formula <- grep("Powf\\(A0x.*i,A0x.*NA\\)", obj$formula)
  expect_equal(bool_grep_formula, 1)
  
  obj <- 3.14^x_i
  bool_grep_formula <- grep("Powf\\(A0x.*NA,A0x.*i\\)", obj$formula)
  expect_equal(bool_grep_formula, 1)
  
  
  # TODO
  expect_error(2i^x_i, 
               "Input arguments should be of class 'LazyTensor' or 'ComplexLazyTensor'.",
               fixed = TRUE)
  
  expect_error(x_i^2i, 
               "Input arguments should be of class 'LazyTensor' or 'ComplexLazyTensor'.",
               fixed = TRUE)
})


test_that("square", {
  D <- 3
  M <- 100
  N <- 150
  x <- matrix(runif(M * D), M, D)
  x_i <- LazyTensor(x, index = 'i')
  xc_i <- LazyTensor(x, index = 'i', is_complex = TRUE)

  expect_true(square(6) == 36)
  expect_is(square(x_i), "LazyTensor")
  expect_is(square(xc_i), "ComplexLazyTensor")
  
  obj <-  square(x_i)
  bool_grep_formula <- grep("Square\\(A0x.*i\\)", obj$formula)
  expect_equal(bool_grep_formula, 1)
  
})


test_that("sqrt", {
  D <- 3
  M <- 100
  N <- 150
  x <- matrix(runif(M * D), M, D)
  x_i <- LazyTensor(x, index = 'i')
  xc_i <- LazyTensor(x, index = 'i', is_complex = TRUE)
  
  expect_true(sqrt(36) == 6)
  expect_is(sqrt(x_i), "LazyTensor")
  expect_is(sqrt(xc_i), "ComplexLazyTensor")
  
  obj <-  sqrt(x_i)
  bool_grep_formula <- grep("Sqrt\\(A0x.*i\\)", obj$formula)
  expect_equal(bool_grep_formula, 1)
  
})


test_that("rsqrt", {
  D <- 3
  M <- 100
  N <- 150
  x <- matrix(runif(M * D), M, D)
  x_i <- LazyTensor(x, index = 'i')
  xc_i <- LazyTensor(x, index = 'i', is_complex = TRUE)
  
  expect_true(rsqrt(4) == 0.5)
  expect_is(rsqrt(x_i), "LazyTensor")
  expect_is(rsqrt(xc_i), "ComplexLazyTensor")
  
  obj <-  rsqrt(x_i)
  bool_grep_formula <- grep("Rsqrt\\(A0x.*i\\)", obj$formula)
  expect_equal(bool_grep_formula, 1)
  
})


test_that("|", {
  # basic example
  D <- 3
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
  bool_grep_formula <- grep("\\(A0x.*i\\|A0x.*j\\)", obj$formula)
  expect_equal(bool_grep_formula, 1)
  
  # errors
  expect_error(x_i | z_j,
               "Operation `|` expects inputs of the same dimension. Received 3 and 7.",
               fixed = TRUE)
  
  expect_error(x_i | 3,
               "Operation `|` expects inputs of the same dimension. Received 3 and 1.",
               fixed = TRUE)
})


test_that("%*%", {
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

  # check classes
  expect_true(is.matrix(x_i %*% y_j))
  #expect_true(is.matrix(xc_i %*% yc_j))

})


test_that("exp", {
  # basic example
  D <- 3
  M <- 100
  N <- 150
  x <- matrix(runif(M * D), M, D)
  y <- matrix(runif(N * D), N, D)
  x_i <- LazyTensor(x, index = 'i')
  y_j <- LazyTensor(y, index = 'j')
  # basic example with complex exponential
  z <- matrix(1i^(-6:5), nrow = 4)                      # create a complex 4x3 matrix
  z_i <- LazyTensor(z, index = 'i', is_complex = TRUE)  # create a ComplexLazyTensor
  
  # check results, formulas & classes
  expect_equal(exp(0), 1)
  expect_true(class(exp(x_i)) == "LazyTensor")
  expect_true(class(exp(z_i)) == "ComplexLazyTensor")
  
  obj <- exp(x_i)
  bool_grep_formula <- grep("Exp\\(A0x.*i\\)", obj$formula)
  expect_equal(bool_grep_formula, 1)
  
  obj <- exp(x_i - y_j)
  bool_grep_formula <- grep("Exp\\(A0x.*i-A0x.*j\\)", obj$formula)
  expect_equal(bool_grep_formula, 1)
  
  obj <- exp(z_i)
  bool_grep_formula <- grep("ComplexExp\\(A0x.*i\\)", obj$formula)
  expect_equal(bool_grep_formula, 1)
})


test_that("log", {
  # basic example
  D <- 3
  M <- 100
  N <- 150
  x <- matrix(runif(M * D), M, D)
  y <- matrix(runif(N * D), N, D)
  x_i <- LazyTensor(x, index = 'i')
  y_j <- LazyTensor(y, index = 'j')
  # basic example with complex exponential
  z <- matrix(1i^(-6:5), nrow = 4)                      # create a complex 4x3 matrix
  z_i <- LazyTensor(z, index = 'i', is_complex = TRUE)  # create a ComplexLazyTensor
  
  # check results, formulas & classes
  expect_equal(log(1), 0)
  expect_true(!is.LazyTensor(log(x)))
  expect_true(is.LazyTensor(log(x_i)))
  expect_true(is.ComplexLazyTensor(log(z_i)))
  
  obj <- log(x_i)
  bool_grep_formula <- grep("Log\\(A0x.*i\\)", obj$formula)
  expect_equal(bool_grep_formula, 1)
  
  obj <-  log(x_i - y_j)
  bool_grep_formula <- grep("Log\\(A0x.*i-A0x.*j\\)", obj$formula)
  expect_equal(bool_grep_formula, 1)
  
  obj <- log(z_i)
  bool_grep_formula <- grep("Log\\(A0x.*i\\)", obj$formula)
  expect_equal(bool_grep_formula, 1)
})


test_that("inv", {
  # basic example
  D <- 3
  M <- 100
  N <- 150
  x <- matrix(runif(M * D), M, D)
  y <- matrix(runif(N * D), N, D)
  x_i <- LazyTensor(x, index = 'i')
  y_j <- LazyTensor(y, index = 'j')
  # basic example with complex exponential
  z <- matrix(1i^(-6:5), nrow = 4)                      # create a complex 4x3 matrix
  z_i <- LazyTensor(z, index = 'i', is_complex = TRUE)  # create a ComplexLazyTensor
  
  # check results, formulas & classes
  expect_equal(inv(1), 1)
  expect_true(!is.LazyTensor(inv(x)))
  expect_true(is.LazyTensor(inv(x_i)))
  expect_true(is.ComplexLazyTensor(inv(z_i)))
  
  obj <- inv(x_i)
  bool_grep_formula <- grep("Inv\\(A0x.*i\\)", obj$formula)
  expect_equal(bool_grep_formula, 1)
  
  obj <-  inv(x_i + y_j)
  bool_grep_formula <- grep("Inv\\(A0x.*i\\+A0x.*j\\)", obj$formula)
  expect_equal(bool_grep_formula, 1)
  
  obj <-  inv(z_i)
  bool_grep_formula <- grep("Inv\\(A0x.*i\\)", obj$formula)
  expect_equal(bool_grep_formula, 1)
})


test_that("cos", {
  # basic example
  D <- 3
  M <- 100
  N <- 150
  x <- matrix(runif(M * D), M, D)
  y <- matrix(runif(N * D), N, D)
  x_i <- LazyTensor(x, index = 'i')
  y_j <- LazyTensor(y, index = 'j')
  # basic example with complex exponential
  z <- matrix(1i^(-6:5), nrow = 4)                      # create a complex 4x3 matrix
  z_i <- LazyTensor(z, index = 'i', is_complex = TRUE)  # create a ComplexLazyTensor
  
  # check results, formulas & classes
  expect_equal(cos(0), 1)
  expect_true(!is.LazyTensor(x))
  expect_true(is.LazyTensor(cos(x_i)))
  expect_true(is.ComplexLazyTensor(cos(z_i)))
  
  obj <- cos(x_i)
  bool_grep_formula <- grep("Cos\\(A0x.*i\\)", obj$formula)
  expect_equal(bool_grep_formula, 1)
  
  obj <-  cos(x_i + y_j)
  bool_grep_formula <- grep("Cos\\(A0x.*i\\+A0x.*j\\)", obj$formula)
  expect_equal(bool_grep_formula, 1)
  
  obj <-  cos(z_i)
  bool_grep_formula <- grep("Cos\\(A0x.*i\\)", obj$formula)
  expect_equal(bool_grep_formula, 1)
})


test_that("sin", {
  # basic example
  D <- 3
  M <- 100
  N <- 150
  x <- matrix(runif(M * D), M, D)
  y <- matrix(runif(N * D), N, D)
  x_i <- LazyTensor(x, index = 'i')
  y_j <- LazyTensor(y, index = 'j')
  # basic example with complex exponential
  z <- matrix(1i^(-6:5), nrow = 4)                      # create a complex 4x3 matrix
  z_i <- LazyTensor(z, index = 'i', is_complex = TRUE)  # create a ComplexLazyTensor
  
  # check results, formulas & classes
  expect_equal(sin(0), 0)
  expect_true(!is.LazyTensor(sin(x)))
  expect_true(is.LazyTensor(sin(x_i)))
  expect_true(is.ComplexLazyTensor(sin(z_i)))
  
  obj <- sin(x_i)
  bool_grep_formula <- grep("Sin\\(A0x.*i\\)", obj$formula)
  expect_equal(bool_grep_formula, 1)
  
  obj <-  sin(x_i + y_j)
  bool_grep_formula <- grep("Sin\\(A0x.*i\\+A0x.*j\\)", obj$formula)
  expect_equal(bool_grep_formula, 1)
  
  obj <- sin(z_i)
  bool_grep_formula <- grep("Sin\\(A0x.*i\\)", obj$formula)
  expect_equal(bool_grep_formula, 1)
})


test_that("acos", {
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
  bool_grep_formula <- grep("Acos\\(A0x.*i\\)", obj$formula)
  expect_equal(bool_grep_formula, 1)
  
  obj <-  acos(x_i + y_j)
  bool_grep_formula <- grep("Acos\\(A0x.*i\\+A0x.*j\\)", obj$formula)
  expect_equal(bool_grep_formula, 1)
})


test_that("asin", {
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
  bool_grep_formula <- grep("Asin\\(A0x.*i\\)", obj$formula)
  expect_equal(bool_grep_formula, 1)
  
  obj <-  asin(x_i + y_j)
  bool_grep_formula <- grep("Asin\\(A0x.*i\\+A0x.*j\\)", obj$formula)
  expect_equal(bool_grep_formula, 1)
})


test_that("atan", {
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
  bool_grep_formula <- grep("Atan\\(A0x.*i\\)", obj$formula)
  expect_equal(bool_grep_formula, 1)
  
  obj <-  atan(x_i + y_j)
  bool_grep_formula <- grep("Atan\\(A0x.*i\\+A0x.*j\\)", obj$formula)
  expect_equal(bool_grep_formula, 1)
})


test_that("atan2", {
  # basic example
  D <- 3
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
  bool_grep_formula <- grep("Atan2\\(A0x.*i,A0x.*j\\)", obj$formula)
  expect_equal(bool_grep_formula, 1)
  
  # errors
  expect_error(atan2(x_i, z_j),
               "Operation `Atan2` expects inputs of the same dimension. Received 3 and 7.",
               fixed = TRUE)
})


test_that("%*%", {
  # basic example
  D <- 3
  M <- 100
  N <- 150
  x <- matrix(runif(M * D), M, D)
  y <- matrix(runif(N * D), N, D)
  x_i <- LazyTensor(x, index = 'i')
  y_j <- LazyTensor(y, index = 'j')
  
  # check results & formulas
  DM <- D %*% M
  expect_equal(DM[1], 300)
  
  obj <- x_i %*% y_j
  expect_equal(dim(obj)[1], 100)
  expect_equal(dim(obj)[2], 3)
  
  obj <-  x_i %*% y
  expect_equal(dim(obj)[1], 100)
  expect_equal(dim(obj)[2], 3)
  
  # TODO add an error in the %*% function when bad dimensions are used
  # TODO add test for `scalar %*% LazyTensor` and `matrix %*% LazyTensor` when problem will be fixed
  # TODO test reduction.LazyTensor and sum before
})


test_that("abs", {
  # basic example
  D <- 3
  M <- 100
  N <- 150
  x <- matrix(runif(M * D), M, D)
  y <- matrix(runif(N * D), N, D)
  x_i <- LazyTensor(x, index = 'i')
  y_j <- LazyTensor(y, index = 'j')
  
  # check results, formulas & classes
  expect_equal(abs(-D), 3)
  expect_true(class(abs(x))[1] != "LazyTensor")
  expect_true(class(abs(x_i)) == "LazyTensor")
  
  obj <- abs(x_i)
  bool_grep_formula <- grep("Abs\\(A0x.*i\\)", obj$formula)
  expect_equal(bool_grep_formula, 1)
  
  obj <-  abs(-y_j)
  bool_grep_formula <- grep("Abs\\(Minus\\(A0x.*j\\)\\)", obj$formula)
  expect_equal(bool_grep_formula, 1)
})


test_that("sign", {
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
  bool_grep_formula <- grep("Sign\\(A0x.*i\\)", obj$formula)
  expect_equal(bool_grep_formula, 1)
  
  obj <-  sign(-y_j)
  bool_grep_formula <- grep("Sign\\(Minus\\(A0x.*j\\)\\)", obj$formula)
  expect_equal(bool_grep_formula, 1)
})


test_that("round", {
  # basic example
  D <- 3
  M <- 100
  N <- 150
  x <- matrix(runif(M * D), M, D)
  y <- matrix(runif(N * D), N, D)
  x_i <- LazyTensor(x, index = 'i')
  y_j <- LazyTensor(y, index = 'j')
  
  # check results, formulas & class
  expect_equal(round(pi, 2), 3.14)
  expect_equal(class(round(pi, 2)) != "LazyTensor", TRUE)
  expect_equal(class(round(x_i, 3)) == "LazyTensor", TRUE)
  
  obj <- round(x_i, 3)
  bool_grep_formula <- grep("Round\\(A0x.*i,3\\)", obj$formula)
  expect_equal(bool_grep_formula, 1)
  
  obj <- round(y_j, 3.14)
  bool_grep_formula <- grep("Round\\(A0x.*j,3.14\\)", obj$formula)
  expect_equal(bool_grep_formula, 1)
  
  # errors
  expect_error(round(x_i, x), 
               "`d` input argument should be a scalar.", 
               fixed = TRUE)
})


test_that("min", {
  # basic example
  D <- 3
  M <- 100
  N <- 150
  x <- matrix(runif(M * D), M, D)
  x_i <- LazyTensor(x, index = 'i')

  # check results, formulas & classes
  expect_equal(min(D), 3)
  expect_true(class(min(x))[1] != "LazyTensor")
  expect_true(class(min(x_i, "i"))[1] != "LazyTensor")
  
  obj <- min(x_i)
  expect_true(class(obj) == "LazyTensor")
  bool_grep_formula <- grep("Min\\(A0x.*i\\)", obj$formula)
  expect_equal(bool_grep_formula, 1)
  
  # errors
  expect_error(min(x_i, "b"),
               "`index` input argument should be a character `i`, `j` or NA.",
               fixed = TRUE)
  
  expect_error(min(x_i, 4),
               "`index` input argument should be a character `i`, `j` or NA.",
               fixed = TRUE)
  
})


test_that("min_reduction", {
  # basic example
  D <- 3
  M <- 100
  N <- 150
  x <- matrix(runif(M * D), M, D)
  x_i <- LazyTensor(x, index = 'i')
  
  # check results, formulas & classes
  expect_true(class(min_reduction(x_i, "i"))[1] != "LazyTensor")
  
  # errors
  expect_error(min_reduction(3, "i"),
               "`x` input should be a LazyTensor.",
               fixed = TRUE)
  
  expect_error(min_reduction(x, "i"),
               "`x` input should be a LazyTensor.",
               fixed = TRUE)
  
  expect_error(min_reduction(x_i, "b"),
               "`index` input argument should be a character `i`, `j`.",
               fixed = TRUE)
  
  expect_error(min_reduction(x_i, 3),
               "`index` input argument should be a character `i`, `j`.",
               fixed = TRUE)
  
})


test_that("argmin", {
  # basic example
  D <- 3
  M <- 100
  N <- 150
  x <- matrix(runif(M * D), M, D)
  x_i <- LazyTensor(x, index = 'i')
  
  # check results, formulas & classes
  expect_true(class(argmin(3)) == "LazyTensor")
  expect_true(class(argmin(x_i, "i"))[1] != "LazyTensor")
  
  
  obj <- argmin(x_i)
  expect_true(class(obj) == "LazyTensor")
  bool_grep_formula <- grep("ArgMin\\(A0x.*i\\)", obj$formula)
  expect_equal(bool_grep_formula, 1)
  
  # errors
  expect_error(argmin(3, "i"),
               "`x` input should be a LazyTensor.",
               fixed = TRUE)
  
  expect_error(argmin(x, "i"),
               "`x` input should be a LazyTensor.",
               fixed = TRUE)
  
  expect_error(argmin(x_i, "b"),
               "`index` input argument should be a character `i`, `j` or NA.",
               fixed = TRUE)
  
  expect_error(argmin(x_i, 3),
               "`index` input argument should be a character `i`, `j` or NA.",
               fixed = TRUE)
})


test_that("argmin_reduction", {
  # basic example
  D <- 3
  M <- 100
  N <- 150
  x <- matrix(runif(M * D), M, D)
  x_i <- LazyTensor(x, index = 'i')
  
  # check results, formulas & classes
  expect_true(class(argmin_reduction(x_i, "i"))[1] != "LazyTensor")
  
  # errors
  expect_error(argmin_reduction(3, "i"),
               "`x` input should be a LazyTensor.",
               fixed = TRUE)
  
  expect_error(argmin_reduction(x_i, 3),
               "`index` input argument should be a character `i`, `j`.",
               fixed = TRUE)
})


test_that("min_argmin", {
  # basic example
  D <- 3
  M <- 100
  N <- 150
  x <- matrix(runif(M * D), M, D)
  x_i <- LazyTensor(x, index = 'i')
  
  # check results, formulas & classes
  expect_true(class(min_argmin(x_i, "i"))[1] != "LazyTensor")
  
  # errors
  expect_error(min_argmin(3, "i"),
               "`x` input should be a LazyTensor.",
               fixed = TRUE)
  
  expect_error(min_argmin(x_i, "b"),
               "`index` input argument should be a character `i`, `j`.",
               fixed = TRUE)
})



test_that("min_argmin_reduction", {
  # basic example
  D <- 3
  M <- 100
  N <- 150
  x <- matrix(runif(M * D), M, D)
  x_i <- LazyTensor(x, index = 'i')
  
  # check results, formulas & classes
  expect_true(class(min_argmin_reduction(x_i, "i"))[1] != "LazyTensor")
  
  # errors
  expect_error(min_argmin_reduction(3, "i"),
               "`x` input should be a LazyTensor.",
               fixed = TRUE)
  
  expect_error(min_argmin_reduction(x_i, "b"),
               "`index` input argument should be a character `i`, `j`.",
               fixed = TRUE)
})



test_that("max", {
  # basic example
  D <- 3
  M <- 100
  x <- matrix(runif(M * D), M, D)
  x_i <- LazyTensor(x, index = 'i')
  
  # check results, formulas & classes
  expect_equal(max(D), 3)
  expect_true(class(max(x))[1] != "LazyTensor")
  expect_true(class(max(x_i, "i"))[1] != "LazyTensor")
  
  obj <- max(x_i)
  expect_true(class(obj) == "LazyTensor")
  bool_grep_formula <- grep("Max\\(A0x.*i\\)", obj$formula)
  expect_equal(bool_grep_formula, 1)
  
  # errors
  expect_error(max(x_i, "b"),
               "`index` input argument should be a character `i`, `j` or NA.",
               fixed = TRUE)
  
  expect_error(max(x_i, 4),
               "`index` input argument should be a character `i`, `j` or NA.",
               fixed = TRUE)
})


test_that("max_reduction", {
  # basic example
  D <- 3
  M <- 100
  x <- matrix(runif(M * D), M, D)
  x_i <- LazyTensor(x, index = 'i')
  
  # check results, formulas & classes
  expect_true(class(max_reduction(x_i, "i"))[1] != "LazyTensor")
  
  # errors
  expect_error(max_reduction(x_i, "b"),
               "`index` input argument should be a character `i`, `j`.",
               fixed = TRUE)
  
  expect_error(max_reduction(x, "i"),
               "`x` input should be a LazyTensor.",
               fixed = TRUE)
})


test_that("argmax", {
  # basic example
  D <- 3
  M <- 100
  N <- 150
  x <- matrix(runif(M * D), M, D)
  x_i <- LazyTensor(x, index = 'i')
  
  # check results, formulas & classes
  expect_true(class(argmax(3)) == "LazyTensor")
  expect_true(class(argmax(x_i, "i"))[1] != "LazyTensor")
  
  
  obj <- argmax(x_i)
  expect_true(class(obj) == "LazyTensor")
  bool_grep_formula <- grep("ArgMax\\(A0x.*i\\)", obj$formula)
  expect_equal(bool_grep_formula, 1)
  
  # errors
  expect_error(argmax(3, "i"),
               "`x` input should be a LazyTensor.",
               fixed = TRUE)
  
  expect_error(argmax(x, "i"),
               "`x` input should be a LazyTensor.",
               fixed = TRUE)
  
  expect_error(argmax(x_i, "b"),
               "`index` input argument should be a character `i`, `j` or NA.",
               fixed = TRUE)
  
  expect_error(argmax(x_i, 3),
               "`index` input argument should be a character `i`, `j` or NA.",
               fixed = TRUE)
})


test_that("argmax_reduction", {
  # basic example
  D <- 3
  M <- 100
  N <- 150
  x <- matrix(runif(M * D), M, D)
  x_i <- LazyTensor(x, index = 'i')
  
  # check results, formulas & classes
  expect_true(class(argmax_reduction(x_i, "i"))[1] != "LazyTensor")
  
  # errors
  expect_error(argmax_reduction(3, "i"),
               "`x` input should be a LazyTensor.",
               fixed = TRUE)
  
  expect_error(argmax_reduction(x_i, 3),
               "`index` input argument should be a character `i`, `j`.",
               fixed = TRUE)
})


test_that("max_argmax", {
  # basic example
  D <- 3
  M <- 100
  N <- 150
  x <- matrix(runif(M * D), M, D)
  x_i <- LazyTensor(x, index = 'i')
  
  # check results, formulas & classes
  expect_true(class(max_argmax(x_i, "i"))[1] != "LazyTensor")
  
  # errors
  expect_error(max_argmax(3, "i"),
               "`x` input should be a LazyTensor.",
               fixed = TRUE)
  
  expect_error(max_argmax(x_i, "b"),
               "`index` input argument should be a character `i`, `j`.",
               fixed = TRUE)
})



test_that("max_argmax_reduction", {
  # basic example
  D <- 3
  M <- 100
  N <- 150
  x <- matrix(runif(M * D), M, D)
  x_i <- LazyTensor(x, index = 'i')
  
  # check results, formulas & classes
  expect_true(class(max_argmax_reduction(x_i, "i"))[1] != "LazyTensor")
  
  # errors
  expect_error(max_argmax_reduction(3, "i"),
               "`x` input should be a LazyTensor.",
               fixed = TRUE)
  
  expect_error(max_argmax_reduction(x_i, "b"),
               "`index` input argument should be a character `i`, `j`.",
               fixed = TRUE)
})



test_that("xlogx", {
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
  expect_true(class(xlogx(x_i)) == "LazyTensor")
  
  obj <- xlogx(x_i)
  bool_grep_formula <- grep("XLogX\\(A0x.*i\\)", obj$formula)
  expect_equal(bool_grep_formula, 1)
})


test_that("sinxdivx", {
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
  expect_true(class(sinxdivx(x_i)) == "LazyTensor")
  
  obj <- sinxdivx(x_i)
  bool_grep_formula <- grep("SinXDivX\\(A0x.*i\\)", obj$formula)
  expect_equal(bool_grep_formula, 1)
})


test_that("inv", {
  # basic example
  D <- 3
  M <- 100
  N <- 150
  x <- matrix(runif(M * D), M, D)
  x_i <- LazyTensor(x, index = 'i')
  
  # check results, formulas & classes
  expect_equal(inv(2), 0.5)
  expect_true(class(inv(x))[1] != "LazyTensor")
  expect_true(class(inv(x_i)) == "LazyTensor")
  
  obj <- inv(x_i)
  bool_grep_formula <- grep("Inv\\(A0x.*i\\)", obj$formula)
  expect_equal(bool_grep_formula, 1)
})


test_that("relu", {
  # basic example
  D <- 3
  M <- 100
  N <- 150
  x <- matrix(runif(M * D), M, D)
  x_i <- LazyTensor(x, index = 'i')
  
  # check results, formulas & classes
  expect_true(class(relu(2)) == "LazyTensor")
  expect_true(class(relu(x_i)) == "LazyTensor")
  
  obj <- relu(x_i)
  bool_grep_formula <- grep("ReLu\\(A0x.*i\\)", obj$formula)
  expect_equal(bool_grep_formula, 1)
})


test_that("step.LazyTensor", {
  # basic example
  D <- 3
  M <- 100
  N <- 150
  x <- matrix(runif(M * D), M, D)
  x_i <- LazyTensor(x, index = 'i')
  
  # check results, formulas & classes
  expect_true(class(step.LazyTensor(2)) == "LazyTensor")
  expect_true(class(step.LazyTensor(x_i)) == "LazyTensor")
  
  obj <- step.LazyTensor(x_i)
  bool_grep_formula <- grep("Step\\(A0x.*i\\)", obj$formula)
  expect_equal(bool_grep_formula, 1)
})


test_that("sqnorm2", {
  # basic example
  D <- 3
  M <- 100
  N <- 150
  x <- matrix(runif(M * D), M, D)
  x_i <- LazyTensor(x, index = 'i')
  
  # check results, formulas & classes
  expect_true(class(sqnorm2(2)) == "LazyTensor")
  expect_true(class(sqnorm2(x_i)) == "LazyTensor")
  
  obj <- sqnorm2(x_i)
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
  
  # check results, formulas & classes
  expect_true(class(norm2(2)) == "LazyTensor")
  expect_true(class(norm2(x_i)) == "LazyTensor")
  
  obj <- norm2(x_i)
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
  
  # check results, formulas & classes
  expect_true(class(normalize(2)) == "LazyTensor")
  expect_true(class(normalize(x_i)) == "LazyTensor")
  
  obj <- normalize(x_i)
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
  
  # check results, formulas & classes
  expect_true(class(sqdist(2, 3)) == "LazyTensor")
  expect_true(class(sqdist(x_i, y_j)) == "LazyTensor")
  
  obj <- sqdist(x_i, y_j)
  bool_grep_formula <- grep("SqDist\\(A0x.*i,A0x.*j\\)", obj$formula)
  expect_equal(bool_grep_formula, 1)
  
  # errors
  expect_error(sqdist(x_i, t_j),
               "Operation `SqDist` expects inputs of the same dimension or dimension 1. Received 3 and 7.",
               fixed = TRUE)
})


test_that("weightedsqnorm", {
  # basic example
  D <- 3
  M <- 100
  N <- 150
  x <- matrix(runif(M * D), M, D)
  y <- matrix(runif(N * D), N, D)
  x_i <- LazyTensor(x, index = 'i')
  y_j <- LazyTensor(y, index = 'j')
  
  # check results, formulas & classes
  expect_true(class(weightedsqnorm(2, 3)) == "LazyTensor")
  expect_true(class(weightedsqnorm(x_i, y_j)) == "LazyTensor")
  
  obj <- weightedsqnorm(x_i, y_j)
  bool_grep_formula <- grep("WeightedSqNorm\\(A0x.*i,A0x.*j\\)", obj$formula)
  expect_equal(bool_grep_formula, 1)
})


# TEST COMPLEX FUNCTIONS =======================================================


test_that("Re", {
  # basic example
  D <- 3
  M <- 100
  x <- matrix(runif(M * D), M, D)
  y <- matrix(1i^ (-6:5), nrow = 4)
  x_i <- LazyTensor(x, index = 'i')
  z_i <- LazyTensor(y, index = 'i', is_complex = TRUE)
  xc_i <- LazyTensor(x, index = 'i', is_complex = TRUE)
  
  # check formulas, args & classes
  expect_true(Re(2) == 2)
  expect_true(Re(2+1i) == 2)
  
  
  obj <-  Re(z_i)
  bool_grep_formula <- grep("ComplexReal\\(A0x.*i\\)", obj$formula)
  expect_equal(bool_grep_formula, 1)
  expect_is(obj, "LazyTensor")
  
  expect_error(Re(x_i), 
               "`Re` cannot be applied to a LazyTensor. See `?Re` for compatible types.",
               fixed = TRUE)
  
})


test_that("Im", {
  # basic example
  D <- 3
  M <- 100
  x <- matrix(runif(M * D), M, D)
  y <- matrix(1i^ (-6:5), nrow = 4)
  x_i <- LazyTensor(x, index = 'i')
  z_i <- LazyTensor(y, index = 'i', is_complex = TRUE)
  xc_i <- LazyTensor(x, index = 'i', is_complex = TRUE)
  
  # check formulas & classes
  expect_true(Im(2) == 0)
  expect_true(Im(2+1i) == 1)
  
  
  obj <-  Im(z_i)
  bool_grep_formula <- grep("ComplexImag\\(A0x.*i\\)", obj$formula)
  expect_equal(bool_grep_formula, 1)
  expect_is(obj, "LazyTensor")
  
  expect_error(Im(x_i), 
               "`Im` cannot be applied to a LazyTensor. See `?Im` for compatible types.",
               fixed = TRUE)
  
})

test_that("Arg", {
  # basic example
  D <- 3
  M <- 100
  x <- matrix(runif(M * D), M, D)
  y <- matrix(1i^ (-6:5), nrow = 4)
  x_i <- LazyTensor(x, index = 'i')
  z_i <- LazyTensor(y, index = 'i', is_complex = TRUE)
  xc_i <- LazyTensor(x, index = 'i', is_complex = TRUE)
  
  # check formulas & classes
  expect_true(Arg(2) == 0)
  expect_true(round(Arg(pi*1i),2) == 1.57)
  
  
  obj <-  Arg(z_i)
  bool_grep_formula <- grep("ComplexAngle\\(A0x.*i\\)", obj$formula)
  expect_equal(bool_grep_formula, 1)
  expect_is(obj, "LazyTensor")
  
  expect_error(Arg(x_i), 
               "`Arg` cannot be applied to a LazyTensor. See `?Arg` for compatible types.",
               fixed = TRUE)
  
})


test_that("real2complex", {
  # basic example
  D <- 3
  M <- 100
  x <- matrix(runif(M * D), M, D)
  x_i <- LazyTensor(x, index = 'i')
  xc_i <- LazyTensor(x, index = 'i', is_complex = TRUE)
  
  # check formulas & classes
  obj <-  real2complex(x_i)
  bool_grep_formula <- grep("Real2Complex\\(A0x.*i\\)", obj$formula)
  expect_equal(bool_grep_formula, 1)
  expect_is(obj, "ComplexLazyTensor")
  
  expect_error(real2complex(xc_i), 
               "`real2complex` cannot be applied to a complex LazyTensor.",
               fixed = TRUE)
  
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
  bool_grep_formula <- grep("Imag2Complex\\(A0x.*i\\)", obj$formula)
  expect_equal(bool_grep_formula, 1)
  expect_is(obj, "ComplexLazyTensor")
  
  expect_error(imag2complex(xc_i), 
               "`imag2complex` cannot be applied to a complex LazyTensor.",
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
  bool_grep_formula <- grep("ComplexExp1j\\(A0x.*i\\)", obj$formula)
  expect_equal(bool_grep_formula, 1)
  expect_is(obj, "ComplexLazyTensor")
  
  expect_error(exp1j(xc_i), 
               "`exp1j` cannot be applied to a complex LazyTensor.",
               fixed = TRUE)
  
})


test_that("Conj", {
  # basic example
  D <- 3
  M <- 100
  N <- 150
  x <- matrix(runif(M * D), M, D)
  x_i <- LazyTensor(x, index = 'i')
  
  z <- matrix(2i^(-6:5), nrow = 4) # complex 4x3 matrix
  z_i <- LazyTensor(z, index = 'i', is_complex = TRUE)  # ComplexLazyTensor
  
  # check results, formulas & classes
  expect_true(class(Conj(z_i)) == "ComplexLazyTensor")
  
  obj <- Conj(z_i)
  bool_grep_formula <- grep("Conj\\(A0x.*i\\)", obj$formula)
  expect_equal(bool_grep_formula, 1)
  
  # errors
  expect_error(Conj(x_i),
               "`Conj` cannot be applied to a LazyTensor. See `?Conj` for compatible types.",
               fixed = TRUE)
})


test_that("Mod", {
  # basic example
  D <- 3
  M <- 100
  N <- 150
  x <- matrix(runif(M * D), M, D)
  x_i <- LazyTensor(x, index = 'i')
  
  z <- matrix(2i^(-6:5), nrow = 4) # complex 4x3 matrix
  z_i <- LazyTensor(z, index = 'i', is_complex = TRUE)  # ComplexLazyTensor
  
  # check results, formulas & classes
  expect_true(class(Mod(z_i)) == "LazyTensor")
  
  obj <- Mod(z_i)
  bool_grep_formula <- grep("ComplexAbs\\(A0x.*i\\)", obj$formula)
  expect_equal(bool_grep_formula, 1)
})


# TEST REDUCTIONS ==============================================================


test_that("reduction.LazyTensor", {
  # basic example
  D <- 3
  M <- 100
  N <- 150
  x <- matrix(runif(M * D), M, D)
  x_i <- LazyTensor(x, index = 'i')
  opstr <- "Sum"
  
  obj <- reduction.LazyTensor(x_i, opstr, "i")
  
  # check results, formulas & classes
  expect_true(class(obj)[1] != "LazyTensor")
  
  # errors
  expect_error(reduction.LazyTensor(3, opstr, "i"),
               "`x` input should be a LazyTensor.", 
               fixed=TRUE)
  
  expect_error(reduction.LazyTensor(x, opstr, "i"),
               "`x` input should be a LazyTensor.", 
               fixed=TRUE)
  
  expect_error(reduction.LazyTensor(x_i, opstr, "b"),
               "`index` input argument should be a character `i`, `j`.", 
               fixed=TRUE)
  
  expect_error(reduction.LazyTensor(x_i, 2, "i"),
               "`opst` input should be a string text.", 
               fixed=TRUE)
  
})


test_that("sum", {
  # basic example
  D <- 3
  M <- 100
  N <- 150
  x <- matrix(runif(M * D), M, D)
  x_i <- LazyTensor(x, index = 'i')
  
  obj <- sum(x_i, "i")
  expect_true(class(obj)[1] != "LazyTensor")
  
  obj <- sum(x_i)
  expect_true(class(obj) == "LazyTensor")
 
  bool_grep_formula <- grep("Sum\\(A0x.*i\\)", obj$formula)
  expect_equal(bool_grep_formula, 1)
  
  # errors
  expect_error(sum(x_i, "b"),
               "`index` input argument should be a character `i`, `j` or NA.", 
               fixed=TRUE)
  
  expect_error(sum(x_i, 2),
               "`index` input argument should be a character `i`, `j` or NA.", 
               fixed=TRUE)
})


test_that("sum_reduction", {
  # basic example
  D <- 3
  M <- 100
  N <- 150
  x <- matrix(runif(M * D), M, D)
  x_i <- LazyTensor(x, index = 'i')
  
  obj <- sum_reduction(x_i, "i")
  expect_true(class(obj)[1] != "LazyTensor")
  
  # errors
  expect_error(sum_reduction(x_i, "b"),
               "`index` input argument should be a character `i`, `j` or NA.", 
               fixed=TRUE)
  
  expect_error(sum_reduction(x_i, 2),
               "`index` input argument should be a character `i`, `j` or NA.", 
               fixed=TRUE)
  
  expect_error(sum_reduction(x, "i"),
               "`x` input should be a LazyTensor.", 
               fixed=TRUE)
  
  expect_error(sum_reduction(3, "i"),
               "`x` input should be a LazyTensor.", 
               fixed=TRUE)
  
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
  
  # check formulas, args & classes
  obj <-  clamp(x_i, y_j, z_i)
  bool_grep_formula <- grep("Clamp\\(A0x.*i,A0x.*j,A0x.*i\\)", obj$formula)
  expect_equal(bool_grep_formula, 1)
  expect_is(obj, "LazyTensor")
  
  obj <-  clamp(x_i, y_j, 3)
  bool_grep_formula <- grep("Clamp\\(A0x.*i,A0x.*j,IntCst\\(3\\)\\)", obj$formula)
  expect_equal(bool_grep_formula, 1)
  
  obj <-  clamp(x_i, 2, 3)
  bool_grep_formula <- grep("ClampInt\\(A0x.*i,2,3\\)", obj$formula)
  expect_equal(bool_grep_formula, 1)
  
  # errors
  expect_error(clamp(x_i, y_j, w_i),
               "Operation `Clamp` expects inputs of the same dimension or dimension 1. Received 3, 3 and 7.",
               fixed = TRUE)
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
  
  # check formulas, args & classes
  obj <-  clampint(x_i, 6, 8)
  bool_grep_formula <- grep("ClampInt\\(A0x.*i,6,8\\)", obj$formula) 
  expect_equal(bool_grep_formula, 1)
  expect_is(obj, "LazyTensor")
  
  # errors
  expect_error(clampint(x_i, y_j, 8),
               "'clampint(x, y, z)' expects integer arguments for `y` and `z`. Use clamp(x, y, z) for different `y` and `z` types.",
               fixed = TRUE)
  expect_error(clampint(x_i, y_j, z_i),
               "'clampint(x, y, z)' expects integer arguments for `y` and `z`. Use clamp(x, y, z) for different `y` and `z` types.",
               fixed = TRUE)
})


test_that("ifelse.LazyTensor", {
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
  
  # check formulas, args & classes
  obj <-  ifelse.LazyTensor(x_i, y_j, z_i)
  bool_grep_formula <- grep("IfElse\\(A0x.*i,A0x.*j,A0x.*i\\)", obj$formula)
  expect_equal(bool_grep_formula, 1)
  expect_is(obj, "LazyTensor")
  
  # errors
  expect_error(ifelse.LazyTensor(x_i, y_j, w_i),
               "Operation `IfElse` expects inputs of the same dimension or dimension 1. Received 3, 3 and 7.",
               fixed = TRUE)
})


test_that("check_index", {
  expect_is(check_index("i"), "logical")
  
  expect_true(check_index("i"))
  expect_true(check_index("j"))
  
  expect_false(check_index(5))
  expect_false(check_index("n"))
})




