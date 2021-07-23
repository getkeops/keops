context("LazyTensor operations")


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
  
  expect_true(is.LazyTensor(x_i + y_j))
  expect_true(is.ComplexLazyTensor(x_i + xc_i))
  expect_true(is.ComplexLazyTensor(xc_i + x_i))
  expect_true(is.ComplexLazyTensor(xc_i + yc_j))
  
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
  expect_error(
    x_i + z_j,
    paste(
      "Operation `+` expects inputs of the same dimension or dimension 1.", 
      " Received 3 and 7.", 
      sep = ""
      ),
    fixed = TRUE
    )
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
  
  expect_true(is.LazyTensor(x_i - y_j))
  expect_true(is.ComplexLazyTensor(x_i - xc_i))
  expect_true(is.ComplexLazyTensor(xc_i - yc_j))
  
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
  
  expect_true(is.LazyTensor(x_i * y_j))
  expect_true(is.ComplexLazyTensor(x_i * yc_j))
  expect_true(is.ComplexLazyTensor(xc_i * y_j))
  expect_true(is.ComplexLazyTensor(xc_i * yc_j))
  
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
  
  expect_true(is.LazyTensor(x_i / y_j))
  expect_true(is.ComplexLazyTensor(x_i / yc_j))
  expect_true(is.ComplexLazyTensor(xc_i / y_j))
  expect_true(is.ComplexLazyTensor(xc_i / yc_j))
  
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
  # TODO add test for `scalar %*% LazyTensor` and `matrix %*% LazyTensor` 
  # when problem will be fixed
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
  xc_i <- LazyTensor(x, index = 'i', is_complex = TRUE)
  
  # check results, formulas & classes
  expect_equal(abs(-D), 3)
  expect_true(!is.LazyTensor(x))
  expect_true(is.LazyTensor(abs(x_i)))
  
  obj <- abs(x_i)
  bool_grep_formula <- grep("Abs\\(A0x.*i\\)", obj$formula)
  expect_equal(bool_grep_formula, 1)
  
  obj <-  abs(-y_j)
  bool_grep_formula <- grep("Abs\\(Minus\\(A0x.*j\\)\\)", obj$formula)
  expect_equal(bool_grep_formula, 1)
  
  obj <-  abs(-xc_i)
  bool_grep_formula <- grep("ComplexAbs\\(Minus\\(A0x.*i\\)\\)", obj$formula)
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
  xc_i <- LazyTensor(x, index = 'i', is_complex = TRUE)
  
  # check results, formulas & class
  expect_equal(round(pi, 2), 3.14)
  expect_false(is.LazyTensor(round(pi, 3)))
  expect_true(is.LazyTensor(round(x_i, 3)))
  expect_true(is.ComplexLazyTensor(round(xc_i, 3)))
  
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
  
  expect_error(round(x_i, 2i), 
               "`d` input argument should be a scalar.", 
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
  expect_true(is.LazyTensor(xlogx(x_i)))
  
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
  expect_true(is.LazyTensor(sinxdivx(x_i)))
  
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
  expect_false(is.LazyTensor(inv(x))[1])
  expect_true(is.LazyTensor(inv(x_i)))
  
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
  expect_true(is.LazyTensor(relu(2)))
  expect_true(is.LazyTensor(relu(x_i)))
  
  obj <- relu(x_i)
  bool_grep_formula <- grep("ReLU\\(A0x.*i\\)", obj$formula)
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
  
  # check results, formulas & classes
  expect_true(is.LazyTensor(step(x_i)))
  expect_true(is.ComplexLazyTensor(step(xc_i)))
  
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
  xc_i <- LazyTensor(x, index = 'i', is_complex = TRUE)
  
  # check results, formulas & classes
  expect_true(is.LazyTensor(sqnorm2(2)))
  expect_true(is.LazyTensor(sqnorm2(x_i)))
  expect_true(is.LazyTensor(sqnorm2(xc_i)))
  expect_false(is.ComplexLazyTensor(sqnorm2(xc_i)))
  
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
  xc_i <- LazyTensor(x, index = 'i', is_complex = TRUE)
  
  # check results, formulas & classes
  expect_true(is.LazyTensor(norm2(2)))
  expect_true(is.LazyTensor(norm2(x_i)))
  expect_true(is.LazyTensor(norm2(xc_i)))
  expect_false(is.ComplexLazyTensor(norm2(xc_i)))
  
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
  xc_i <- LazyTensor(x, index = 'i', is_complex = TRUE)
  
  # check results, formulas & classes
  expect_true(is.LazyTensor(normalize(2)))
  expect_true(is.LazyTensor(normalize(x_i)))
  expect_true(is.ComplexLazyTensor(normalize(x_i)))
  
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
  bool_grep_formula <- grep("SqDist\\(A0x.*i,A0x.*j\\)", obj$formula)
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
  N <- 150
  x <- matrix(runif(M * D), M, D)
  y <- matrix(runif(N * D), N, D)
  x_i <- LazyTensor(x, index = 'i')
  y_j <- LazyTensor(y, index = 'j')
  xc_i <- LazyTensor(x, index = 'i', is_complex = TRUE)
  yc_j <- LazyTensor(y, index = 'j', is_complex = TRUE)
  
  # check results, formulas & classes
  expect_true(is.LazyTensor(weightedsqnorm(2, 3)))
  expect_true(is.LazyTensor(weightedsqnorm(x_i, y_j)))
  expect_false(is.ComplexLazyTensor(weightedsqnorm(xc_i, yc_j)))
  expect_true(is.LazyTensor(weightedsqnorm(xc_i, yc_j)))
  expect_false(is.ComplexLazyTensor(weightedsqnorm(x_i, yc_j)))
  expect_true(is.LazyTensor(weightedsqnorm(x_i, yc_j)))
  expect_false(is.ComplexLazyTensor(weightedsqnorm(xc_i, y_j)))
  expect_true(is.LazyTensor(weightedsqnorm(xc_i, y_j)))
  
  obj <- weightedsqnorm(x_i, y_j)
  bool_grep_formula <- grep("WeightedSqNorm\\(A0x.*i,A0x.*j\\)", obj$formula)
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
  expect_error(
    clamp(x_i, y_j, w_i),
    paste("Operation `Clamp` expects inputs of the same dimension or dimension 1.",
          " Received 3, 3 and 7.", sep = ""
          ),
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
  
  # check formulas, args & classes
  obj <-  clampint(x_i, 6, 8)
  bool_grep_formula <- grep("ClampInt\\(A0x.*i,6,8\\)", obj$formula) 
  expect_equal(bool_grep_formula, 1)
  expect_is(obj, "LazyTensor")
  
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
  
})


test_that("ifelse", {
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
  obj <-  ifelse(x_i, y_j, z_i)
  bool_grep_formula <- grep("IfElse\\(A0x.*i,A0x.*j,A0x.*i\\)", obj$formula)
  expect_equal(bool_grep_formula, 1)
  expect_is(obj, "LazyTensor")
  
  # errors
  expect_error(
    ifelse(x_i, y_j, w_i),
    paste("Operation `IfElse` expects inputs of the same dimension or dimension 1.",
          " Received 3, 3 and 7.", sep = ""
          ),
    fixed = TRUE
    )
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
  
  expect_error(
    Re(x_i), 
    "`Re` cannot be applied to a LazyTensor. See `?Re` for compatible types.",
    fixed = TRUE
    )
  
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
  
  expect_error(
    Im(x_i), 
    "`Im` cannot be applied to a LazyTensor. See `?Im` for compatible types.",
    fixed = TRUE
    )
  
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
  expect_true(is.ComplexLazyTensor(Conj(z_i)))
  
  obj <- Conj(z_i)
  bool_grep_formula <- grep("Conj\\(A0x.*i\\)", obj$formula)
  expect_equal(bool_grep_formula, 1)
  
  # errors
  expect_error(
    Conj(x_i),
    "`Conj` cannot be applied to a LazyTensor. See `?Conj` for compatible types.",
    fixed = TRUE
    )
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
  expect_true(is.LazyTensor(Mod(z_i)))
  expect_false(is.ComplexLazyTensor(Mod(z_i)))
  
  obj <- Mod(z_i)
  bool_grep_formula <- grep("ComplexAbs\\(A0x.*i\\)", obj$formula)
  expect_equal(bool_grep_formula, 1)
})


# TEST CONSTANT AND PADDING/CONCATENATION OPERATIONS ===========================


test_that("elem", {
  # basic example
  x <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows, 3 columns
  x_i <- LazyTensor(x, index = 'i')   # LazyTensor from matrix x, indexed by 'i'
  m <- 2
  
  # check formulas, args & classes
  obj <- elem(x_i, m)
  expect_true(is.LazyTensor(obj))
  bool_grep_formula <- grep("Elem\\(A0x.*i,2\\)", obj$formula)
  expect_equal(bool_grep_formula, 1)
  
  # errors
  expect_error(elem(x_i, 3.14),
               "`m` input argument should be an integer.",
               fixed = TRUE)
  expect_error(elem(x_i, 4),
               "Index `m` is out of bounds. Should be in [1, 3].",
               fixed = TRUE)
  
})


test_that("extract", {
  # basic example
  x <- matrix(runif(150 * 5), 150, 5) # arbitrary R matrix, 150 rows, 5 columns
  x_i <- LazyTensor(x, index = 'i')   # LazyTensor from matrix x, indexed by 'i'
  m <- 1
  d <- 3
  
  # check formulas, args & classes
  obj <- extract(x_i, m, d)
  expect_true(is.LazyTensor(obj))
  bool_grep_formula <- grep("Extract\\(A0x.*i,1,3\\)", obj$formula)
  expect_equal(bool_grep_formula, 1)
  
  # errors
  expect_error(extract(x_i, 3.14, d),
               "`m` input argument should be an integer.",
               fixed = TRUE)
  expect_error(extract(x_i, m, 3.14),
               "`d` input argument should be an integer.",
               fixed = TRUE)
  expect_error(extract(x_i, 7, d),
               "Index `m` is out of bounds. Should be in [1, 5].",
               fixed = TRUE)
  expect_error(extract(x_i, 4, d),
               paste("Slice dimension is out of bounds. Input `d` should be ",
                     "in [1, 5-m] where `m` is the starting index.", sep = ""),
               fixed = TRUE)
})


test_that("concat", {
  # basic example
  x <- matrix(runif(150 * 3), 150, 3) # arbitrary R matrix, 150 rows, 3 columns
  y <- matrix(runif(250 * 3), 250, 3) # arbitrary R matrix, 250 rows, 3 columns
  x_i <- LazyTensor(x, index = 'i')   # LazyTensor from matrix x, indexed by 'i'
  y_j <- LazyTensor(y, index = 'j')   # LazyTensor from matrix x, indexed by 'j'
  
  # check formulas, args & classes
  obj <- concat(x_i, y_j)
  expect_true(is.LazyTensor(obj))
  bool_grep_formula <- grep("Concat\\(A0x.*i,A0x.*j\\)", obj$formula)
  expect_equal(bool_grep_formula, 1)
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
  
  z <- matrix(1i^ (-6:5), nrow = 4)                     # complex 4x3 matrix
  z_i <- LazyTensor(z, index = 'i', is_complex = TRUE)  # ComplexLazyTensor
  
  D <- 7
  # check formulas, args & classes
  obj <- one_hot(LT_s, D)
  expect_true(is.LazyTensor(obj))
  bool_grep_formula <- grep("OneHot\\(IntCst\\(13\\),7\\)", obj$formula)
  expect_equal(bool_grep_formula, 1)
  
  # errors
  expect_error(one_hot(x_i, D),
               "One-hot encoding is only supported for scalar formulas.",
               fixed = TRUE)
  expect_error(one_hot(y_i, D),
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
  expect_false(is.LazyTensor(obj))
  
  # TODO : reduction with optional arg with Kmin or not ?
  
  # errors
  expect_error(reduction.LazyTensor(3, opstr, "i"),
               "`x` input should be a LazyTensor or a ComplexLazyTensor.", 
               fixed=TRUE)
  
  expect_error(reduction.LazyTensor(x, opstr, "i"),
               "`x` input should be a LazyTensor or a ComplexLazyTensor.", 
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
  expect_false(is.LazyTensor(obj))
  
  obj <- sum(x_i)
  expect_true(is.LazyTensor(obj))
  
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
  expect_false(is.LazyTensor(obj))
  
  # errors
  expect_error(sum_reduction(x_i, "b"),
               "`index` input argument should be a character `i`, `j` or NA.", 
               fixed=TRUE)
  
  expect_error(sum_reduction(x_i, 2),
               "`index` input argument should be a character `i`, `j` or NA.", 
               fixed=TRUE)
  
  expect_error(sum_reduction(x, "i"),
               "`x` input should be a LazyTensor or a ComplexLazyTensor.", 
               fixed=TRUE)
  
  expect_error(sum_reduction(3, "i"),
               "`x` input should be a LazyTensor or a ComplexLazyTensor.", 
               fixed=TRUE)
  
})


test_that("min", {
  # basic example
  D <- 3
  M <- 100
  N <- 150
  x <- matrix(runif(M * D), M, D)
  x_i <- LazyTensor(x, index = "i")
  xc_i <- LazyTensor(x, index = "i", is_complex = TRUE)
  
  # check results, formulas & classes
  expect_equal(min(D), 3)
  expect_false(is.LazyTensor(min(x)))
  expect_false(is.LazyTensor(min(x_i, "i")))
  expect_true(is.ComplexLazyTensor(min(xc_i)))
  
  obj <- min(x_i)
  expect_true(is.LazyTensor(obj))
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
  
  # check class
  expect_false(is.LazyTensor(min_reduction(x_i, "i")))
  
  # errors
  expect_error(min_reduction(3, "i"),
               "`x` input should be a LazyTensor or a ComplexLazyTensor.",
               fixed = TRUE)
  
  expect_error(min_reduction(x, "i"),
               "`x` input should be a LazyTensor or a ComplexLazyTensor.",
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
  xc_i <- LazyTensor(x, index = "i", is_complex = TRUE)
  
  
  # check results, formulas & classes
  expect_true(is.LazyTensor(argmin(3)))
  expect_true(!is.LazyTensor(argmin(x_i, "i")))
  
  
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
  expect_false(is.LazyTensor(argmin_reduction(x_i, "i")))
  
  # errors
  expect_error(argmin_reduction(3, "i"),
               "`x` input should be a LazyTensor or a ComplexLazyTensor.",
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
  expect_false(is.LazyTensor(min_argmin(x_i, "i")))
  
  # errors
  expect_error(min_argmin(3, "i"),
               "`x` input should be a LazyTensor or a ComplexLazyTensor.",
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
  expect_true(is.LazyTensor(min_argmin_reduction(x_i, "i")))
  
  # errors
  expect_error(min_argmin_reduction(3, "i"),
               "`x` input should be a LazyTensor or a ComplexLazyTensor.",
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
  expect_false(is.LazyTensor(max(x)))
  expect_false(is.LazyTensor(max(x_i, "i")))
  
  obj <- max(x_i)
  expect_true(is.LazyTensor(obj))
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
  expect_false(is.LazyTensor(max_reduction(x_i, "i")))
  
  # errors
  expect_error(max_reduction(x_i, "b"),
               "`index` input argument should be a character `i`, `j`.",
               fixed = TRUE)
  
  expect_error(max_reduction(x, "i"),
               "`x` input should be a LazyTensor or a ComplexLazyTensor.",
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
  expect_true(is.LazyTensor(argmax(3)))
  expect_false(is.LazyTensor(argmax(x_i, "i")))
  
  
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
  expect_false(is.LazyTensor(argmax_reduction(x_i, "i")))
  
  # errors
  expect_error(argmax_reduction(3, "i"),
               "`x` input should be a LazyTensor or a ComplexLazyTensor.",
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
  expect_false(is.LazyTensor(max_argmax(x_i, "i")))
  
  # errors
  expect_error(max_argmax(3, "i"),
               "`x` input should be a LazyTensor or a ComplexLazyTensor.",
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
  expect_false(is.LazyTensor(max_argmax_reduction(x_i, "i")))
  
  # errors
  expect_error(max_argmax_reduction(3, "i"),
               "`x` input should be a LazyTensor or a ComplexLazyTensor.",
               fixed = TRUE)
  
  expect_error(max_argmax_reduction(x_i, "b"),
               "`index` input argument should be a character `i`, `j`.",
               fixed = TRUE)
})


test_that("Kmin", {
  # basic example
  x <- matrix(runif(150 * 3), 150, 3) 
  x_i <- LazyTensor(x, index = 'i') 
  y <- matrix(runif(100 * 3), 100, 3)
  y_j <- LazyTensor(y, index = 'j')
  
  S_ij = sum( (x_i - y_j)^2 )
  
  K <- 2
  
  # check formulas, args & classes
  obj <- Kmin(S_ij, K, "i")
  expect_false(is.LazyTensor(obj))
  expect_true(is.matrix(obj))
  
  # errors
  expect_error(Kmin(x_i, 3.14, "i"),
               "`K` input argument should be an integer.",
               fixed = TRUE)
  expect_error(Kmin(x_i, K, "k"),
               "`index` input argument should be a character, either 'i' or 'j'.",
               fixed = TRUE)
  
})


test_that("Kmin_reduction", {
  # basic example
  x <- matrix(runif(150 * 3), 150, 3) 
  x_i <- LazyTensor(x, index = 'i') 
  y <- matrix(runif(100 * 3), 100, 3)
  y_j <- LazyTensor(y, index = 'j')
  
  S_ij = sum( (x_i - y_j)^2 )
  
  K <- 2
  
  # check formulas, args & classes
  obj <- Kmin_reduction(S_ij, K, "i")
  expect_false(is.LazyTensor(obj))
  expect_true(is.matrix(obj))
  
  # errors
  expect_error(Kmin_reduction(x_i, 3.14, "i"),
               "`K` input argument should be an integer.",
               fixed = TRUE)
  expect_error(Kmin_reduction(x_i, K, "k"),
               "`index` input argument should be a character, either 'i' or 'j'.",
               fixed = TRUE)
  
})


test_that("argKmin", {
  # basic example
  x <- matrix(runif(150 * 3), 150, 3) 
  x_i <- LazyTensor(x, index = 'i') 
  y <- matrix(runif(100 * 3), 100, 3)
  y_j <- LazyTensor(y, index = 'j')
  
  S_ij = sum( (x_i - y_j)^2 )
  
  K <- 2
  
  # check formulas, args & classes
  obj <- argKmin(S_ij, K, "i")
  expect_false(is.LazyTensor(obj))
  expect_true(is.matrix(obj))
  
  # errors
  expect_error(argKmin(x_i, 3.14, "i"),
               "`K` input argument should be an integer.",
               fixed = TRUE)
  expect_error(argKmin(x_i, K, "k"),
               "`index` input argument should be a character, either 'i' or 'j'.",
               fixed = TRUE)
  
})


test_that("argKmin_reduction", {
  # basic example
  x <- matrix(runif(150 * 3), 150, 3) 
  x_i <- LazyTensor(x, index = 'i') 
  y <- matrix(runif(100 * 3), 100, 3)
  y_j <- LazyTensor(y, index = 'j')
  
  S_ij = sum( (x_i - y_j)^2 )
  
  K <- 2
  
  # check formulas, args & classes
  obj <- argKmin_reduction(S_ij, K, "i")
  expect_false(is.LazyTensor(obj))
  expect_true(is.matrix(obj))
  
  # errors
  expect_error(argKmin_reduction(x_i, 3.14, "i"),
               "`K` input argument should be an integer.",
               fixed = TRUE)
  expect_error(argKmin_reduction(x_i, K, "k"),
               "`index` input argument should be a character, either 'i' or 'j'.",
               fixed = TRUE)
  
})


test_that("Kmin_argKmin", {
  # basic example
  x <- matrix(runif(150 * 3), 150, 3) 
  x_i <- LazyTensor(x, index = 'i') 
  y <- matrix(runif(100 * 3), 100, 3)
  y_j <- LazyTensor(y, index = 'j')
  
  S_ij = sum( (x_i - y_j)^2 )
  
  K <- 2
  
  # check formulas, args & classes
  obj <- Kmin_argKmin(S_ij, K, "i")
  expect_false(is.LazyTensor(obj))
  expect_true(is.matrix(obj))
  
  # errors
  expect_error(Kmin_argKmin(x_i, 3.14, "i"),
               "`K` input argument should be an integer.",
               fixed = TRUE)
  expect_error(Kmin_argKmin(x_i, K, "k"),
               "`index` input argument should be a character, either 'i' or 'j'.",
               fixed = TRUE)
  
})


test_that("Kmin_argKmin_reduction", {
  # basic example
  x <- matrix(runif(150 * 3), 150, 3) 
  x_i <- LazyTensor(x, index = 'i') 
  y <- matrix(runif(100 * 3), 100, 3)
  y_j <- LazyTensor(y, index = 'j')
  
  S_ij = sum( (x_i - y_j)^2 )
  
  K <- 2
  
  # check formulas, args & classes
  obj <- Kmin_argKmin_reduction(S_ij, K, "i")
  
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
  # basic example
  x <- matrix(runif(150 * 3), 150, 3) 
  x_i <- LazyTensor(x, index = 'i') 
  y <- matrix(runif(100 * 3), 100, 3)
  y_j <- LazyTensor(y, index = 'j')
  w <- matrix(runif(150 * 3), 150, 3) # weight LazyTensor
  w_j <- LazyTensor(y, index = 'j')
  
  V_ij <- x_i - y_j
  S_ij <- sum(V_ij^2)
  # check formulas, args & classes
  obj <- logsumexp(S_ij, 'i', V_ij)
  expect_false(is.LazyTensor(obj))
  
  obj <- logsumexp(S_ij, 'i')
  expect_false(is.LazyTensor(obj))
  
  # errors
  expect_error(
    logsumexp(S_ij, '9'),
    "`index` input argument should be a character, either 'i' or 'j'.",
    fixed = TRUE
  )
  
})


test_that("logsumexp_reduction", {
  # basic example
  x <- matrix(runif(150 * 3), 150, 3) 
  x_i <- LazyTensor(x, index = 'i') 
  y <- matrix(runif(100 * 3), 100, 3)
  y_j <- LazyTensor(y, index = 'j')
  w <- matrix(runif(150 * 3), 150, 3) # weight LazyTensor
  w_j <- LazyTensor(y, index = 'j')
  
  S_ij = sum( (x_i - y_j)^2 )
  # check formulas, args & classes
  obj <- logsumexp_reduction(S_ij, 'i', w_j)
  expect_false(is.LazyTensor(obj))
  
  obj <- logsumexp(S_ij, 'i')
  expect_false(is.LazyTensor(obj))
  
  # errors
  expect_error(
    logsumexp_reduction(S_ij, '9'),
    "`index` input argument should be a character, either 'i' or 'j'.",
    fixed = TRUE
  )
  
})


test_that("sumsoftmaxweight", {
  # basic example
  x <- matrix(runif(150 * 3), 150, 3) 
  x_i <- LazyTensor(x, index = 'i') 
  y <- matrix(runif(100 * 3), 100, 3)
  y_j <- LazyTensor(y, index = 'j')
  
  V_ij <- x_i - y_j
  S_ij <- sum(V_ij^2)
  
  # check formulas, args & classes
  obj <- sumsoftmaxweight(S_ij, 'i', V_ij)
  expect_false(is.LazyTensor(obj))
  
  obj <- logsumexp(S_ij, 'i')
  expect_false(is.LazyTensor(obj))
  
  # errors
  expect_error(
    sumsoftmaxweight(S_ij, '9', V_ij),
    "`index` input argument should be a character, either 'i' or 'j'.",
    fixed = TRUE
  )
  
})


test_that("sumsoftmaxweight_reduction", {
  # basic example
  x <- matrix(runif(150 * 3), 150, 3) 
  x_i <- LazyTensor(x, index = 'i') 
  y <- matrix(runif(100 * 3), 100, 3)
  y_j <- LazyTensor(y, index = 'j')
  
  V_ij <- x_i - y_j
  S_ij <- sum(V_ij^2)
  
  # check formulas, args & classes
  obj <- sumsoftmaxweight_reduction(S_ij, 'i', V_ij)
  expect_false(is.LazyTensor(obj))
  
  obj <- logsumexp(S_ij, 'i')
  expect_false(is.LazyTensor(obj))
  
  # errors
  expect_error(
    sumsoftmaxweight_reduction(S_ij, '9', V_ij),
    "`index` input argument should be a character, either 'i' or 'j'.",
    fixed = TRUE
  )
  
})






