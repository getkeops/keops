context("LazyTensor operations")

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
  # check the object class
  classes <- c(class(out_i), class(out_j), class(out_u), class(out_D))
  k <- length(classes)
  expect_equal(classes, rep("LazyTensor", k))
  # check object formula and args
  bool_grep_i <- grep("A0x.*i", out_i$formula)
  expect_equal(bool_grep_i, 1)
  bool_grep_j <- grep("A0x.*j", out_j$formula)
  expect_equal(bool_grep_j, 1)
  bool_grep_NA <- grep("A0x.*NA", out_u$formula)
  expect_equal(bool_grep_NA, 1)
  bool_grep_Pm <- grep("A0x.*NA=Pm\\(1\\)", out_D$args)
  expect_equal(bool_grep_Pm, 1)
  # errors
  expect_error(LazyTensor("x"), 
               "`x` input argument should be a matrix, a vector or a scalar.", 
               fixed = TRUE)
  expect_error(LazyTensor(x), 
               "missing `index` argument", 
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
               "`x` input argument should be a LazyTensor, a vector or a scalar.", 
               fixed = TRUE)
})


test_that("binaryop.LazyTensor", {
  # basic example
  D <- 3
  M <- 100
  N <- 150
  x <- matrix(runif(M * D), M, D)
  y <- matrix(runif(N * D), N, D)
  x_i <- LazyTensor(x, index = 'i')
  y_j <- LazyTensor(y, index = 'j')
  
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
  
  obj <-  binaryop.LazyTensor(x_i, 3, "Pow")
  bool_grep_formula <- grep("Pow\\(A0x.*i,3\\)", obj$formula)
  bool_grep_args1 <- grep("A0x.*i=Vi\\(3\\)", obj$args[1])
  expect_equal(bool_grep_formula, 1)
  expect_equal(length(obj$args), 1)
  expect_is(obj, "LazyTensor")
})


# Test operations


test_that("+", {
  # basic example
  D <- 3
  M <- 100
  N <- 150
  x <- matrix(runif(M * D), M, D)
  y <- matrix(runif(N * D), N, D)
  x_i <- LazyTensor(x, index = 'i')
  y_j <- LazyTensor(y, index = 'j')
  
  # check results & formulas
  expect_equal(D + M, 103)
  
  expect_true(class(x_i + y_j) == "LazyTensor")
  
  obj <- x_i + y_j
  bool_grep_formula <- grep("A0x.*i\\+A0x.*j", obj$formula)
  expect_equal(bool_grep_formula, 1)
  
  obj <-  x_i + 3
  bool_grep_formula <- grep("A0x.*i\\+A0x.*NA", obj$formula)
  expect_equal(bool_grep_formula, 1)
  
  obj <-  3.14 + x_i
  bool_grep_formula <- grep("A0x.*NA\\+A0x.*i", obj$formula)
  expect_equal(bool_grep_formula, 1)
})


test_that("-", {
  # basic example
  D <- 3
  M <- 100
  N <- 150
  x <- matrix(runif(M * D), M, D)
  y <- matrix(runif(N * D), N, D)
  x_i <- LazyTensor(x, index = 'i')
  y_j <- LazyTensor(y, index = 'j')
  
  # check results & formulas
  expect_equal(D - D, 0)
  expect_equal(-D, -3)
  
  expect_true(class(x_i - y_j) == "LazyTensor")
  
  obj <- x_i - y_j
  bool_grep_formula <- grep("A0x.*i-A0x.*j", obj$formula)
  expect_equal(bool_grep_formula, 1)
  
  obj <-  x_i - 3
  bool_grep_formula <- grep("A0x.*i-A0x.*NA", obj$formula)
  expect_equal(bool_grep_formula, 1)
  
  obj <-  -x_i
  bool_grep_formula <- grep("Minus\\(A0x.*i\\)", obj$formula)
  expect_equal(bool_grep_formula, 1)
  
  obj <-  3.14 - x_i
  bool_grep_formula <- grep("A0x.*NA-A0x.*i", obj$formula)
  expect_equal(bool_grep_formula, 1)
})


test_that("*", {
  # basic example
  D <- 3
  M <- 100
  N <- 150
  x <- matrix(runif(M * D), M, D)
  y <- matrix(runif(N * D), N, D)
  x_i <- LazyTensor(x, index = 'i')
  y_j <- LazyTensor(y, index = 'j')
  
  # check results & formulas
  expect_equal(D * M, 300)
  
  expect_true(class(x_i * y_j) == "LazyTensor")
  
  obj <- x_i * y_j
  bool_grep_formula <- grep("A0x.*i\\*A0x.*j", obj$formula)
  expect_equal(bool_grep_formula, 1)
  
  obj <-  x_i * 3
  bool_grep_formula <- grep("A0x.*i\\*A0x.*NA", obj$formula)
  expect_equal(bool_grep_formula, 1)
  
  obj <-  3.14 * x_i
  bool_grep_formula <- grep("A0x.*NA\\*A0x.*i", obj$formula)
  expect_equal(bool_grep_formula, 1)
})


test_that("/", {
  # basic example
  D <- 3
  M <- 100
  N <- 150
  x <- matrix(runif(M * D), M, D)
  y <- matrix(runif(N * D), N, D)
  x_i <- LazyTensor(x, index = 'i')
  y_j <- LazyTensor(y, index = 'j')
  
  # check results & formulas
  expect_equal(D / M, 0.03)
  
  expect_true(class(x_i / y_j) == "LazyTensor")
  
  obj <- x_i / y_j
  bool_grep_formula <- grep("A0x.*i/A0x.*j", obj$formula)
  expect_equal(bool_grep_formula, 1)
  
  obj <-  x_i / 3
  bool_grep_formula <- grep("A0x.*i/A0x.*NA", obj$formula)
  expect_equal(bool_grep_formula, 1)
  
  obj <-  3.14 / x_i
  bool_grep_formula <- grep("A0x.*NA/A0x.*i", obj$formula)
  expect_equal(bool_grep_formula, 1)
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
  
  # check results & formulas
  expect_equal(D^D, 27)
  
  expect_true(class(x_i^y_j) == "LazyTensor")
  
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
  
  obj <-  3.14^x_i
  bool_grep_formula <- grep("Powf\\(A0x.*NA,A0x.*i\\)", obj$formula)
  expect_equal(bool_grep_formula, 1)
})



test_that("|", {
  # basic example
  D <- 3
  M <- 100
  N <- 150
  x <- matrix(runif(M * D), M, D)
  y <- matrix(runif(N * D), N, D)
  x_i <- LazyTensor(x, index = 'i')
  y_j <- LazyTensor(y, index = 'j')
  
  # check results & formulas
  expect_equal(D | M, TRUE)
  
  expect_true(class(x_i | y_j) == "LazyTensor")
  
  obj <- x_i | y_j
  bool_grep_formula <- grep("\\(A0x.*i\\|A0x.*j\\)", obj$formula)
  expect_equal(bool_grep_formula, 1)
  
  obj <-  x_i | 3
  bool_grep_formula <- grep("\\(A0x.*i\\|A0x.*NA\\)", obj$formula)
  expect_equal(bool_grep_formula, 1)
  
  obj <-  3.14 | x_i
  bool_grep_formula <- grep("A0x.*NA\\|A0x.*i", obj$formula)
  expect_equal(bool_grep_formula, 1)
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
  
  # check results, formulas & classes
  expect_equal(exp(0), 1)
  expect_true(class(exp(x_i)) == "LazyTensor")
  
  obj <- exp(x_i)
  bool_grep_formula <- grep("Exp\\(A0x.*i\\)", obj$formula)
  expect_equal(bool_grep_formula, 1)
  
  obj <-  exp(x_i - y_j)
  bool_grep_formula <- grep("Exp\\(A0x.*i-A0x.*j\\)", obj$formula)
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
  
  # check results, formulas & classes
  expect_equal(log(1), 0)
  expect_true(class(log(x_i)) == "LazyTensor")
  
  obj <- log(x_i)
  bool_grep_formula <- grep("Log\\(A0x.*i\\)", obj$formula)
  expect_equal(bool_grep_formula, 1)
  
  obj <-  log(x_i - y_j)
  bool_grep_formula <- grep("Log\\(A0x.*i-A0x.*j\\)", obj$formula)
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
  
  # check results, formulas & classes
  expect_equal(inv(1), 1)
  expect_true(class(inv(x_i)) == "LazyTensor")
  
  obj <- inv(x_i)
  bool_grep_formula <- grep("Inv\\(A0x.*i\\)", obj$formula)
  expect_equal(bool_grep_formula, 1)
  
  obj <-  inv(x_i + y_j)
  bool_grep_formula <- grep("Inv\\(A0x.*i\\+A0x.*j\\)", obj$formula)
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
  
  # check results, formulas & classes
  expect_equal(cos(0), 1)
  expect_true(class(cos(x))[1] != "LazyTensor")
  expect_true(class(cos(x_i)) == "LazyTensor")
  
  obj <- cos(x_i)
  bool_grep_formula <- grep("Cos\\(A0x.*i\\)", obj$formula)
  expect_equal(bool_grep_formula, 1)
  
  obj <-  cos(x_i + y_j)
  bool_grep_formula <- grep("Cos\\(A0x.*i\\+A0x.*j\\)", obj$formula)
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
  
  # check results, formulas & classes
  expect_equal(sin(0), 0)
  expect_true(class(sin(x))[1] != "LazyTensor")
  expect_true(class(sin(x_i)) == "LazyTensor")
  
  obj <- sin(x_i)
  bool_grep_formula <- grep("Sin\\(A0x.*i\\)", obj$formula)
  expect_equal(bool_grep_formula, 1)
  
  obj <-  sin(x_i + y_j)
  bool_grep_formula <- grep("Sin\\(A0x.*i\\+A0x.*j\\)", obj$formula)
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
  
  # check results, formulas & classes
  expect_equal(acos(1), 0)
  expect_true(class(acos(x))[1] != "LazyTensor")
  expect_true(class(acos(x_i)) == "LazyTensor")
  
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
  
  # check results, formulas & classes
  expect_equal(atan2(0, 0), 0)
  expect_true(class(atan2(x_i, y_j)) == "LazyTensor")
  
  obj <- atan2(x_i, y_j)
  bool_grep_formula <- grep("Atan2\\(A0x.*i,A0x.*j\\)", obj$formula)
  expect_equal(bool_grep_formula, 1)
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


#test_that("round", {
#  # basic example
#  D <- 3
#  M <- 100
#  N <- 150
#  x <- matrix(runif(M * D), M, D)
#  y <- matrix(runif(N * D), N, D)
#  x_i <- LazyTensor(x, index = 'i')
#  y_j <- LazyTensor(y, index = 'j')
#  
#  # check results, formulas & class
#  expect_equal(sign(-D), -1)
#  expect_equal(class(sign(x))[1] != "LazyTensor", TRUE)
#  expect_equal(class(sign(x_i)) == "LazyTensor", TRUE)
#  
#  obj <- sign(x_i)
#  bool_grep_formula <- grep("Sign\\(A0x.*i\\)", obj$formula)
#  expect_equal(bool_grep_formula, 1)
#  
#  obj <-  sign(-y_j)
#  bool_grep_formula <- grep("Sign\\(Minus\\(A0x.*j\\)\\)", obj$formula)
#  expect_equal(bool_grep_formula, 1)
#})


test_that("min", {
  # basic example
  D <- 3
  M <- 100
  N <- 150
  x <- matrix(runif(M * D), M, D)
  y <- matrix(runif(N * D), N, D)
  x_i <- LazyTensor(x, index = 'i')
  y_j <- LazyTensor(y, index = 'j')
  
  # check results, formulas & classes
  expect_equal(min(D), 3)
  expect_true(class(min(x))[1] != "LazyTensor")
  expect_true(class(min(x_i)) == "LazyTensor")
  
  obj <- min(x_i)
  bool_grep_formula <- grep("Min\\(A0x.*i\\)", obj$formula)
  expect_equal(bool_grep_formula, 1)
  
  # error
  expect_error(min(x_i, y_j), "unused argument (y_j)", fixed = TRUE)
})


test_that("max", {
  # basic example
  D <- 3
  M <- 100
  N <- 150
  x <- matrix(runif(M * D), M, D)
  y <- matrix(runif(N * D), N, D)
  x_i <- LazyTensor(x, index = 'i')
  y_j <- LazyTensor(y, index = 'j')
  
  # check results, formulas & classes
  expect_equal(max(D), 3)
  expect_true(class(max(x))[1] != "LazyTensor")
  expect_true(class(max(x_i)) == "LazyTensor")
  
  obj <- max(x_i)
  bool_grep_formula <- grep("Max\\(A0x.*i\\)", obj$formula)
  expect_equal(bool_grep_formula, 1)
  
  # error
  expect_error(max(x_i, y_j), "unused argument (y_j)", fixed = TRUE)
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
  expect_true(class(xlogx(x))[1] != "LazyTensor")
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
  expect_true(class(sinxdivx(x))[1] != "LazyTensor")
  expect_true(class(sinxdivx(x_i)) == "LazyTensor")
  
  obj <- sinxdivx(x_i)
  bool_grep_formula <- grep("SinXDivX\\(A0x.*i\\)", obj$formula)
  expect_equal(bool_grep_formula, 1)
  
  # TODO manage case when x = 0 with limit at 1 ?
})


# TODO : add other tests : reduction.LazyTensor, sum, Inv, round, sinxdivx (to finish) 



