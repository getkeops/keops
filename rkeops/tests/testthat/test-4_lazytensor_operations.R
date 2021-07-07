context("LazyTensor operations")

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
  obj <- unaryop.LazyTensor(x_i, "Square")
  bool_grep_formula <- grep("Square\\(A0x.*i\\)", obj$formula)
  expect_equal(bool_grep_formula, 1)
  # errors
  expect_error(unaryop.LazyTensor(x, "Square"), 
               "`x` input argument should be a LazyTensor, a vector or a scalar.", 
               fixed = TRUE)
})


test_that("binaryop.LazyTensor", {
  D <- 3
  M <- 100
  N <- 150
  x <- matrix(runif(M * D), M, D)
  y <- matrix(runif(N * D), N, D)
  x_i <- LazyTensor(x, index = 'i')
  y_j <- LazyTensor(y, index = 'j')
  
  obj <-  binaryop.LazyTensor(x_i, y_j, "Sum")
  bool_grep_formula <- grep("Sum\\(A0x.*i,A0x.*j\\)", obj$formula)
  expect_equal(bool_grep_formula, 1)
  
  obj <-  binaryop.LazyTensor(x_i, y_j, "-", is_operator = TRUE)
  bool_grep_formula <- grep("A0x.*i-A0x.*j", obj$formula)
  expect_equal(bool_grep_formula, 1)
  
  obj <-  binaryop.LazyTensor(x_i, 3, "Pow")
  bool_grep_formula <- grep("Pow\\(A0x.*i,3\\)", obj$formula)
  expect_equal(bool_grep_formula, 1)
})


test_that("^", {
  D <- 3
  M <- 100
  N <- 150
  x <- matrix(runif(M * D), M, D)
  y <- matrix(runif(N * D), N, D)
  x_i <- LazyTensor(x, index = 'i')
  y_j <- LazyTensor(y, index = 'j')
  # check formulas
  expect_equal(D^D, 27)
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
})


# TODO : add other tests




