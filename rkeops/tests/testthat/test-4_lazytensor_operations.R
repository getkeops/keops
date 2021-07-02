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
  # check object formula
  expect_equal(out_i$formula, "Var(0,3,0)")
  expect_equal(out_j$formula, "Var(0,3,1)")
  expect_equal(out_u$formula, "Var(0,100,2)")
  expect_equal(out_D$formula, "Var(0,1,2)")
  # errors
  expect_error(LazyTensor("Var(0,3,0)"), 
               "`x` input argument should be a matrix, a vector or a scalar.", 
               fixed = TRUE)
  expect_error(LazyTensor(x), 
               "missing `index` argument", 
               fixed = TRUE)
})