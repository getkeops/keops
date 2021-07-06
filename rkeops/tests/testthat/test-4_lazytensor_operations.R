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
  expect_equal(out_i$formula, "var0")
  expect_equal(out_j$args, "var0=Vj(3)")
  expect_equal(out_u$args, "var0=Pm(100)")
  expect_equal(out_D$args, "var0=Pm(1)")
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
  expect_equal(obj$formula, "Square(var0)")
  
  # errors
  expect_error(unaryop.LazyTensor(x, "Square"), 
               "`x` input argument should be a LazyTensor, a vector or a scalar.", 
               fixed = TRUE)
})


# TODO : add other tests
test_that("binaryop.LazyTensor", {
  D <- 3
  M <- 100
  N <- 150
  x <- matrix(runif(M * D), M, D)
  y <- matrix(runif(N * D), N, D)
  x_i <- LazyTensor(x, index = 'i')
  y_j <- LazyTensor(y, index = 'j')
  
  obj <-  binaryop.LazyTensor(x_i, y_j, "Sum")
  expect_equal(obj$formula, "Sum(var0,var1)")
  
  obj <-  binaryop.LazyTensor(x_i, y_j, "-", is_operator = TRUE)
  expect_equal(obj$formula, "var0-var1")
  
  obj <-  binaryop.LazyTensor(x_i, 3, "Pow")
  expect_equal(obj$formula, "Pow(var0,3)")
})


test_that("^", {
  D <- 3
  M <- 100
  N <- 150
  x <- matrix(runif(M * D), M, D)
  y <- matrix(runif(N * D), N, D)
  x_i <- LazyTensor(x, index = 'i')
  y_j <- LazyTensor(y, index = 'j')
  
  expect_equal(D^D, 27)
  obj1 <- x_i^y_j
  expect_equal(obj1$formula, "Powf(var0,var1)")
  obj2 <-  x_i^3
  expect_equal(obj2$formula, "Pow(var0,3)")
  obj3 <-  x_i^(-0.5)
  expect_equal(obj3$formula, "Pow(var0,-0.5)")
  obj4 <-  x_i^(0.5)
  expect_equal(obj3$formula, "Sqrt(var0)")
  obj5 <-  x_i^2
  expect_equal(obj3$formula, "Square(var0)")
  obj6 <-  x_i^0.314
  expect_equal(obj3$formula, "Powf(var0,0.314)") # TODO change Pow in Powf
})





