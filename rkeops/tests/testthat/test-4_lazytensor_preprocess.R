context("LazyTensor preprocess")


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
  still_good_z_Vi <- LazyTensor(z, index = 'i') # without specifying 
  # "is_complex = TRUE": should work as well.
  
  # check the object class
  classes <- c(class(out_i), class(out_j), class(out_u), class(out_D))
  k <- length(classes)
  expect_equal(classes, rep("LazyTensor", k))
  
  expect_true(is.ComplexLazyTensor(z_i))
  expect_true(is.ComplexLazyTensor(still_good_z_Vi))
  
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
  still_good_z_Vi <- Vi(z) # without specifying 
  # "is_complex = TRUE": should work as well.
  
  # check arguments 
  expect_true(x_i$args == x_Vi$args)
  expect_true(z_i$args == z_Vi$args)
  expect_true(z_i$args == still_good_z_Vi$args)
  expect_true(is.LazyTensor(x_Vi))
  expect_true(is.ComplexLazyTensor(z_Vi))
  expect_true(is.ComplexLazyTensor(still_good_z_Vi))
  
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
  still_good_z_Vj <- Vj(z) # without specifying 
  # "is_complex = TRUE": should work as well.
  
  # check arguments 
  expect_true(x_j$args == x_Vj$args)
  expect_true(z_j$args == z_Vj$args)
  expect_true(z_j$args == still_good_z_Vj$args)
  expect_true(is.LazyTensor(x_Vi))
  expect_true(is.ComplexLazyTensor(z_Vi))
  expect_true(is.ComplexLazyTensor(still_good_z_Vi))
  
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
  # check arguments 
  expect_true(D_LT$args == D_Pm$args)
  expect_true(is.LazyTensor(D_Pm))
  
  u_LT <- LazyTensor(u)
  u_Pm <- Pm(u)
  # check arguments 
  expect_true(u_LT$args == u_Pm$args)
  expect_true(is.LazyTensor(u_Pm))
  
  z_LT <- LazyTensor(z)
  z_Pm <- Pm(z)
  # check arguments 
  expect_true(z_LT$args == z_Pm$args)
  expect_true(is.ComplexLazyTensor(z_Pm))
  
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
  z <- matrix(1i^ (-6:5), nrow = 4) # complex 4x3 matrix
  # ComplexLazyTensor
  z_j <- LazyTensor(z, index = 'j', is_complex = TRUE)
  
  # check formulas, args & classes
  obj <- unaryop.LazyTensor(x_i, "Square")
  bool_grep_formula <- grep("Square\\(A0x.*i\\)", obj$formula)
  bool_grep_args <- grep("A0x.*i=Vi\\(3\\)", obj$args[1])
  expect_equal(bool_grep_formula, 1)
  expect_equal(bool_grep_args, 1)
  expect_true(is.LazyTensor(obj))
  
  obj <- unaryop.LazyTensor(z_j, "Square")
  expect_true(is.ComplexLazyTensor(obj))
  
  obj <- unaryop.LazyTensor(z_j, "ComplexAbs", res_type = "LazyTensor")
  expect_false(is.ComplexLazyTensor(obj))
  expect_true(is.LazyTensor(obj))
  
  # errors
  expect_error(unaryop.LazyTensor(x, "Square"), 
               paste("`x` input argument should be a LazyTensor, a vector or a scalar.",
                     "\nIf you want to use a matrix, convert it to LazyTensor first.",
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
  y_j <- LazyTensor(y, index = 'j')
  z_j <- LazyTensor(z, index = 'j')
  # ComplexLazyTensor
  xc_i <- LazyTensor(x, index = 'i', is_complex = TRUE)
  
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
  bool_grep_formula <- grep("Powf\\(A0x.*i,IntCst\\(3\\)\\)", obj$formula)
  bool_grep_args1 <- grep("A0x.*i=Vi\\(3\\)", obj$args[1])
  expect_equal(bool_grep_formula, 1)
  expect_equal(bool_grep_args1, 1)
  expect_equal(length(obj$args), 2)
  expect_is(obj, "LazyTensor")
  
  obj <-  binaryop.LazyTensor(xc_i, x_i, "+")
  expect_true(is.ComplexLazyTensor(obj))
  
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
               paste(
                 "`", "z", 
                 "` input argument should be a LazyTensor, a ComplexLazyTensor,",
                 " a vector or a scalar.",
                 "\nIf you want to use a matrix, convert it to LazyTensor first.", 
                 sep = ""
               ), 
               fixed = TRUE)
  expect_error(ternaryop.LazyTensor(x, y_j, z, "Clamp"), 
               paste(
                 "`", "x", 
                 "` input argument should be a LazyTensor, a ComplexLazyTensor,",
                 " a vector or a scalar.",
                 "\nIf you want to use a matrix, convert it to LazyTensor first.", 
                 sep = ""
               ), 
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
  expect_true(is.LazyTensor(z_i)) # a ComplexLazyTensor is a LazyTensor
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


test_that("check_index", {
  expect_is(check_index("i"), "logical")
  
  expect_true(check_index("i"))
  expect_true(check_index("j"))
  
  expect_false(check_index(5))
  expect_false(check_index("n"))
})
