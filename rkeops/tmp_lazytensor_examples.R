# Basic example

 
D <- 3
M <- 100
N <- 150
E <- 4
x <- matrix(runif(M * D), M, D)
y <- matrix(runif(N * D), N, D)
z <- matrix(runif(N * E), N, E)
b <- matrix(runif(N * E), N, E)

vect <- rep(1, 10)
vect_LT <- LazyTensor(vect)

s <- 0.25
#
## creating LazyTensor from matrices
x_i <- LazyTensor(x, index = 'i')
y_j <- LazyTensor(y, index = 'j')
z_j <- LazyTensor(z, index = 'j')

z <- matrix(1i^ (-6:5), nrow = 4) # complex 4x3 matrix
z_i <- LazyTensor(z, index = 'i', is_complex = TRUE)
conj_z_i <- Conj(z_i)
b_j = b

# Symbolic matrix of squared distances:
SqDist_ij = sum( (x_i - y_j)^2 )

# Symbolic Gaussian kernel matrix:
K_ij = exp( - SqDist_ij / (2 * s^2) )

# Genuine matrix:
v = K_ij %*% b_j
# equivalent
# v = "%*%.LazyTensor"(K_ij, b_j)

s2 = (2 * s^2)
# equivalent
op <- keops_kernel(
    formula = "Sum_Reduction(Exp(Minus(Sum(Square(x-y)))/s)*b,0)",
    args = c("x=Vi(3)", "y=Vj(3)", "s=Pm(1)", "b=Vj(4)")
)

v2 <- op(list(x, y, s2, b))

sum((v2-v)^2)



# we compare to standard R computation
SqDist = 0
onesM = matrix(1, 1, 2)
onesN = matrix(1, 1, 2)

for(k in 1:D) {
    print(SqDist)
    SqDist = SqDist + (x[, k] %*% onesN - t(y[, k] %*% onesM))^2
    print(SqDist)
}
    

K = exp(-SqDist/(2*s^2))

v2 = K %*% b

print(mean(abs(v-v2)))

# ==============================================================================

nx <- 100
ny <- 150
x <- matrix(runif(nx * 3), nrow = nx, ncol = 3) # arbitrary R matrix representing
# 100 data points in R^3
y <- matrix(runif(ny * 3), nrow = ny, ncol = 3) # arbitrary R matrix representing
# 150 data points in R^3
s <- 0.1                                        # scale parameter  

x_i = LazyTensor(x,"i")
y_j = LazyTensor(y,"j")
D_ij = sum((x_i-y_j)^2)
K_ij = exp(-D_ij/s^2)   
res = sum(K_ij,index="i")

# ===================

x <- matrix(1:15, nrow = 1, ncol = 5)
y <- matrix(1:10, nrow = 1, ncol = 5)

formula = "Sum_Reduction(Concat(x, y), 1)"
args <- c("x=Vi(5)", "y=Vj(5)")
op <- keops_kernel(formula = formula, args)
op(list(x, y))


# Sum_reduction x + y ==========================================================

xl <- c(1, 2, 3)
x <- matrix(xl, 3, 2)

fl <- c(1, 2, 3, 1)
f <- matrix(fl, 4, 2)

vect <- c(1, 2)
y <- matrix(vect, nrow = 1, ncol = 2)


vect2 <- c(1, 2, 3, 4, 5)
#vect <- matrix(c(1, 2), nrow = 1)

## creating LazyTensor from matrices and vectors
x_i <- LazyTensor(x, index = 'i')
vect_LT <- LazyTensor(vect)

formula = "Sum_Reduction(x+y, 0)"
args = c("x=Vi(2)", "y=Vj(2)")
op0 <- keops_kernel(formula, args)

formula = "Sum_Reduction(x+y, 1)"
args = c("x=Vi(2)", "y=Vj(2)")
op5 <- keops_kernel(formula, args)

formula = "Sum_Reduction(x+y, 0)"
args = c("x=Vi(2)", "y=Pm(2)")
op1 <- keops_kernel(formula, args)

formula = "Sum_Reduction(x+y, 1)"
args = c("x=Vj(2)", "y=Pm(2)")
op2 <- keops_kernel(formula, args)

formula = "Sum_Reduction(x+y, 1)"
args = c("x=Vi(2)", "y=Pm(2)")
op3 <- keops_kernel(formula, args)

formula = "Sum_Reduction(x+y, 0)"
args = c("x=Vj(2)", "y=Pm(2)")
op4 <- keops_kernel(formula, args)

op1(list(x, vect))
#       [,1] [,2]
# [1,]    2    3
# [2,]    3    4
# [3,]    4    5
op2(list(x, vect))
#       [,1] [,2]
# [1,]    2    3
# [2,]    3    4
# [3,]    4    5
op3(list(x, vect))
#       [,1] [,2]
# [1,]    9   12
op4(list(x, vect))
#       [,1] [,2]
# [1,]    9   12
op0(list(x, y))
op5(list(x, y))

# ==============================================================================

formula = "Sum_Reduction(Elem(f,3),0)"
args = c("f=Pm(3)")
op1 <- keops_kernel(formula, args)

# formulae and args
formula = "Sum_Reduction(MatVecMult(x, y), 0)"
args = c("x=Vi(2)", "y=Pm(2)")
op1 <- keops_kernel(formula, args)

# Sum_Reduction Extract ========================================================

fl <- c(1, 2, 3, 1, 5, 8, 1, 7, 3, 4)
f <- matrix(fl, 5, 4)
#      [,1] [,2] [,3] [,4]
# [1,]    1    8    1    8
# [2,]    2    1    2    1
# [3,]    3    7    3    7
# [4,]    1    3    1    3
# [5,]    5    4    5    4

# ------------------------------

formula = "Sum_Reduction(Extract(x, 1, 3),0)"
args = c("x=Vi(4)")
op7 <- keops_kernel(formula, args)

op7(list(f))
#       [,1] [,2] [,3]
# [1,]    8    1    8
# [2,]    1    2    1
# [3,]    7    3    7
# [4,]    3    1    3
# [5,]    4    5    4

# ------------------------------

formula = "Sum_Reduction(Extract(x, 1, 3),1)"
args = c("x=Vi(4)")
op8 <- keops_kernel(formula, args)

op8(list(f))
#       [,1] [,2] [,3]
# [1,]   23   12   23

# ------------------------------

formula = "Sum_Reduction(Extract(x, 4, 0),1)"
args = c("x=Vi(4)")
op12 <- keops_kernel(formula, args)

op12(list(f))
# [,1]

# ------------------------------

formula = "Sum_Reduction(Extract(x, 4, 0),0)"
args = c("x=Vi(4)")
op13 <- keops_kernel(formula, args)

op13(list(f))
# [1,]
# [2,]
# [3,]
# [4,]
# [5,]

# ------------------------------

formula = "Sum_Reduction(Extract(x, 1, 3),0)"
args = c("x=Pm(4)")
op10 <- keops_kernel(formula, args)

op10(list(c(1, 2, 3, 4)))
#       [,1] [,2] [,3]
# [1,]    2    3    4

# ------------------------------

formula = "Sum_Reduction(Extract(x, 1, 3),1)"
args = c("x=Pm(4)")
op11 <- keops_kernel(formula, args)

op11(list(c(1, 2, 3, 4)))
#       [,1] [,2] [,3]
# [1,]    2    3    4

# Sum_Reduction Extract with LazyTensor ========================================

gl <- c(1, 2, 3, 1, 5, 8, 1, 7, 3, 4,
        1, 2, 4, 3, 9, 3, 7, 5, 0, 4)
g <- matrix(gl, 5, 4)
#      [,1] [,2] [,3] [,4]
# [1,]    1    8    1    3
# [2,]    2    1    2    7
# [3,]    3    7    4    5
# [4,]    1    3    3    0
# [5,]    5    4    9    4

g_i <- LazyTensor(g, index = 'i')

a <- sum(extract(g_i, 1, 3), index = 'i')
#       [,1] [,2] [,3]
# [1,]   23   19   19
b <- sum(extract(g_i, 1, 3), index = 'j')
#       [,1] [,2] [,3]
# [1,]    8    1    3
# [2,]    1    2    7
# [3,]    7    4    5
# [4,]    3    3    0
# [5,]    4    9    4

b <- sum(extract(LazyTensor(c(1, 2, 3, 1, 5)), 2, 3), index = 'j')
b <- sum(extract(LazyTensor(c(1, 2, 3, 1, 5)), 2, 3), index = 'i')

# Sum_Reduction Elem with LazyTensor ===========================================

scal <- 3.14
scal_Pm <- Pm(scal)
# > scal_Pm$args
# [1] "A.*NA=Pm(1)"

x <- matrix(c(1, 2, 3, 4, 5, 6), nrow = 3, ncol = 2)
# > x
#       [,1] [,2]
# [1,]    1    4
# [2,]    2    5
# [3,]    3    6

x_i <- LazyTensor(x, index = 'i')
x_j <- LazyTensor(x, index = 'j')

a <- sum(elem(x_i, 1), index = 'i')
# > a
#       [,1]
# [1,]   15
b <- sum(elem(x_i, 1), index = 'j')
# > b
#       [,1]
# [1,]    4
# [2,]    5
# [3,]    6
c <- sum(elem(x_j, 1), index = 'j')
# > c
#       [,1]
# [1,]   15
d <- sum(elem(x_j, 1), index = 'i')
# > d
#       [,1]
# [1,]    4
# [2,]    5
# [3,]    6

aT <- sum(elemT(scal_Pm, 1, 7), index = 'i')


# Sum_Reduction Concat =========================================================

formula = "Sum_Reduction(Concat(x, y), 1)"
args = c("x=Vi(5)", "y=Vj(4)")
op1 <- keops_kernel(formula, args)

formula = "Sum_Reduction(Concat(x, y), 0)"
args = c("x=Vi(5)", "y=Vj(4)")
op2 <- keops_kernel(formula, args)

d1 = 5
d2 = 4
nx = 1
ny = 1
x = matrix(1:(nx * d1), nrow = nx, ncol = d1)
#      [,1] [,2] [,3] [,4] [,5]
# [1,]    1    2    3    4    5
y = matrix(1:(ny * d2), nrow = ny, ncol = d2)
#      [,1] [,2] [,3] [,4]
# [1,]    1    2    3    4
op1(list(x, y))
#      [,1] [,2] [,3] [,4] [,5] [,6] [,7] [,8] [,9]
# [1,]    1    2    3    4    5    1    2    3    4
op2(list(x, y))
#      [,1] [,2] [,3] [,4] [,5] [,6] [,7] [,8] [,9]
# [1,]    1    2    3    4    5    1    2    3    4

d1 = 5
d2 = 4
nx = 2
ny = 2
x = matrix(1:(nx * d1), nrow = nx, ncol = d1)
#      [,1] [,2] [,3] [,4] [,5]
# [1,]    1    2    3    4    5
# [2,]    2    4    6    8   10
y = matrix(1:(ny * d2), nrow = ny, ncol = d2)
#      [,1] [,2] [,3] [,4]
# [1,]    1    3    5    7
# [2,]    2    4    6    8
op1(list(x, y))
#      [,1] [,2] [,3] [,4] [,5] [,6] [,7] [,8] [,9]
# [1,]    3    7   11   15   19    2    6   10   14
# [2,]    3    7   11   15   19    4    8   12   16
op2(list(x, y))
#      [,1] [,2] [,3] [,4] [,5] [,6] [,7] [,8] [,9]
# [1,]    2    6   10   14   18    3    7   11   15
# [2,]    4    8   12   16   20    3    7   11   15

d1 = 5
d2 = 4
nx = 2
ny = 1
x = matrix(1:(nx * d1), nrow = nx, ncol = d1)
#      [,1] [,2] [,3] [,4] [,5]
# [1,]    1    2    3    4    5
# [2,]    2    4    6    8   10
y = matrix(1:(ny * d2), nrow = ny, ncol = d2)
#      [,1] [,2] [,3] [,4]
# [1,]    1    2    3    4
op1(list(x, y))
#      [,1] [,2] [,3] [,4] [,5] [,6] [,7] [,8] [,9]
# [1,]    3    7   11   15   19    2    4    6    8
op2(list(x, y))
#      [,1] [,2] [,3] [,4] [,5] [,6] [,7] [,8] [,9]
# [1,]    1    3    5    7    9    1    2    3    4
# [2,]    2    4    6    8   10    1    2    3    4

d1 = 5
d2 = 4
nx = 1
ny = 2
x = matrix(1:(nx * d1), nrow = nx, ncol = d1)
#      [,1] [,2] [,3] [,4] [,5]
# [1,]    1    2    3    4    5
y = matrix(1:(ny * d2), nrow = ny, ncol = d2)
#      [,1] [,2] [,3] [,4]
# [1,]    1    3    5    7
# [2,]    2    4    6    8
op1(list(x, y))
#      [,1] [,2] [,3] [,4] [,5] [,6] [,7] [,8] [,9]
# [1,]    1    2    3    4    5    1    3    5    7
# [2,]    1    2    3    4    5    2    4    6    8
op2(list(x, y))
#      [,1] [,2] [,3] [,4] [,5] [,6] [,7] [,8] [,9]
# [1,]    2    4    6    8   10    3    7   11   15

d1 = 5
d2 = 4
nx = 3
ny = 2
x = matrix(1:(nx * d1), nrow = nx, ncol = d1)
#      [,1] [,2] [,3] [,4] [,5]
# [1,]    1    4    7   10   13
# [2,]    2    5    8   11   14
# [3,]    3    6    9   12   15
y = matrix(1:(ny * d2), nrow = ny, ncol = d2)
#      [,1] [,2] [,3] [,4]
# [1,]    1    3    5    7
# [2,]    2    4    6    8
op1(list(x, y))
#      [,1] [,2] [,3] [,4] [,5] [,6] [,7] [,8] [,9]
# [1,]    6   15   24   33   42    3    9   15   21
# [2,]    6   15   24   33   42    6   12   18   24
op2(list(x, y))
#      [,1] [,2] [,3] [,4] [,5] [,6] [,7] [,8] [,9]
# [1,]    2    8   14   20   26    3    7   11   15
# [2,]    4   10   16   22   28    3    7   11   15
# [3,]    6   12   18   24   30    3    7   11   15

d1 = 5
d2 = 4
nx = 2
ny = 3
x = matrix(1:(nx * d1), nrow = nx, ncol = d1)
#      [,1] [,2] [,3] [,4] [,5]
# [1,]    1    3    5    7    9
# [2,]    2    4    6    8   10
y = matrix(1:(ny * d2), nrow = ny, ncol = d2)
#      [,1] [,2] [,3] [,4]
# [1,]    1    4    7   10
# [2,]    2    5    8   11
# [3,]    3    6    9   12
op1(list(x, y))
#      [,1] [,2] [,3] [,4] [,5] [,6] [,7] [,8] [,9]
# [1,]    3    7   11   15   19    2    8   14   20
# [2,]    3    7   11   15   19    4   10   16   22
# [3,]    3    7   11   15   19    6   12   18   24
op2(list(x, y))
#      [,1] [,2] [,3] [,4] [,5] [,6] [,7] [,8] [,9]
# [1,]    3    9   15   21   27    6   15   24   33
# [2,]    6   12   18   24   30    6   15   24   33


