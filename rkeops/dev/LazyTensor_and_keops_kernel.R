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
op_ij <- keops_kernel(formula, args)

formula = "Sum_Reduction(x+y, 0)"
args = c("x=Vi(2)", "y=Vi(2)")
op_ii <- keops_kernel(formula, args)

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


op_ij(list(x, f))
#       [,1] [,2]
# [1,]   11   11
# [2,]   15   15
# [3,]   19   19

op_ii(list(x, f))
# Error in r_genred(input, param) : 
#     [KeOps] Wrong value of the 'i' dimension 0for arg at position 1 : is 4 but was 3 in previous 'i' arguments. 


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

# matvecmult  ==================================================================

xl <- c(1, 2, 3)
X <- matrix(xl, 3, 3)
#       [,1] [,2] [,3]
# [1,]    1    1    1
# [2,]    2    2    2
# [3,]    3    3    3

v_Pm_1 <- c(2)
v_Pm_n <- c(3, 4, 5)

# With Pm(1) (vector of length 1)
formula = "Sum_Reduction(MatVecMult(x, y), 1)"
args = c("x=Vi(3)", "y=Pm(1)")
op1_Pm_1 <- keops_kernel(formula, args)

formula = "Sum_Reduction(MatVecMult(x, y), 0)"
args = c("x=Vi(3)", "y=Pm(1)")
op2_Pm_1 <- keops_kernel(formula, args)

# With Pm(n) (vector of length n)
formula = "Sum_Reduction(MatVecMult(x, y), 1)"
args = c("x=Vi(3)", "y=Pm(3)")
op1_Pm_n <- keops_kernel(formula, args)

formula = "Sum_Reduction(MatVecMult(x, y), 0)"
args = c("x=Vi(3)", "y=Pm(3)")
op2_Pm_n <- keops_kernel(formula, args)

# With Vi(1) (matrix of inner dim 1)
formula <- "Sum_Reduction(MatVecMult(x, y), 1)"
args <- c("x=Vi(1)", "y=Pm(1)")
op1_Vi_1 <- keops_kernel(formula, args)

formula <- "Sum_Reduction(MatVecMult(x, y), 1)"
args <- c("x=Vi(1)", "y=Pm(3)")
op2_Vi_1 <- keops_kernel(formula, args)
# Error in compile_formula(formula, var_aliases$var_aliases, dllname) : 
#     Error during cmake call. 

# reductions:

op1_Pm_1(list(X, v_Pm_1))
#       [,1] [,2] [,3]
# [1,]   12   12   12

op2_Pm_1(list(X, v_Pm_1))
#       [,1] [,2] [,3]
# [1,]    2    2    2
# [2,]    4    4    4
# [3,]    6    6    6

op1_Pm_n(list(X, v_Pm_n))
#       [,1]
# [1,]   72

op2_Pm_n(list(X, v_Pm_n))
#       [,1]
# [1,]   12
# [2,]   24
# [3,]   36


# vecmatmult  ==================================================================

xl <- c(1, 2, 3)
X <- matrix(xl, 3, 3)
#       [,1] [,2] [,3]
# [1,]    1    1    1
# [2,]    2    2    2
# [3,]    3    3    3

v_Pm_1 <- c(2)
v_Pm_n <- c(3, 4, 5)

# With Pm(1) (vector of length 1)
formula <- "Sum_Reduction(VecMatMult(x, y), 1)"
args <- c("x=Pm(1)", "y=Vi(3)")
op1_Pm_1 <- keops_kernel(formula, args)

formula <- "Sum_Reduction(VecMatMult(x, y), 0)"
args <- c("x=Pm(1)", "y=Vi(3)")
op2_Pm_1 <- keops_kernel(formula, args)

# With Pm(n) (vector of length n)
formula <- "Sum_Reduction(VecMatMult(x, y), 1)"
args <- c("x=Pm(3)", "y=Vi(3)")
op1_Pm_n <- keops_kernel(formula, args)

formula <- "Sum_Reduction(VecMatMult(x, y), 0)"
args <- c("x=Pm(3)", "y=Vi(3)")
op2_Pm_n <- keops_kernel(formula, args)

# With Vi(1) (matrix of inner dim 1)
formula <- "Sum_Reduction(VecMatMult(x, y), 1)"
args <- c("x=Pm(1)", "y=Vi(1)")
op1_Vi_1 <- keops_kernel(formula, args)

formula <- "Sum_Reduction(VecMatMult(x, y), 1)"
args <- c("x=Pm(3)", "y=Vi(1)")
op2_Vi_1 <- keops_kernel(formula, args)
# Error in compile_formula(formula, var_aliases$var_aliases, dllname) : 
#     Error during cmake call. 

# reductions:

op1_Pm_1(list(v_Pm_1, X))
#       [,1] [,2] [,3]
# [1,]   12   12   12

op2_Pm_1(list(v_Pm_1, X))
#       [,1] [,2] [,3]
# [1,]    2    2    2
# [2,]    4    4    4
# [3,]    6    6    6

op1_Pm_n(list(v_Pm_n, X))
#       [,1]
# [1,]   72

op2_Pm_n(list(v_Pm_n, X))
#       [,1]
# [1,]   12
# [2,]   24
# [3,]   36


# Tensorprod ===================================================================

xl <- c(1, 2, 3)
xl2 <- c(1, 2, 3, 4)
X <- matrix(xl, 3, 3)
#       [,1] [,2] [,3]
# [1,]    1    1    1
# [2,]    2    2    2
# [3,]    3    3    3

X2 <- matrix(xl2, 4, 3)
#       [,1] [,2] [,3]
# [1,]    1    1    1
# [2,]    2    2    2
# [3,]    3    3    3
# [4,]    4    4    4

X3 <- matrix(xl2, 4, 4)
#       [,1] [,2] [,3] [,4]
# [1,]    1    1    1    1
# [2,]    2    2    2    2
# [3,]    3    3    3    3
# [4,]    4    4    4    4

v_Pm_1 <- c(2)
v_Pm_n <- c(3, 4, 5)

formula <- "Sum_Reduction(TensorProd(x, y), 1)"
args <- c("x=Pm(1)", "y=Vi(3)")
op1_Pm_1 <- keops_kernel(formula, args)

formula <- "Sum_Reduction(TensorProd(x, y), 1)"
args <- c("x=Pm(3)", "y=Vi(3)")
op1_Pm_n <- keops_kernel(formula, args)

formula <- "Sum_Reduction(TensorProd(x, y), 1)"
args <- c("x=Vi(3)", "y=Vi(3)")
op1_ViVi_n <- keops_kernel(formula, args)

formula <- "Sum_Reduction(TensorProd(x, y), 1)"
args <- c("x=Vi(3)", "y=Vj(3)")
op1_ViVj_n <- keops_kernel(formula, args)

formula <- "Sum_Reduction(TensorProd(x, y), 1)"
args <- c("x=Vi(3)", "y=Vj(4)")
op1_ViVj_nm <- keops_kernel(formula, args)

op1_Pm_1(list(v_Pm_1, X))
#       [,1] [,2] [,3]
# [1,]   12   12   12

op1_Pm_n(list(v_Pm_n, X))
#       [,1] [,2] [,3] [,4] [,5] [,6] [,7] [,8] [,9]
# [1,]   18   24   30   18   24   30   18   24   30

op1_Pm_n(list(v_Pm_n, X2))
#       [,1] [,2] [,3] [,4] [,5] [,6] [,7] [,8] [,9]
# [1,]   30   40   50   30   40   50   30   40   50

op1_ViVi_n(list(X, X2))
# Error in r_genred(input, param) : 
#     [KeOps] Wrong value of the 'i' dimension 0for arg at position 1 : is 4 but was 3 in previous 'i' arguments. 

op1_ViVj_n(list(X, X2)) # different indexing, different nrow but same ncol, 
#       [,1] [,2] [,3] [,4] [,5] [,6] [,7] [,8] [,9]
# [1,]    6    6    6    6    6    6    6    6    6
# [2,]   12   12   12   12   12   12   12   12   12
# [3,]   18   18   18   18   18   18   18   18   18
# [4,]   24   24   24   24   24   24   24   24   24

op1_ViVj_nm(list(X2, X3))
#       [,1] [,2] [,3] [,4] [,5] [,6] [,7] [,8] [,9] [,10] [,11] [,12]
# [1,]   10   10   10   10   10   10   10   10   10    10    10    10
# [2,]   20   20   20   20   20   20   20   20   20    20    20    20
# [3,]   30   30   30   30   30   30   30   30   30    30    30    30
# [4,]   40   40   40   40   40   40   40   40   40    40    40    40


# %*% ==========================================================================

xl <- c(1, 2, 3)
xl2 <- c(1, 2, 3, 4)
X <- matrix(xl, 3, 3)
#       [,1] [,2] [,3]
# [1,]    1    1    1 
# [2,]    2    2    2 
# [3,]    3    3    3 

Y <- matrix(c(3, 1, 4, 5, 1, 1), nrow = 3, ncol = 3)
#       [,1] [,2] [,3]
# [1,]    3    5    3
# [2,]    1    1    1
# [3,]    4    1    4

X2 <- matrix(xl2, 4, 3)
#       [,1] [,2] [,3]
# [1,]    1    1    1
# [2,]    2    2    2
# [3,]    3    3    3
# [4,]    4    4    4

X3 <- matrix(xl2, 4, 4)
#       [,1] [,2] [,3] [,4]
# [1,]    1    1    1    1
# [2,]    2    2    2    2
# [3,]    3    3    3    3
# [4,]    4    4    4    4


formula <- "Sum_Reduction(x*y, 1)"
args <- c("x=Vi(3)", "y=Vi(3)")
op2 <- keops_kernel(formula, args)

formula <- "Sum_Reduction(x*y, 0)"
args <- c("x=Vi(3)", "y=Vi(3)")
op3_ii <- keops_kernel(formula, args)

formula <- "Sum_Reduction(x*y, 0)"
args <- c("x=Vi(3)", "y=Vj(3)")
op3_ij <- keops_kernel(formula, args)

formula <- "Sum_Reduction(x*y, 0)"
args <- c("x=Vi(3)", "y=Pm(3)")
op3_iPm <- keops_kernel(formula, args)

formula <- "Sum_Reduction(x*y, 0)"
args <- c("x=Vj(3)", "y=Pm(3)")
op3_jPm <- keops_kernel(formula, args)

formula <- "Sum_Reduction(x*y, 0)"
args <- c("x=Vi(3)", "y=Pm(1)")
op3_Pm_one <- keops_kernel(formula, args)

op3_iPm(list(X2, xl))
#       [,1] [,2] [,3]
# [1,]    1    2    3
# [2,]    2    4    6
# [3,]    3    6    9
# [4,]    4    8   12

op3_jPm(list(X2, xl))
#       [,1] [,2] [,3]
# [1,]   10   20   30

op3_ij(list(X, X2))

op2(list(X, X))
#       [,1] [,2] [,3]
# [1,]   14   14   14

op3_ii(list(X, X))
#       [,1] [,2] [,3]
# [1,]    1    1    1
# [2,]    4    4    4
# [3,]    9    9    9

op3_ii(list(X, Y))
#       [,1] [,2] [,3]
# [1,]    3    5    3
# [2,]    2    2    2
# [3,]   12    3   12

op3_ij(list(X, Y))
#       [,1] [,2] [,3]
# [1,]    8    7    8
# [2,]   16   14   16
# [3,]   24   21   24

op3_ij(list(X2, X2))
#       [,1] [,2] [,3]
# [1,]   10   10   10
# [2,]   20   20   20
# [3,]   30   30   30
# [4,]   40   40   40

op3_ii(list(X2, X2))
#       [,1] [,2] [,3]
# [1,]    1    1    1
# [2,]    4    4    4
# [3,]    9    9    9
# [4,]   16   16   16

op3_ij(list(X, X2))
#       [,1] [,2] [,3]
# [1,]   10   10   10
# [2,]   20   20   20
# [3,]   30   30   30

# Addition & sum reduction =====================================================

X <- matrix(xl, 3, 3)
#       [,1] [,2] [,3]
# [1,]    1    1    1
# [2,]    2    2    2
# [3,]    3    3    3

formula = "Sum_Reduction(x+y, 1)"
args = c("x=Vi(3)", "y=Vj(3)")
op2 <- keops_kernel(formula, args)

op2(list(X, X))
#       [,1] [,2] [,3]
# [1,]    9    9    9
# [2,]   12   12   12
# [3,]   15   15   15

formula = "Sum_Reduction(x+y, 0)"
args = c("x=Vi(3)", "y=Vj(3)")
op3 <- keops_kernel(formula, args)

op3(list(X, X))
#       [,1] [,2] [,3]
# [1,]    9    9    9
# [2,]   12   12   12
# [3,]   15   15   15

formula = "Sum_Reduction(x+y, 0)"
args = c("x=Vi(3)", "y=Vi(3)")
op4 <- keops_kernel(formula, args)


op4(list(X, X))
#       [,1] [,2] [,3]
# [1,]    2    2    2
# [2,]    4    4    4
# [3,]    6    6    6

formula = "Sum_Reduction(x+y, 1)"
args = c("x=Vi(3)", "y=Vi(3)")
op5 <- keops_kernel(formula, args)

op5(list(X, X))
#       [,1] [,2] [,3]
# [1,]   12   12   12

# Unweighted/Weighted squared norm and distance ================================

x <- matrix(c(1, 2, 3, 4), nrow = 4, ncol = 3)
#      [,1] [,2] [,3]
# [1,]    1    1    1
# [2,]    2    2    2
# [3,]    3    3    3
# [4,]    4    4    4
x_i <- LazyTensor(x, index = 'i')

y <- matrix(c(1, 1, 1, 1), nrow = 4, ncol = 3)
#      [,1] [,2] [,3]
# [1,]    1    1    1
# [2,]    1    1    1
# [3,]    1    1    1
# [4,]    1    1    1
y_i <- LazyTensor(y, index = 'i')

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

# ------------------------------

z <- matrix(2 + 1i^(-6:5), nrow = 4)
#      [,1] [,2] [,3]
# [1,] 1+0i 1+0i 1+0i
# [2,] 2-1i 2-1i 2-1i
# [3,] 3+0i 3+0i 3+0i
# [4,] 2+1i 2+1i 2+1i
z_i <- LazyTensor(z, index = 'i', is_complex = TRUE)
#      [,1] [,2] [,3] [,4] [,5] [,6]
# [1,]    1    0    1    0    1    0
# [2,]    2   -1    2   -1    2   -1
# [3,]    3    0    3    0    3    0
# [4,]    2    1    2    1    2    1

formula <- "Sum_Reduction(Extract(z, 1, 2),0)"
args <- c("z=Vi(3)")
op14 <- keops_kernel(formula, args)

op14(list(z))
#      [,1] [,2]
# [1,]    1    1
# [2,]    2    2
# [3,]    3    3
# [4,]    2    2
# Warning message:
#     In r_genred(input, param) : imaginary parts discarded in coercion


OP_i <- sum(extract(z_i, 1, 2), index = 'i')
#       [,1] [,2]
# [1,]    0    8

OP_j <- sum(extract(z_i, 1, 2), index = 'j')
#       [,1] [,2]
# [1,]    0    1
# [2,]   -1    2
# [3,]    0    3
# [4,]    1    2

# Sum_Reduction x + y with different nrow ======================================

x <- matrix(c(1, 2, 3, 4), nrow = 4, ncol = 3)
y <- matrix(c(1, 2, 3), nrow = 3, ncol = 3)

formula <- "Sum_Reduction(x + y,1)"
args <- c("x=Vi(3)", "y=Vi(3)")
op15 <- keops_kernel(formula, args)

op15(list(x, y)) # Consistent error (nothing to fix: you don't add together two matrices with different nrows)
# Error in r_genred(input, param) : 
#     [KeOps] Wrong value of the 'i' dimension 0for arg at position 1 : is 3 but was 4 in previous 'i' arguments. 


# Sum_Reduction x + y with Complexes ===========================================

z <- matrix(2 + 3i, nrow = 4, ncol = 3)
#      [,1] [,2] [,3]
# [1,] 2+3i 2+3i 2+3i
# [2,] 2+3i 2+3i 2+3i
# [3,] 2+3i 2+3i 2+3i
# [4,] 2+3i 2+3i 2+3i
z2 <- matrix(1 + 1i, nrow = 4, ncol = 3)
#      [,1] [,2] [,3]
# [1,] 1+1i 1+1i 1+1i
# [2,] 1+1i 1+1i 1+1i
# [3,] 1+1i 1+1i 1+1i
# [4,] 1+1i 1+1i 1+1i
x <- matrix(c(1, 2, 3, 4), nrow = 4, ncol = 3)
#      [,1] [,2] [,3]
# [1,]    1    1    1
# [2,]    2    2    2
# [3,]    3    3    3
# [4,]    4    4    4

# -----------------------

formula = "Sum_Reduction(x + z,1)"
args = c("x=Vi(3)", "z=Vi(3)")
op17 <- keops_kernel(formula, args)

op17(list(x, z))
#       [,1] [,2] [,3]
# [1,]   18   18   18
# Warning message:
#     In r_genred(input, param) : imaginary parts discarded in coercion

# -----------------------

formula = "Sum_Reduction(z + z2,1)"
args = c("z=Vi(3)", "z2=Vi(3)")
op18 <- keops_kernel(formula, args)

op18(list(z, z2))
#      [,1] [,2] [,3]
# [1,]   12   12   12
# Warning messages:
# 1: In r_genred(input, param) : imaginary parts discarded in coercion
# 2: In r_genred(input, param) : imaginary parts discarded in coercion





# IntCst ==============

Scal <- 2
LT_Scal <- Pm(Scal)
z <- matrix(2 + 3i, nrow = 4, ncol = 3)


formula = "Sum_Reduction(z+p,0)"
args = c("z=Vi(6)", "p=Pm(1)")
op <- keops_kernel(formula, args)

formula = "Sum_Reduction(z+IntCst(4),0)"
args = c("z=Vi(6)")
op <- keops_kernel(formula, args)

op(list(z, 4))

a <- list()


# Sum_Reduction Extract with Complex ===========================================

z <- matrix(2 + 3i, nrow = 4, ncol = 3)
formula = "Sum_Reduction(Extract(z, 1, 2),0)"
args = c("z=Vi(3)")
op14 <- keops_kernel(formula, args)

op14(list(z))
#       [,1] [,2]
# [1,]   -1   -1
# [2,]    0    0
# [3,]    1    1
# [4,]    0    0
# Warning message:
#     In r_genred(input, param) : imaginary parts discarded in coercion


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

# Sum_Reduction with new ComplexLazyensor ======================================

# Matrices
x <- matrix(c(1, 2, 3, 4), nrow = 4, ncol = 3)
#      [,1] [,2] [,3]
# [1,]    1    1    1
# [2,]    2    2    2
# [3,]    3    3    3
# [4,]    4    4    4
x_i <- LazyTensor(x, index = 'i')
x_j <- LazyTensor(x, index = 'j')

y <- matrix(c(1, 7, 3, 4), nrow = 4, ncol = 3)
#      [,1] [,2] [,3]
# [1,]    1    1    1
# [2,]    7    7    7
# [3,]    3    3    3
# [4,]    4    4    4
y_i <- LazyTensor(y, index = 'i')

z <- matrix(2 + 1i^(-6:5), nrow = 4)
#      [,1] [,2] [,3]
# [1,] 1+0i 1+0i 1+0i
# [2,] 2-1i 2-1i 2-1i
# [3,] 3+0i 3+0i 3+0i
# [4,] 2+1i 2+1i 2+1i
z_i <- LazyTensor(z, index = 'i', is_complex = TRUE)
# > z_i$vars
# [[1]]
#       [,1] [,2] [,3] [,4] [,5] [,6]
# [1,]    1    0    1    0    1    0
# [2,]    2   -1    2   -1    2   -1
# [3,]    3    0    3    0    3    0
# [4,]    2    1    2    1    2    1
z_j <- LazyTensor(z, index = 'j', is_complex = TRUE)

z2 <- matrix(3 + 1i^(-6:5), nrow = 4)
#      [,1] [,2] [,3]
# [1,] 2+0i 2+0i 2+0i
# [2,] 3-1i 3-1i 3-1i
# [3,] 4+0i 4+0i 4+0i
# [4,] 3+1i 3+1i 3+1i
z2_i <- LazyTensor(z2, index = 'i', is_complex = TRUE)
z2_j <- LazyTensor(z2, index = 'j', is_complex = TRUE)

# Matrix/matrix sums -----------------------------------------------------------

Sum_xz <- x_i + z_i
# > Sum_xz$formula
# [1] "Add(Real2Complex(A0x55ec34f7e7f8i),A0x7f5e90004440i)"

R2C <- sum(real2complex(x_i), index = 'j')
# No sum is done because nothing is indexed by 'j', but this is
# just to see the result:

#       [,1] [,2] [,3] [,4] [,5] [,6]
# [1,]    1    0    1    0    1    0
# [2,]    2    0    2    0    2    0
# [3,]    3    0    3    0    3    0
# [4,]    4    0    4    0    4    0

sum_z_i_z_j <- sum(z_i + z_j, index = 'i')
# Error in op(c(x$vars)) : 
#     The number of elements in the 'input' argument does not correspond to the number input arguments in the formula. 

sum_z_i_z_i <- sum(z_i + z_i, index = 'i')
#       [,1] [,2] [,3] [,4] [,5] [,6]
# [1,]   16    0   16    0   16    0

sum_z_i_z2_i <- sum(z_i + z2_i, index = 'i')
#       [,1] [,2] [,3] [,4] [,5] [,6]
# [1,]   20    0   20    0   20    0

sum_z_i_z2_j <- sum(z_i + z2_j, index = 'i')
#       [,1] [,2] [,3] [,4] [,5] [,6]
# [1,]   16    0   16    0   16    0
# [2,]   20   -4   20   -4   20   -4
# [3,]   24    0   24    0   24    0
# [4,]   20    4   20    4   20    4

sum_x_i_z_i <- sum(x_i + z_i, index = 'j') # j-indexed to see
#       [,1] [,2] [,3] [,4] [,5] [,6]
# [1,]    2    0    2    0    2    0
# [2,]    4   -1    4   -1    4   -1
# [3,]    6    0    6    0    6    0
# [4,]    6    1    6    1    6    1

sum_x_i_z_j_on_i <- sum(x_i + z_j, index = 'i')
#       [,1] [,2] [,3] [,4] [,5] [,6]
# [1,]   14    0   14    0   14    0
# [2,]   18   -4   18   -4   18   -4
# [3,]   22    0   22    0   22    0
# [4,]   18    4   18    4   18    4

sum_x_i_z_j_on_j <- sum(x_i + z_j, index = 'j') # j-indexed to see
#      [,1] [,2] [,3] [,4] [,5] [,6]
# [1,]   12    0   12    0   12    0
# [2,]   16    0   16    0   16    0
# [3,]   20    0   20    0   20    0
# [4,]   24    0   24    0   24    0


# Vectors/vector sums ----------------------------------------------------------

# Vectors
Z <- c(2 + 3i, 1 + 1i, 4 + 9i)
Pm_Z <- LazyTensor(Z)

Z2 <- c(3 + 3i, 1 + 1i, 4 + 9i)
Pm_Z2 <- LazyTensor(Z2)

V <- c(5, 6, 7)
Pm_V <- LazyTensor(V)

# Vector/vector addition ---
Sum_ZV <- Pm_Z + Pm_V
# > Sum_ZV$formula
# [1] "Add(A0x55c7cff96a28NA,Real2Complex(A0x55c7d0124dc8NA))"


sum_Pm_Z_Pm_V_on_i <- sum(Pm_Z + Pm_V, index = 'i')
#       [,1] [,2] [,3] [,4] [,5] [,6]
# [1,]    7    3    7    1   11    9

sum_Pm_Z_Pm_V_on_j <- sum(Pm_Z + Pm_V, index = 'j')
#       [,1] [,2] [,3] [,4] [,5] [,6]
# [1,]    7    3    7    1   11    9

sum_Pm_Z_Pm_Z2 <- sum(Pm_Z + Pm_Z2, index = 'i')
#       [,1] [,2] [,3] [,4] [,5] [,6]
# [1,]    5    6    2    2    8   18


# Single value/Single value sums -----------------------------------------------

# Single Values
Cplx <- LazyTensor(2 + 9i)
Cplx2 <- LazyTensor(3 + 7i)
Scal <- LazyTensor(2.4)

A <- Cplx + Scal
# > A$formula
# [1] "Add(A0x55c7cfeb3408NA,Real2Complex(A0x55c7d34eca68NA))"

sum_Cplx_Cplx2 <- sum(Cplx + Cplx2, index = 'j')
#       [,1] [,2]
# [1,]    5   16

sum_Cplx_Scal <- sum(Cplx + Scal, index = 'j')
#       [,1] [,2]
# [1,]  4.4    9 


# Matrix/single value sums -----------------------------------------------------

sum_z_i_Cplx <- sum(z_i + Cplx, index = 'i')
# Error in compile_formula(formula, var_aliases$var_aliases, dllname) : 
#     Error during cmake call.

sum_z_i_Scal <- sum(z_i + Scal, index = 'i')
# Error in compile_formula(formula, var_aliases$var_aliases, dllname) : 
#     Error during cmake call.

sum_x_i_Scal <- sum(x_i + Scal, index = 'i')
#       [,1] [,2] [,3]
# [1,] 19.6 19.6 19.6

sum_x_i_Cplx <- sum(x_i + Cplx, index = 'i')
# Error in compile_formula(formula, var_aliases$var_aliases, dllname) : 
#     Error during cmake call.

A <- x_i + Cplx
# > A$formula
# [1] "Add(Real2Complex(A0x55c7d001c3c8i),A0x55c7cfeb3408NA)"


# --- attempt with "formula-args" syntax ---

# formula <- "Sum_Reduction(Add(Real2Complex(s), z),1)"
# args <- c("Real2Complex(s)=Pm(1)", "z=Pm(1)") # idem with  c("z=Pm(2)", "s=Pm(1)")
# op_Scal_Cplx <- keops_kernel(formula, args)

formula <- "Sum_Reduction(Add(Real2Complex(s), z),1)"
args <- c("s=Pm(1)", "z=Pm(1)") # idem with  c("z=Pm(2)", "s=Pm(1)")
sum_Scal_Cplx <- keops_kernel(formula, args)

sum_Scal_Cplx(list(Cplx, Scal))
# Error in r_genred(input, param) : 
#     Not compatible with requested type: [type=list; target=double]. 

formula <- "Sum_Reduction(Add(z1,z2),1)"
args <- c("z1=Pm(2)", "z2=Pm(2)") # error with c("z1=Pm(1)", "z2=Pm(1)")
sum_Cplx_Cplx <- keops_kernel(formula, args)

sum_Cplx_Cplx(list(Cplx, Cplx2))
# Error in r_genred(input, param) : 
#     Not compatible with requested type: [type=list; target=double]. 

sum_Pm_Z_Scal <- sum(Pm_Z + Scal, index = 'j')
# Error in compile_formula(formula, var_aliases$var_aliases, dllname) : 
#     Error during cmake call.


formula <- "Sum_Reduction(Add(z,s),1)"
args <- c("z=Pm(6)", "s=Pm(1)")
sum_Pm_6_Pm_1 <- keops_kernel(formula, args)

sum_Pm_6_Pm_1(list(Pm_Z, Scal))
# Error in r_genred(input, param) : 
#     Not compatible with requested type: [type=list; target=double]. 


formula <- "Sum_Reduction(Add(z,s),1)"
args <- c("z=Pm(6)", "s=Pm(2)")
sum_Pm_6_Pm_2 <- keops_kernel(formula, args)
# Error in compile_formula(formula, var_aliases$var_aliases, dllname) : 
#     Error during cmake call.


formula <- "Sum_Reduction(v+s,1)"
args <- c("v=Pm(3)", "s=Pm(1)")
op22 <- keops_kernel(formula, args)

op22(list(V, 2))
#       [,1] [,2] [,3]
# [1,]    7    8    9

formula <- "Sum_Reduction(Add(v,s),1)"
args <- c("v=Pm(6)", "s=Pm(1)")
op23 <- keops_kernel(formula, args)

op23(list(Pm_Z$vars[[1]], 3))
#       [,1] [,2] [,3] [,4] [,5] [,6]
# [1,]    5    6    4    4    7   12


formula <- "Sum_Reduction(Add(v,Real2Complex(s)),1)"
args <- c("v=Pm(6)", "s=Pm(3)")
op23 <- keops_kernel(formula, args)

op23(list(Pm_Z$vars[[1]], V))
#       [,1] [,2] [,3] [,4] [,5] [,6]
# [1,]    5    6    4    4    7   12

# for issue

# works
formula <- "Sum_Reduction(Add(v,s),1)"
args <- c("v=Pm(6)", "s=Pm(1)")
op1 <- keops_kernel(formula, args)

# doesn't work
formula <- "Sum_Reduction(Add(v,Real2Complex(s)),1)"
args <- c("v=Pm(6)", "s=Pm(1)")
op2 <- keops_kernel(formula, args)
# Error in compile_formula(formula, var_aliases$var_aliases, dllname) : 
#     Error during cmake call. 

# works
formula <- "Sum_Reduction(Add(v,Real2Complex(s)),1)"
args <- c("v=Pm(6)", "s=Pm(3)")
op3 <- keops_kernel(formula, args)



# --------------
formula <- "Sum_Reduction(Add(z,s)),1)"
args <- c("z=Pm(6)", "s=Pm(1)")
op21 <- keops_kernel(formula, args)


O13 <- sum(Pm_Z + Pm_Z2, index = 'j')
#       [,1] [,2] [,3] [,4] [,5] [,6]
# [1,]    5    6    2    2    8   18


formula <- "Sum_Reduction(Powf(x, y),1)"
args <- c("x=Pm(3)", "y=Pm(3)")
op24 <- keops_kernel(formula, args)




O <- sum(Scal * Pm_Z, index = 'j') # TODO: FIX ME
O2 <- sum(Scal + Pm_Z, index = 'j') # TODO: FIX ME

R0 <- sum(Pm_Z + Pm_Z, index = 'j')
#       [,1] [,2] [,3] [,4] [,5] [,6]
# [1,]    4    6    2    2    8   18
R1 <- sum(Cplx + Cplx, index = 'j')
#       [,1] [,2]
# [1,]    4   18
R2 <- sum(Scal + Pm_V, index = 'j')
#       [,1] [,2] [,3]
# [1,]    7    8    9

O21 <- sum(Scal + Pm_Vect_z, index = 'j') # WORKS
O3 <- sum(Cplx * Vect_z, index = 'j') # TODO: FIX ME # works with "*"
O4 <- sum(Pm_Vect_z * Cplx, index = 'j') # TODO: FIX ME (error different from above)
O5 <- sum(z_i + Cplx, index = 'j') # TODO: FIX ME # works with "*" but not with "+"



P13 <- sum(x_i + y_i, index = 'i')
#       [,1] [,2] [,3]
# [1,]   25   25   25
P12 <- sum(x_i + x_i, index = 'i')
#       [,1] [,2] [,3]
# [1,]   20   20   20

P14 <- sum(x_i ^ x_i, index = 'i')

O6 <- sum(Scal + z_i, index = 'i') # TODO: FIX ME
# Error in compile_formula(formula, var_aliases$var_aliases, dllname) : 
#     Error during cmake call. 


O7 <- sum(Scal + z_i, index = 'j') # TODO: FIX ME
# ("Add(Real2Complex(A0x56491d949b40NA),A0x7f9dd8003c60i)")
# Error in op(x$vars) : 
#     The number of elements in the 'input' argument does not correspond to the number input arguments in the formula. 

O8 <- sum(z_i + Scal, index = 'j') # TODO: FIX ME
# ("Add(A0x7f9dd8003c60i,Real2Complex(A0x56491d949b40NA))")
# Error in op(x$vars) : 
#     The number of elements in the 'input' argument does not correspond to the number input arguments in the formula. 

P1 <- sum(x_i + Scal, index = 'j')
#       [,1] [,2] [,3]
# [1,]    3    3    3
# [2,]    4    4    4
# [3,]    5    5    5
# [4,]    6    6    6

O9 <- sum(z_i + z_i, index = 'j')
#       [,1] [,2] [,3] [,4] [,5] [,6]
# [1,]    2    0    2    0    2    0
# [2,]    4   -2    4   -2    4   -2
# [3,]    6    0    6    0    6    0
# [4,]    4    2    4    2    4    2 

O9_ij <- sum(z_i + z_j, index = 'i')
#Error in op(c(x$vars)) : 
#    The number of elements in the 'input' argument does not correspond to the number input arguments in the formula. 

formula <- "Sum_Reduction(Add(zi,zj),1)"
args <- c("zi=Vi(6)", "zj=Vj(6)")
op_z_i_z_j <- keops_kernel(formula, args)

op_z_i_z_j(list(z_i, z_j))
# Error in r_genred(input, param) : 
#     Not compatible with requested type: [type=list; target=double]. 

O10 <- sum(z_i + z2_i, index = 'j')
#           [,1]     [,2] [,3] [,4] [,5] [,6]
# [1,] 2.999936  0.00000 2.96  0.0  -22    0
# [2,] 4.000000 -1.00032 4.00 -1.2    4 -126
# [3,] 5.001600  0.00000 6.00  0.0  630    0
# [4,] 4.000000  1.00800 4.00  6.0    4 3126


P2 <- sum(Scal^Scal, index = 'i') # TODO: FIX ME
#       [,1]
# [1,]    4

P3 <- sum(square(x_i), index = 'i')
#       [,1] [,2] [,3]
# [1,]   30   30   30


Sum_yz <- y_i + z_i
O10 <- sum(y_i + z_i, index = 'i')

O11 <- sum(Sum_yz, index = 'i')

O5 <- sum()

L0 <- sum(Scal + x_i, index = 'j')
L1 <- sum(x_i + Scal, index = 'i')
L2 <- sum(x_i + Pm_V, index = 'i')
L3 <- sum(Pm_V + x_i, index = 'i')
L4 <- sum(Pm_V + Scal, index = 'i')
L5 <- sum(Scal + Pm_V, index = 'i')

M0 <- sum(Cplx * x_i, index = 'i')
M1 <- sum(x_i + Cplx, index = 'i')

N0 <- sum(Cplx + Cplx, index = 'i')
N1 <- sum(Cplx + Vect_z, index = 'i')
N1 <- sum(Pm_Vect_z + Pm_Vect_z, index = 'j')

formula = "Sum_Reduction(x + y,1)"
args = c("x=Vi(3)", "y=Vi(3)")
op21 <- keops_kernel(formula, args)

op21(list(x, y))

sum_reduction(x_i * y_i, index = 'i')


# Sum_Reduction ExtractT =======================================================

x <- matrix(c(1, 2, 3, 4, 5, 6), nrow = 3, ncol = 2)
#       [,1] [,2]
# [1,]    1    4
# [2,]    2    5
# [3,]    3    6

formula <- "Sum_Reduction(ExtractT(x, 1, 5),0)"
args <- c("x=Vi(2)")
op20 <- keops_kernel(formula, args)

op20(list(x))
#      [,1] [,2] [,3] [,4] [,5]
# [1,]    0    1    4    0    0
# [2,]    0    2    5    0    0
# [3,]    0    3    6    0    0

# A test with "Min_Reduction"
formula <- "Min_Reduction(ExtractT(x, 1, 5),1)"
args <- c("x=Vi(2)")
op21 <- keops_kernel(formula, args)

op21(list(x)) # consistent result
#       [,1] [,2] [,3] [,4] [,5]
# [1,]    0    1    4    0    0



# --------------------------

formula = "Sum_Reduction(ExtractT(x, 2, 4),0)"
args = c("x=Pm(1)")
op19 <- keops_kernel(formula, args)

op19(list(3.14))
#       [,1] [,2] [,3]
# [1,]    0 3.14    0

# --------------------------

formula = "Sum_Reduction(ExtractT(x, 1, 8),1)"
args = c("x=Pm(5)")
op19 <- keops_kernel(formula, args)

op19(list(c(1, 2, 3, 1, 5)))
#       [,1] [,2] [,3] [,4] [,5] [,6] [,7] [,8]
# [1,]    0    1    2    3    1    5    0    0

# ---------------------------

x <- matrix(c(1, 2, 3), nrow = 3, ncol = 2)
#       [,1] [,2]
# [1,]    1    1
# [2,]    2    2
# [3,]    3    3
x_i <- LazyTensor(x, index = 'i')

sum(extractT(x_i, 1, 8), index = 'i')
#       [,1] [,2] [,3] [,4] [,5] [,6] [,7] [,8]
# [1,]    0    6    6    0    0    0    0    0



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
# > aT
#       [,1] [,2] [,3] [,4] [,5] [,6] [,7]
# [1,]    0 3.14    0    0    0    0    0

x <- c(1, 2, 3, 4, 5)
scal <- 3.14
aformula <- "Sum_Reduction(x + ElemT(y, 5, 1), 1)"
args <- c("x=Vi(5)", "y=Pm(1)")
op1 <- keops_kernel(formula, args)
# > op1(list(x, scal))
#      [,1] [,2] [,3] [,4] [,5]
# [1,]    1 5.14    3    4    5

OH <- sum(one_hot(Pm(1.1), 3), index = 'i')
#       [,1] [,2] [,3]
# [1,]    0    1    0
OH_LT <- one_hot(Pm(1.1), 3)

PH <- sum(Pm(1) + Pm(1), index = 'i')

# Sum_Reduction Concat =========================================================

formula <- "Sum_Reduction(Concat(x, y), 1)"
args <- c("x=Vi(5)", "y=Vj(4)")
op1 <- keops_kernel(formula, args)

formula <- "Sum_Reduction(Concat(x, y), 0)"
args <- c("x=Vi(5)", "y=Vj(4)")
op2 <- keops_kernel(formula, args)

d1 <- 5
d2 <- 4
nx <- 1
ny <- 1

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

# mod ==========================================================================

# Sum_Reduction(Mod(Var(0,2,0), IntCst(1), IntCst(2)),1)

x <- matrix(c(1, 2, 3, 4, 5, 6), nrow = 3, ncol = 2)
#       [,1] [,2]
# [1,]    1    4
# [2,]    2    5
# [3,]    3    6

# works
formula <- "Sum_Reduction(Mod(x, IntCst(7), IntCst(2)),0)"
args <- c("x=Vi(2)")
op <- keops_kernel(formula, args)
op(list(x))

# works
formula <- "Sum_Reduction(Mod(x, y, IntCst(2)),0)"
args <- c("x=Vi(2)", "y=Vi(2)")
op <- keops_kernel(formula, args)
op(list(x, y))

# works
formula <- "Sum_Reduction(Mod(x, y, z),0)"
args <- c("x=Vi(2)", "y=Vi(2)", "z=Vi(2)")
op <- keops_kernel(formula, args)
op(list(x, y, z))
