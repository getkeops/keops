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

# ------------------------------

z <- matrix(1i^(-6:5), nrow = 4)
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


# Sum_Reduction x + y with different nrow ======================================

x <- matrix(c(1, 2, 3, 4), nrow = 4, ncol = 3)
y <- matrix(c(1, 2, 3), nrow = 3, ncol = 3)

formula = "Sum_Reduction(x + y,1)"
args = c("x=Vi(3)", "y=Vi(3)")
op15 <- keops_kernel(formula, args)

op15(list(x, y))
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

x <- matrix(c(1, 2, 3, 4), nrow = 4, ncol = 3)
#      [,1] [,2] [,3]
# [1,]    1    1    1
# [2,]    2    2    2
# [3,]    3    3    3
# [4,]    4    4    4
x_i <- LazyTensor(x, index = 'i')

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

z2 <- matrix(2 + 5i^(-6:5), nrow = 4)
#                   [,1]      [,2]      [,3]
# [1,] 1.999936+0.00000i 1.96+0.0i -23+   0i
# [2,] 2.000000-0.00032i 2.00-0.2i   2- 125i
# [3,] 2.001600+0.00000i 3.00+0.0i 627+   0i
# [4,] 2.000000+0.00800i 2.00+5.0i   2+3125i
z2_i <- LazyTensor(z2, index = 'i', is_complex = TRUE)


Sum_xz <- x_i + z_i
# > Sum_xz$formula
# [1] "Add(Real2Complex(A0x55ec34f7e7f8i),A0x7f5e90004440i)"

A <- sum(Sum_xz, index = 'i')
#      [,1] [,2] [,3] [,4] [,5] [,6]
# [1,]   18    0   18    0   18    0
A2 <- sum(z_i + x_i, index = 'i')
#      [,1] [,2] [,3] [,4] [,5] [,6]
# [1,]   18    0   18    0   18    0


B <- sum(Sum_xz, index = 'j') # No sum is done because nothing is indexed by 'j'
#       [,1] [,2] [,3] [,4] [,5] [,6]
# [1,]    2    0    2    0    2    0
# [2,]    4   -1    4   -1    4   -1
# [3,]    6    0    6    0    6    0
# [4,]    6    1    6    1    6    1

B2 <- sum(x_i + z_i, index = 'j')

C <- sum(real2complex(x_i), index = 'j')
# No sum is done because nothing is indexed by 'j', but this is
# just to see the result:

#       [,1] [,2] [,3] [,4] [,5] [,6]
# [1,]    1    0    1    0    1    0
# [2,]    2    0    2    0    2    0
# [3,]    3    0    3    0    3    0
# [4,]    4    0    4    0    4    0


# Try with vectors of complex instead of matrices --------

Z <- c(2 + 3i, 1 + 1i, 4 + 9i)
Pm_Z <- LazyTensor(Z)

Z2 <- c(3 + 3i, 1 + 1i, 4 + 9i)
Pm_Z2 <- LazyTensor(Z2)

V <- c(5, 6, 7)
Pm_V <- LazyTensor(V)

# Vector/vector addition ---
Sum_ZV <- Pm_Z + Pm_V

# Below, no sum is done because there are no index but this is just
# for verification purpose.
D <- sum(Sum_ZV, index = 'i')
#       [,1] [,2] [,3] [,4] [,5] [,6]
# [1,]    7    3    7    1   11    9

E <- sum(Sum_ZV, index = 'j')
#       [,1] [,2] [,3] [,4] [,5] [,6]
# [1,]    7    3    7    1   11    9

# Vector/matrix addition ---

H <- sum(z_i + Pm_Z, index = 'j')
# ("Add(A0x5575749b46b0i,A0x5575774e9328NA)")
# No sum is done because nothing is 'j' indexed:

#       [,1] [,2] [,3] [,4] [,5] [,6]
# [1,]    3    3    2    1    5    9
# [2,]    4    2    3    0    6    8
# [3,]    5    3    4    1    7    9
# [4,]    4    4    3    2    6   10

G <- sum(z_i + Pm_Z, index = 'i')
#       [,1] [,2] [,3] [,4] [,5] [,6]
# [1,]   16   12   12    4   24   36

# Vector/complex value addition ---
Cplx <- LazyTensor(2 + 9i)
Scal <- LazyTensor(2)


O1 <- sum(Pm_Z * Scal, index = 'j') # WORKS
#       [,1] [,2] [,3] [,4] [,5] [,6]
# [1,]    4    6    2    2    8   18

O12 <- sum(Pm_Z + Scal, index = 'j') # TODO: FIX ME
# Error in compile_formula(formula, var_aliases$var_aliases, dllname) : 
#     Error during cmake call.


formula <- "Sum_Reduction(Add(z,Real2Complex(s)),1)"
args <- c("z=Pm(6)", "s=Pm(1)")
op21 <- keops_kernel(formula, args)

#op21(list(x, s))

formula <- "Sum_Reduction(v+s,1)"
args <- c("v=Pm(3)", "s=Pm(1)")
op22 <- keops_kernel(formula, args)

op22(list(V, 2))
#       [,1] [,2] [,3]
# [1,]    7    8    9

formula <- "Sum_Reduction(Add(v,s),1)"
args <- c("v=Pm(6)", "s=Pm(2)")
op23 <- keops_kernel(formula, args)



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
O5 <- sum(Mat_z_i * Cplx, index = 'j') # TODO: FIX ME # works with "*"



P13 <- sum(x_i + y_i, index = 'i')
#       [,1] [,2] [,3]
# [1,]   25   25   25
P12 <- sum(x_i + x_i, index = 'i')
#       [,1] [,2] [,3]
# [1,]   20   20   20

P14 <- sum(x_i ^ x_i, index = 'i')

O6 <- sum(Scal * z_i, index = 'j') # TODO: FIX ME
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

formula = "Sum_Reduction(ExtractT(x, 1, 5),0)"
args = c("x=Vi(2)")
op20 <- keops_kernel(formula, args)

op20(list(x))
#      [,1] [,2] [,3] [,4] [,5]
# [1,]    0    1    4    0    0
# [2,]    0    2    5    0    0
# [3,]    0    3    6    0    0

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

aT <- sum(elemT(scal_Pm, 7, 1), index = 'i')
# > aT
#       [,1] [,2] [,3] [,4] [,5] [,6] [,7]
# [1,]    0 3.14    0    0    0    0    0

x = c(1,2,3,4,5)
scal = 3.14
aformula = "Sum_Reduction(x + ElemT(y, 5, 1), 1)"
args = c("x=Vi(5)", "y=Pm(1)")
op1 <- keops_kernel(formula, args)
# > op1(list(x, scal))
#      [,1] [,2] [,3] [,4] [,5]
# [1,]    1 5.14    3    4    5


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

