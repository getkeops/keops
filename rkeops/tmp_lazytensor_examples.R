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

