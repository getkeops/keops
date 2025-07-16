# set working dir to `./rkeops` beforehand
# (see `./rkeops/dev_history.Rmd` for that)

# load
devtools::load_all()

# check rkeops setup
check_rkeops()

# Data
M <- 10000
N <- 15000
x <- matrix(runif(M * 3), nrow = M, ncol = 3) # arbitrary R matrix representing 
# 10000 data points in R^3
y <- matrix(runif(N * 3), nrow = N, ncol = 3) # arbitrary R matrix representing 
# 15000 data points in R^3
s <- 0.1                                      # scale parameter

# Turn our Tensors into KeOps symbolic variables:
x_i <- LazyTensor(x, "i")     # symbolic object representing an arbitrary row
                              # of x, indexed by the letter "i"
y_j <- LazyTensor(y, "j")     # symbolic object representing an arbitrary row
                              # of y, indexed by the letter "j"

# Perform large-scale computations, without memory overflows:
D_ij <- sum((x_i - y_j)^2)    # symbolic matrix of pairwise squared distances, 
                              # with 10000 rows and 15000 columns

K_ij <- exp(- D_ij / s^2)     # symbolic matrix, 10000 rows and 15000 columns

# D_ij and K_ij are only symbolic at that point, no computation is done

# Computing the result without storing D_ij and K_ij:
a_j <- sum(K_ij, index = "i") # actual R matrix (in fact a row vector of 
                              # length 15000 here)
                              # containing the column sums of K_ij
                              # (i.e. the sums over the "i" index, for each 
                              # "j" index)

# clean
clean_rkeops(remove_cache_dir = TRUE)
