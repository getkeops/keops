# req
library(conflicted)
library(tibble)
library(dplyr)
library(rbenchmark)
library(reticulate)

virtualenv_create("rkeops")
virtualenv_install("rkeops", "pykeops")
use_virtualenv(virtualenv = "rkeops", required = TRUE)
py_config()

proj_dir <- rprojroot::find_root(".git/index")
pkg_dir <- file.path(proj_dir, "rkeops")
devtools::load_all(pkg_dir)

rkeops_use_float32()

## gaussian
gaussian_keops <- function(x, y, b, lambda) {
    x_i <- Vi(x)                    # init LazyTensor ("i" -> rows 1:N)
    y_j <- Vj(y)                    # init LazyTensor ("j" -> rows 1:N)
    b_j <- Vj(b)                    # init LazyTensor ("j" -> rows 1:N)
    Pm_lambda <- Pm(lambda)
    
    res <- sum(exp(-Pm_lambda*sqdist(x_i, y_j)) * b_j, 'j')
    
    return(res)
}

gaussian_base <- function(x, y, b, lambda) {
    
    nx <- nrow(x)
    ny <- nrow(y)
    d <- ncol(x)
    
    res <- matrix(0, nx, d)
    for(i in 1:nx) {
        tmp <- matrix(0, ny, d)
        for(j in 1:ny) {
            tmp[j,] <- exp(-lambda * sum((x[i,] - y[j,])^2)) * b[j,]
        }
        res[i,] <- colSums(tmp)
    }
    return(res)
}


# check results
N <- 6; D <- 3              # dimensions
x <- matrix(rnorm(N*D), N, D)   # data: x of dim N x D
y <- matrix(rnorm(N*D), N, D)   # data: y of dim N x D
b <- matrix(rnorm(N*D), N, D)   # data: b of dim N x D
lambda <- 1.0                   # parameter: lambda

res1 <- gaussian_keops(x,y,b,lambda)
res2 <- gaussian_base(x,y,b,lambda)
sum(abs(res1 - res2))



bench_gaussian <- function(N, D, n_rep = 10) {
    
    # data
    x <- matrix(rnorm(N*D), N, D)   # data: x of dim N x D
    y <- matrix(rnorm(N*D), N, D)   # data: y of dim N x D
    b <- matrix(rnorm(N*D), N, D)   # data: b of dim N x D
    lambda <- 1.0
    
    # benchmark
    
    res <- benchmark(
        "keops_cpu_float32" = {
            rkeops_use_cpu()
            rkeops_use_float32()
            gaussian_keops(x, y, b, lambda)
        },
        "keops_cpu_float64" = {
            rkeops_use_cpu()
            rkeops_use_float64()
            gaussian_keops(x, y, b, lambda)
        },
        "keops_gpu_float32" = {
            rkeops_use_gpu()
            rkeops_use_float32()
            gaussian_keops(x, y, b, lambda)
        },
        "keops_gpu_float64" = {
            rkeops_use_gpu()
            rkeops_use_float64()
            gaussian_keops(x, y, b, lambda)
        },
        "base" = {
            gaussian_base(x, y, b, lambda)
        },
        replications = n_rep,
        columns = c("test", "replications", "elapsed")
    ) %>% as_tibble() %>%
        mutate(
            n_data = N
        )
    write.table(
        res, file = paste0("res_benchmark_gaussian_N_", as.integer(N), ".csv"), 
        \row.names = FALSE, col.names = TRUE)
    
    return(res)
}

N_val <- c(100, 1000, 5000, 10000, 50000)

res_bench_gaussian <- Reduce("rbind", lapply(
    N_val, function(N) {
        print(paste0("#### N = ", N))
        bench_gaussian(N, 15, n_rep = 100)
    }
))

str(res_bench_gaussian)
write.table(
    res_bench_gaussian, file = "res_benchmark_gaussian.csv", 
    row.names = FALSE, col.names = TRUE)
