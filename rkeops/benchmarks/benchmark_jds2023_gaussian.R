library(conflicted)
library(tidyr)
library(tibble)
library(dplyr)
library(microbenchmark)
library(reticulate)


virtualenv_create("rkeops")
virtualenv_install("rkeops", "pykeops")
use_virtualenv(virtualenv = "rkeops", required = TRUE)
py_config()


proj_dir <- rprojroot::find_root(".git/index")
pkg_dir <- file.path(proj_dir, "rkeops")
devtools::load_all(pkg_dir)


rkeops_use_float32()


# ---------------- gaussian convolution ---------------

gaussian_keops <- function(x, y, b, lambda) {
    x_i <- Vi(x)                    # init LazyTensor ("i" -> rows 1:N)
    y_j <- Vj(y)                    # init LazyTensor ("j" -> rows 1:N)
    b_j <- Vj(b)                    # init LazyTensor ("j" -> rows 1:N)
    Pm_lambda <- Pm(lambda)
    
    res <- sum(exp(-Pm_lambda*sqdist(x_i, y_j)) * b_j, "j")
    
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
            tmp[j, ] <- exp(-lambda * sum((x[i, ] - y[j, ])^2)) * b[j, ]
        }
        res[i, ] <- colSums(tmp)
    }
    return(res)
}


# ------------------- check results -------------------

N <- 6; D <- 3                  # dimensions
x <- matrix(rnorm(N*D), N, D)   # data: x of dim N x D
y <- matrix(rnorm(N*D), N, D)   # data: y of dim N x D
b <- matrix(rnorm(N*D), N, D)   # data: b of dim N x D
lambda <- 1.0                   # parameter: lambda

res1 <- gaussian_keops(x,y,b,lambda)
res2 <- gaussian_base(x,y,b,lambda)
sum(abs(res1 - res2))


# --------------------- benchmark ---------------------

bench_gaussian <- function(N, D, n_rep = 100L, max_base_N = 1000L) {
    
    # data
    x <- matrix(rnorm(N*D), N, D)   # data: x of dim N x D
    y <- matrix(rnorm(N*D), N, D)   # data: y of dim N x D
    b <- matrix(rnorm(N*D), N, D)   # data: b of dim N x D
    lambda <- 1.0
    
    # benchmark
    print("On CPU using float32...")
    res_cpu_32 <- microbenchmark(
        "keops_cpu_float32" = {
            rkeops_use_cpu()
            rkeops_use_float32()
            gaussian_keops(x, y, b, lambda)
        }, times = n_rep, unit = "s"
    )
    
    print("On CPU using float64...")
    res_cpu_64 <- microbenchmark(
        "keops_cpu_float64" = {
            rkeops_use_cpu()
            rkeops_use_float64()
            gaussian_keops(x, y, b, lambda)
        }, times = n_rep, unit = "s"
    )
    
    print("On GPU using float32...")
    res_gpu_32 <- microbenchmark(
        "keops_gpu_float32" = {
            rkeops_use_gpu()
            rkeops_use_float32()
            gaussian_keops(x, y, b, lambda)
        }, times = n_rep, unit = "s"
    )
    
    print("On GPU using float64...")
    res_gpu_64 <- microbenchmark(
        "keops_gpu_float64" = {
            rkeops_use_gpu()
            rkeops_use_float64()
            gaussian_keops(x, y, b, lambda)
        }, times = n_rep, unit = "s"
    )
    
    results <- list(res_cpu_32, res_cpu_64, res_gpu_32, res_gpu_64)

    if (N < max_base_N) {
        print("With base R...")
        res_base <- microbenchmark(
            "base_R" = {
                gaussian_base(x, y, b, lambda)
            }, times = n_rep, unit = "s"
	)

        results <- list(res_cpu_32, res_cpu_64, res_gpu_32, res_gpu_64, res_base)
    }
    
    print("Done.")

    results <- lapply(results, summary)
    
    res <- as_tibble(bind_rows(results)) %>%
        mutate(
            n_data = N
        )
   
    write.table(
        res, file = paste0("res_benchmark_gaussian_N_", as.integer(N), ".csv"), 
        row.names = FALSE, col.names = TRUE)
    
    return(res)
}


# ------------------ run benchmarks -------------------

N_val <- c(100, 1000, 5000, 10000, 50000, 100000, 200000)

res_bench_gaussian <- Reduce("rbind", lapply(
    N_val, function(N) {
        print(paste0("#### N = ", N))
        bench_gaussian(N, 15, n_rep = 100L, max_base_N = 10000L)
    }
))


# ------------------- save results --------------------

filename = "res_benchmark_gaussian.csv"
print(paste0("Done. Saving results in ", filename))
write.table(
    res_bench_gaussian, file = filename, 
    row.names = FALSE, col.names = TRUE)

