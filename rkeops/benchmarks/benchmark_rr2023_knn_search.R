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

## k-nn search
knn_search_keops <- function(x, y, K) {
    x_i <- Vi(x)
    y_j <- Vj(y)
    
    res <- argKmin(sum(abs(x_i - y_j)), K, 'j')
    
    return(res)
}

knn_search_base <- function(x, y, K) {
    N <- nrow(x)
    D <- ncol(x)
    res <- matrix(0,N,K)
    for(i in 1:N) {
        res[i,] <- order(
            rowSums(
                abs(y - matrix(x[i,], N, D, byrow = TRUE))))[1:K]
    }
    return(res)
}



# check results
N <- 100; D <- 15
x <- matrix(rnorm(N*D), N, D)
y <- matrix(rnorm(N*D), N, D)
K <- 5

res1 <- knn_search_keops(x, y, K)
res2 <- knn_search_base(x, y, K)
sum(abs(res1+1 - res2))


bench_knn_search <- function(N, D, K, n_rep = 10) {
    
    # data
    x <- matrix(rnorm(N*D), N, D)
    y <- matrix(rnorm(N*D), N, D)
    
    # benchmark
    
    res <- benchmark(
        "keops_cpu_float32" = {
            rkeops_use_cpu()
            rkeops_use_float32()
            knn_search_keops(x, y, K)
        },
        "keops_cpu_float64" = {
            rkeops_use_cpu()
            rkeops_use_float64()
            knn_search_keops(x, y, K)
        },
        "keops_gpu_float32" = {
            rkeops_use_gpu()
            rkeops_use_float32()
            knn_search_keops(x, y, K)
        },
        "keops_gpu_float64" = {
            rkeops_use_gpu()
            rkeops_use_float64()
            knn_search_keops(x, y, K)
        },
        "base" = {
            knn_search_base(x, y, K)
        },
        replications = n_rep,
        columns = c("test", "replications", "elapsed")
    ) %>% as_tibble() %>%
        mutate(
            n_data = N
        )
    write.table(
        res, 
        file = paste0("res_benchmark_knn_search_N_", as.integer(N), ".csv"), 
        row.names = FALSE, col.names = TRUE)
    
    return(res)
}

N_val <- c(100, 500, 1000, 5000, 10000)

res_bench_knn_search <- Reduce("rbind", lapply(
    N_val, function(N) {
        print(paste0("#### N = ", N))
        bench_knn_search(N, D = 3, K = 10, n_rep = 100)
    }
))

str(res_bench_knn_search)
write.table(
    res_bench_knn_search, file = "res_benchmark_knn_search.csv", 
    row.names = FALSE, col.names = TRUE)
