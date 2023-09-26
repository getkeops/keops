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


# --------------------- knn search --------------------

knn_search_keops <- function(x, y, K) {
    x_i <- Vi(x)
    y_j <- Vj(y)
    
    res <- argKmin(sum(abs(x_i - y_j)), K, 'j')
    
    return(res)
}

knn_search_base <- function(x, y, K) {
    N <- nrow(x)
    D <- ncol(x)
    res <- matrix(0, N, K)
    for(i in 1:N) {
        res[i, ] <- order(
            rowSums(
                abs(y - matrix(x[i, ], N, D, byrow = TRUE))))[1:K]
    }
    return(res)
}


# ------------------- check results -------------------

N <- 100; D <- 15
x <- matrix(rnorm(N*D), N, D)
y <- matrix(rnorm(N*D), N, D)
K <- 5

res1 <- knn_search_keops(x, y, K)
res2 <- knn_search_base(x, y, K)
sum(abs(res1+1 - res2))


# --------------------- benchmark ---------------------

bench_knn_search <- function(N, D, K, n_rep = 100L, max_base_N = 1000L) {
    
    # data
    x <- matrix(rnorm(N*D), N, D)
    y <- matrix(rnorm(N*D), N, D)
    
    # benchmark
    print("On CPU using float32...")
    res_cpu_32 <- microbenchmark(
        "keops_cpu_float32" = {
            rkeops_use_cpu()
            rkeops_use_float32()
            knn_search_keops(x, y, K)
        }, times = n_rep, unit = "s"
    )
    
    print("On CPU using float64...")
    res_cpu_64 <- microbenchmark(
        "keops_cpu_float64" = {
            rkeops_use_cpu()
            rkeops_use_float64()
            knn_search_keops(x, y, K)
        }, times = n_rep, unit = "s"
    )
    
    print("On GPU using float32...")
    res_gpu_32 <- microbenchmark(
        "keops_gpu_float32" = {
            rkeops_use_gpu()
            rkeops_use_float32()
            knn_search_keops(x, y, K)
        }, times = n_rep, unit = "s"
    )
    
    print("On GPU using float64...")
    res_gpu_64 <- microbenchmark(
        "keops_gpu_float64" = {
            rkeops_use_gpu()
            rkeops_use_float64()
            knn_search_keops(x, y, K)
        }, times = n_rep, unit = "s"
    )
    
    results <- list(res_cpu_32, res_cpu_64, res_gpu_32, res_gpu_64)

    if (N <= max_base_N) {
	print("With base R...")
        res_base <- microbenchmark(
            "base_R" = {
                knn_search_base(x, y, K)
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
        res, file = paste0("res_benchmark_knn_search_N_", as.integer(N), ".csv"), 
        row.names = FALSE, col.names = TRUE)
    
    return(res)
}


# ------------------ run benchmarks -------------------

N_val <- c(100, 1000, 5000, 10000, 50000, 100000, 500000, 1000000)

res_bench_knn_search <- Reduce("rbind", lapply(
    N_val, function(N) {
        print(paste0("#### N = ", N))
        bench_knn_search(N, D = 3, K = 10, n_rep = 100L, max_base_N = 50000L)
    }
))


# ------------------- save results --------------------

filename = "res_benchmark_knn_search.csv"
print(paste0("Done. Saving results in ", filename))
write.table(
    res_bench_knn_search, file = filename, 
    row.names = FALSE, col.names = TRUE)
