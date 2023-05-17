# only for dev purpose
library(devtools)
devtools::load_all('.')

# ----------------------- R only ----------------------
M <- 1000
N <- 2000
D <- 3

# Generate data
gamma <- .5
x <- matrix(rnorm(M*D), M, D)
y <- matrix(rnorm(N*D), N, D)
b <- matrix(rnorm(N*2), N, 2)

# Define a custom formula
formula <- 'Sum_Reduction(Exp(-G * SqDist(X,Y)) * B, 1)'
variables <- c('G = Pm(1)',
               'X = Vi(3)',
               'Y = Vj(3)',
               'B = Vj(2)'
               )

# Compile operator
gaussian_conv <- keops_kernel(formula, variables)
# Compute reduction
res_rk <- gaussian_conv(list(gamma, x, y, b))


# ----------------- Same with PyKeOps -----------------
library(reticulate)
source_python('dev/eg_GaussianRBF.py')

res_pknp <- pykeops_GaussianRBF(gamma, x, y, b)


# ------------------ Compare results ------------------
res_diff <- sum(abs(res_rk - res_pknp))
print(paste0('Sum of absolute differences between RKeOps and PyKeOps results: ', res_diff))


# ------------ Compute grdaient with RKeOps -----------
formula_grad <- 'Grad(Sum_Reduction(Exp(-G * SqDist(X,Y)) * B, 1), Y, e)'
variables_grad <- c(variables, 'e = Vi(2)')
gaussian_conv_grad <- keops_kernel(formula_grad, variables_grad)

e <- matrix(rnorm(M*D), M, D)
res_grad_rk <- gaussian_conv_grad(list(gamma, x, y, b, e))


# ------------------ And with PyKeOps -----------------
res_grad_pknp <- pykeops_GaussianRBF_grad(gamma, x, y, b, e)


# ------------------ Compare results ------------------
res_grad_diff <- sum(abs(res_grad_rk - res_grad_pknp))
print(paste0('Sum of absolute differences between RKeOps and PyKeOps results: ', res_grad_diff))


# ---------- Plot results next to each other ----------
#library(ggplot2)
#library(dplyr)
#library(tidyr)
#
#trunc <- 30
#df <- bind_rows('RKeOps' = as_tibble(res_rk[0:trunc, ]), 'PyKeOps' = as_tibble(res_pknp[0:trunc, ]),
#                .id = 'Method') %>%
#      mutate(index = rep(1:trunc, n_distinct(Method))) %>%
#      gather(key = 'dim', value = 'value', -c(Method, index))
#
#p <- ggplot(df, aes(x = index, y = value, group = Method, col = Method, linetype = Method)) +
#        geom_line() + theme_bw() +
#        facet_wrap(~ dim, ncol = 1, switch = 'y') +
#        theme(strip.background = element_blank(), strip.text.y = element_blank())
#
#print(p)
