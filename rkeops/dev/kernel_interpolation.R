set_rkeops_option("precision", "double")

# ----------------
# 1D INTERPOLATION
# ----------------
N <- 10000
x <- matrix(runif(N * 1), N, 1)
pert <- matrix(runif(N * 1), N, 1) # random perturbation to create b

b <- x + 0.5 * sin(6 * x) + 0.1 * sin(20 * x) + 0.05 * pert


gaussian_kernel <- function(x, y, sigma = 0.1) {
    x_i <- Vi(x)
    y_j <- Vj(y)
    D_ij <- sum((x_i - y_j)^2)
    res <- exp(-D_ij / (2 * sigma^2))
    return(res)
}


K_xx <- gaussian_kernel(x, x)


CG_solve <- function(K, b, lambda, eps = 1e-6) {
    # ----------------------------------------------------------------
    # Conjugate gradient algorithm to solve linear system of the form
    # (K + lambda * Id) * a = b.
    #
    # K: a LazyTensor encoding a symmetric positive definite matrix
    #       (the spectrum of the matrix must not contain zero)
    # b: a vector corresponding to the second member of the equation
    # lambda: Non-negative ridge regularization parameter
    #       (lambda = 0 means no regularization)
    # eps (default=1e-6): precision parameter
    # ----------------------------------------------------------------
    delta <- length(b) * eps^2
    a <- 0
    r <- b
    nr2 <- sum(r^2) # t(r)*r (L2-norm)
    if(nr2 < delta) {
        return(0 * r)
    }
    p <- r
    k <- 0
    while (TRUE) {
      #Mp <- K %*% p
      #Mp <- K %*% p + lambda * p
      Mp <- K %*% Vj(p) + lambda * p
      alp <- nr2 / sum(p * Mp)
      a <- a + (alp * p)
      r <- r - (alp * Mp)
      nr2new <- sum(r^2)
      if (nr2new < delta) {
          break
      }
      p <- r + (nr2new / nr2) * p
      nr2 <- nr2new
      k <- k + 1
    }
    return(a) # should be such that K%*%Vj(a) + lambda * Id * a = b (eps close) 
}

lambda <- 1

start <- Sys.time()
a <- CG_solve(K_xx, b, lambda = lambda)
end <- Sys.time()

time <- round(as.numeric(end - start), 5)
print(paste("Time to perform an RBF interpolation with",
            N, "samples in 1D:", time, "s.",
            sep = " "))

# check result
is_it_b <- (K_xx %*% Vj(a)) + lambda * a
is_it_b - b  # should (almost) be a zero vector


# ------- (gg)plot -------
t <- as.matrix(seq(from = 0, to = 1, length.out = N))

K_tx <- gaussian_kernel(t, x)
mean_t <- K_tx %*% Vj(a)

D <- as.data.frame(cbind(x, b, t, mean_t))
colnames(D) <- c("x", "b", "t", "mean_t")

ggplot(aes(x = x, y = b), data = D) +
  geom_point(color = '#1193a8', alpha = 0.5, size = 500 / length(x)) +
  geom_line(aes(x = t, y = mean_t), color = 'darkred') +
  annotate("text", x = .75, y = .1,
           label = paste("Number of samples: ", N, sep = "")
           ) +
  theme_bw()
# -------------------------


# --- plot (without ggplot) ---
t <- as.matrix(seq(from = 0, to = 1, length.out = N))

K_tx <- gaussian_kernel(t, x)
mean_t <- K_tx %*% a

plot(x[, 1], b[, 1],
     xlim = c(0, 1),
     ylim = c(0, 1),
     col = "blue")
lines(t, mean_t,
      type = "l",
      col = "red",
      lwd = 2.5)
# -----------------------------

# Compare results for different lambda values (not in the vignette)
# -----------------------------------------------------------------
# compare CGS with different lambda
eps <- 1e-6
lambda_vec <- seq(from = 0.7, to = 5., length.out = 35)

lambda_compare <- function(lambda_vec, K, b, eps = 1e-6) {
  sols <- c()
  errs <- c()
  for (lambda in lambda_vec) {
    a <- CG_solve(K, b, lambda = lambda, eps = eps)
    err <- ((K_xx %*% Vj(a)) + lambda * a) - b
    sols <- cbind(sols, a)
    errs <- cbind(errs, err)
  }
  colnames(sols) <- lambda_vec
  colnames(errs) <- lambda_vec
  return(list('solutions' = sols,
              'errors' = errs
  )
  )
}


res <- lambda_compare(lambda_vec = lambda_vec,
                      K = K_xx,
                      b = b
)

errors <- res$errors
mean_errors <- colMeans(errors)
D <- as.data.frame(mean_errors)
D$lambda <- row.names(D)

library(ggplot2)

D$lambda <- as.numeric(D$lambda)
D$mean_error <- as.numeric(D$mean_error)

ggplot(aes(x = lambda, y = mean_errors), data = D) +  
  geom_point(color = 'darkred', alpha = 0.5) +
  geom_path(group = 1, color = 'darkred') +
  scale_x_continuous(breaks = seq(from = 0.001, to = 5., by = 0.25)) +
  scale_y_continuous(breaks = as.numeric(format(seq(from = min(D$mean_error),
                                                    to = max(D$mean_errors),
                                                    by = 1e-7), digits = 3))) +
  geom_hline(yintercept = 0) +
  theme_bw() +
  theme(axis.text.x = element_text(angle = 45, vjust = -0.1))



# ----------------
# INTERPOLATION 2D
# ----------------
N <- 10000
# Sampling locations:
x <- matrix(runif(N * 2), N, 2)

# Some random-ish 2D signal:
b <- as.matrix(rowSums((x - 0.5)^2))
b[b > 0.4^2] = 0
b[b < 0.3^2] = 0
b[b >= 0.3^2] = 1

pert <- matrix(runif(N * 1), N, 1) # random perturbation to create b
b <- b + 0.05 * pert

# Add 25% of outliers:
Nout <- N %/% 4
b[(length(b) - Nout + 1):length(b)] <- matrix(runif(Nout * 1), Nout, 1)


laplacian_kernel <- function(x, y, sigma = 0.1) {
  x_i <- Vi(x)
  y_j <- Vj(y)
  D_ij <- sum((x_i - y_j)^2)
  res <- exp(-sqrt(D_ij) / sigma)
  return(res)
}


# check result with (Ridge) regularization
lambda <- 10

start <- Sys.time()
K_xx <- laplacian_kernel(x, x)
a <- CG_solve(K_xx, b, lambda = lambda)
end <- Sys.time()

time <- round(as.numeric(end - start), 5)
print(paste("Time to perform an RBF interpolation with",
            N, "samples in 2D:", time, "s.",
            sep = " "))

# is_it_b <- (K_xx %*% Vj(a)) + lambda * a
# is_it_b - b # should (almost) be a zero vector


# plot ----
library(ggplot2)
library(pracma) # to create meshgrid

# Extrapolate on a uniform sample:
X <- seq(from = 0, to = 1, length.out = 100)
Y <- seq(from = 0, to = 1, length.out = 100)

G <- meshgrid(X, Y)
X <- as.vector(G$X)
Y <- as.vector(G$Y)

t <- cbind(X, Y)

K_tx <- laplacian_kernel(t, x)
mean_t <- K_tx %*% Vj(a)

mean_t <- matrix(mean_t, 100, 100)
mean_t <- mean_t[nrow(mean_t):1, ]

# --- PLOT ---

# GGPLOT VERSION: TO IMPROVE
# --------------------------
# D <- as.data.frame(x)
# colnames(D) <- c("x1", "x2")
# 
# library(reshape)
# G2 <- as.data.frame(mean_t)
# G3 <- melt(G2)
# G3$toto <- seq(from = 0, to = 1, length.out = 100)
# 
# q <- ggplot(NULL) + 
#   geom_tile(data = G3, aes(variable, toto, fill= value)) +
#   scale_fill_gradient(low = "#C9C9C9", high = "darkred") # "#C2C2C9", "darkred"
# q
# 
# p <- q + geom_point(data = D,
#                     aes(x = 100*x1, y = x2),
#                     color = 1 + as.vector(b),
#                     alpha = 0.5)
# p


# PLOTLY VERSION
# --------------
library(plotly)
fig <- plot_ly(z = mean_t,
               type = "heatmap",
               colors = colorRamp(c("#C2C2C9", "darkred")),
               zsmooth ="best"
               )

fig <- fig %>% add_trace(type = "scatter",
                         x = ~(100 * x[, 1]),
                         y = ~(100 * x[, 2]),
                         mode = "markers",
                         marker = list(size = 4, color = as.vector(b))
                         )

fig <- fig %>% layout(xaxis = list(title = ""),
                      yaxis = list(title = ""))

colorbar(fig, limits = c(0, 1), x = 1, y = 0.75)
