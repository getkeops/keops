#"""
#Advanced syntax in formulas
#===========================
#
#Let's write generic formulas using the KeOps syntax.
#
# 
#"""

####################################################################
# Setup
# ------------------
# First, the standard imports:
library('reticulate')


torch <- import('torch')
plt <- import('matplotlib.pyplot')

kt <- import('pykeops.torch')

# Choose the storage place for our data : CPU (host) or GPU (device) memory.
device <- torch$device(if (torch$cuda$is_available()) "cuda" else "cpu")

####################################################################
# Then, the definition of our dataset:
#
# - :math:`p`,   a vector of size 2.
# - :math:`x = (x_i)`, an N-by-D array.
# - :math:`y = (y_j)`, an M-by-D array.

N <- 1000L
M <- 2000L
D <- 3L

torch$set_default_dtype(torch$float64)

# PyTorch tip: do not 'require_grad' of 'x' if you do not intend to
#              actually compute a gradient wrt. said variable 'x'.
#              Given this info, PyTorch (+ KeOps) is smart enough to
#              skip the computation of unneeded gradients.

# Generate data with R
p <- rnorm(2L)
x <- matrix(rnorm(N*D), N, D)
y <- matrix(rnorm(M*D), M, D)

# R arrays to Torch tensors
p <- torch$tensor(p, requires_grad=TRUE, device=device)$contiguous()
x <- torch$tensor(x, requires_grad=FALSE, device=device)$contiguous()
y <- torch$tensor(y, requires_grad=TRUE, device=device)$contiguous()

# + some random gradient to backprop:
g <- matrix(rnorm(N*D), N, D)
g <- torch$tensor(g, requires_grad=TRUE, device=device)$contiguous()


####################################################################
# Computing an arbitrary formula
# -------------------------------------
#
# Thanks to the `Elem` operator,
# we can now compute :math:`(a_i)`, an N-by-D array given by:
#
# .. math::
#
#   a_i = \sum_{j=1}^M (\langle x_i,y_j \rangle^2) (p_0 x_i + p_1 y_j)
#
# where the two real parameters are stored in a 2-vector :math:`p=(p_0,p_1)`.

# Keops implementation.
# Note that Square(...) is more efficient than Pow(...,2)
formula <- "Square((X|Y)) * ((Elem(P, 0) * X) + (Elem(P, 1) * Y))"
variables <- c("P = Pm(2)",  # 1st argument,  a parameter, dim 2.
               "X = Vi(3)",  # 2nd argument, indexed by i, dim D.
               "Y = Vj(3)"  # 3rd argument, indexed by j, dim D.
               )

my_routine <- kt$Genred(formula, variables, reduction_op="Sum", axis=1L)
a_keops <- my_routine(p, x, y)

# Vanilla PyTorch implementation
scals <- torch$pow(torch$mm(x, torch$t(y)), 2L)  # Memory-intensive computation!

a_pytorch <- torch$add(torch$multiply(torch$multiply(p[0], scals$sum(1L)$view(-1L, 1L)), x),
                       torch$multiply(p[1], torch$mm(scals, y)))


# Plot the results next to each other:
for (i in 0L:(D - 1L)){
    plt$subplot(D, 1L, i + 1L)
    plt$plot(a_keops$detach()$cpu()$numpy()[1:40, i + 1], '-', label='KeOps')
    plt$plot(a_pytorch$detach()$cpu()$numpy()[1:40, i + 1], '--', label='PyTorch')
    plt$legend(loc='lower right')
}
plt$tight_layout()
plt$show()
