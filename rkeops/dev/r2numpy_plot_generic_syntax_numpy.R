#"""
#Sum reduction 
#=====================
#"""

####################################################################
# Let's compute the (3000,3) tensor :math:`c` whose entries
# :math:`c_i^u` are given by:
#
# .. math::
#   c_i^u = \sum_j (p-a_j)^2 \exp(x_i^u+y_j^u)
#
# where
#
# * :math:`x` is a (3000,3) tensor, with entries :math:`x_i^u`.
# * :math:`y` is a (5000,3) tensor, with entries :math:`y_j^u`.
# * :math:`a` is a (5000,1) tensor, with entries :math:`a_j`.
# * :math:`p` is a scalar, encoded as a vector of size (1,).
#

####################################################################
# Setup
# -----
#
# Standard imports:

library('reticulate')


np <- import('numpy')
plt <- import('matplotlib.pyplot')

knp <- import('pykeops.numpy')

#####################################################################
# Declare random inputs:
M <- 30L
N <- 50L

dtype <- 'float32'

# Generate data with R
x <- matrix(runif(M*3), M, 3L)
y <- matrix(runif(N*3), N, 3L)
a <- matrix(runif(N*1), N, 1L)
p <- matrix(runif(1L))

# R arrays to NumPy arrays
# With reticulate, need to use np_array() to specify dtype
x <- np_array(x, dtype=dtype)
y <- np_array(y, dtype=dtype)
a <- np_array(a, dtype=dtype)
p <- np_array(p, dtype=dtype)

####################################################################
# Define a custom formula
# -----------------------

formula <- 'Square(p-a)*Exp(x+y)'
variables <- c('x = Vi(3)',
               'y = Vj(3)',
               'a = Vj(1)',
               'p = Pm(1)'
               )


####################################################################
# Our sum reduction is performed over the index :math:`j`,
# i.e. on the axis ``1`` of the kernel matrix.
# The output c is an :math:`x`-variable indexed by :math:`i`.

my_routine <- knp$Genred(formula, variables, reduction_op="Sum", axis=1L)
c <- my_routine(x, y, a, p, backend="auto")


####################################################################
# The equivalent code in NumPy:

#c_np = (
#    (
#        (p - a.T)[:, np.newaxis] ** 2
#        * np.exp(x.T[:, :, np.newaxis] + y.T[:, np.newaxis, :])
#    )
#    .sum(2)
#    .T
#)

# Above NumPy equivalent with reticulate:
first_term <- np$expand_dims(np$subtract(p, np$transpose(a)), axis=0L)
first_term <- np$power(first_term, 2L)

x_expand <- np$expand_dims(np$transpose(x), axis=2L)
y_expand <- np$expand_dims(np$transpose(y), axis=1L)
second_term <- np$exp(np$add(x_expand, y_expand))

c_np <- np$transpose(np$sum(np$multiply(first_term, second_term), axis=2L))

for (i in 0L:2L){
    plt$subplot(3L, 1L, i + 1L)
    # TODO: should we use python slicing here instead of R's?
    plt$plot(c[1:30, i + 1], '-', label='KeOps')
    plt$plot(c_np[1:30, i + 1], '--', label='NumPy')
    plt$legend(loc='lower right')
}
plt$tight_layout()
plt$show()


####################################################################
# Compute the gradient
# --------------------
# Now, let's compute the gradient of :math:`c` with
# respect to :math:`y`. Since :math:`c` is not scalar valued,
# its "gradient" :math:`\partial c` should be understood as the adjoint of the
# differential operator, i.e. as the linear operator that:
#
# - takes as input a new tensor :math:`e` with the shape of :math:`c`
# - outputs a tensor :math:`g` with the shape of :math:`y`
#
# such that for all variation :math:`\delta y` of :math:`y` we have:
#
# .. math::
#
#    \langle \text{d} c . \delta y , e \rangle  =  \langle g , \delta y \rangle  =  \langle \delta y , \partial c . e \rangle
#
# Backpropagation is all about computing the tensor :math:`g=\partial c . e` efficiently, for arbitrary values of :math:`e`:


# Declare a new tensor of shape (M,3) used as the input of the gradient operator.
# It can be understood as a "gradient with respect to the output c"
# and is thus called "grad_output" in the documentation of PyTorch.
e <- np_array(np$random$randn(M, 3L), dtype=dtype)

####################################################################
# KeOps provides an autodiff engine for formulas. Unfortunately though, as NumPy does not provide any support for backpropagation, we need to specify some informations by hand and add the gradient operator around the formula: ``Grad(formula , variable_to_differentiate, input_of_the_gradient)``
formula_grad <- paste0('Grad(', formula, ', y, e)')

# This new formula makes use of a new variable (the input tensor e)
variables_grad <- c(variables, 'e = Vi(3)')  # Fifth arg: an i-variable of size 3... Just like "c"!

# The summation is done with respect to the 'i' index (axis=0) in order to get a 'j'-variable
my_grad <- knp$Genred(formula_grad, variables_grad, reduction_op="Sum", axis=0L)

g <- my_grad(x, y, a, p, e)

####################################################################
# To generate an equivalent code in numpy, we must compute explicitly the adjoint
# of the differential (a.k.a. the derivative).
# To do so, let see :math:`c^i_u` as a function of :math:`y_j`:
#
# .. math::
#
#   g_j^u = [(\partial_{y} c^u(y)) . e^u]_j = \sum_{i} (p-a_j)^2 \exp(x_i^u+y_j^u) \cdot e_i^u
#
# and implement the formula:
#
#g_np = (
#    (
#        (p - a.T)[:, np.newaxis, :] ** 2
#        * np.exp(x.T[:, :, np.newaxis] + y.T[:, np.newaxis, :])
#        * e.T[:, :, np.newaxis]
#    )
#    .sum(1)
#    .T
#)

# Above NumPy equivalent with reticulate:
first_term <- np$expand_dims(np$subtract(p, np$transpose(a)), axis=0L)
first_term <- np$power(first_term, 2L)

x_expand <- np$expand_dims(np$transpose(x), axis=2L)
y_expand <- np$expand_dims(np$transpose(y), axis=1L)
second_term <- np$exp(np$add(x_expand, y_expand))

third_term <- np$expand_dims(np$transpose(e), axis=2L)

g_np <- np$transpose(
            np$sum(
                np$multiply(first_term, np$multiply(second_term, third_term)),
                axis=1L
            )
        )

# Plot the results next to each other:
for (i in 0L:2L){
    plt$subplot(3L, 1L, i + 1L)
    # TODO: should we use python slicing here instead of R's?
    plt$plot(g[1:30, i + 1], '-', label='KeOps')
    plt$plot(g_np[1:30, i + 1], '--', label='NumPy')
    plt$legend(loc='lower right')
}
plt$tight_layout()
plt$show()
