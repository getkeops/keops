Kriging - Gaussian Process regression
-----------------------------------------

Thanks to a simple **conjugate gradient** solver, 
the :func:`pykeops.numpy.KernelSolve` and :func:`pykeops.pytorch.KernelSolve`
operators can be used to solve large-scale interpolation problems
with a **linear memory footprint**.