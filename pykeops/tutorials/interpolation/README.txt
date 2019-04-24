Interpolation - Splines
-----------------------

Thanks to a simple **conjugate gradient** solver, 
the :class:`numpy.KernelSolve <pykeops.numpy.operations.KernelSolve>` and :class:`torch.KernelSolve <pykeops.torch.operations.KernelSolve>`
operators can be used to solve large-scale interpolation problems
with a **linear memory footprint**.