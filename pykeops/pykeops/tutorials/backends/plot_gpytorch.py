"""
=================================
Linking KeOps with GPytorch
=================================

Out-of-the-box, KeOps only provides :ref:`limited support <interpolation-tutorials>` for
`Kriging <https://en.wikipedia.org/wiki/Kriging>`_
or `Gaussian process regression <https://scikit-learn.org/stable/modules/gaussian_process.html>`_:
the :class:`KernelSolve <pykeops.torch.KernelSolve>` operator
implements a conjugate gradient solver for kernel linear systems...
and that's about it.

Fortunately though, the GPytorch team has now integrated
`explicit KeOps kernels <https://docs.gpytorch.ai/en/latest/keops_kernels.html>`_ within
their repository: they are documented
`in this tutorial <https://docs.gpytorch.ai/en/latest/examples/02_Scalable_Exact_GPs/KeOps_GP_Regression.html>`_ .

"""
