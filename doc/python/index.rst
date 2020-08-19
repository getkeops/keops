
.. figure:: ../_static/logo/pykeops_logo.png
   :width: 100%
   :alt: KeopsLab logo

#########################
Python bindings for KeOps
#########################

We now fully document the public interface of the :mod:`pykeops` module, which is a NumPy or PyTorch front-end for the KeOps C++/Cuda library. This package contains four sets of instructions:

1. :doc:`The LazyTensor wrapper <LazyTensor>`: our **high-level** interface, which provides a pythonic support for the most useful features of KeOps.

2. :doc:`The Genred module <Genred>`: our **low-level** syntax, compatible with NumPy and PyTorch.

3. :doc:`The generic_reduction functions <generic-reduction>`: a **math-friendly** set of helper routines for the Genred module.


.. toctree::
   :maxdepth: 2

   installation
   LazyTensor
   Genred
   generic-solver
   generic-reduction
   sparsity


