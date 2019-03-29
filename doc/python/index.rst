
.. figure:: ../_static/logo/pykeops_logo.png
   :width: 100%
   :alt: KeopsLab logo

#########################
Python bindings for KeOps
#########################

We now fully document the public interface of the :mod:`pykeops` module, which is a NumPy or PyTorch front-end for the KeOps C++/Cuda library. It contains three sets of instructions:

1. :doc:`The Genred module <Genred>`: our **low-level** pythonic syntax, compatible with NumPy and PyTorch.

2. :doc:`The generic_reduction functions <generic-reduction>`: a **math-friendly** set of helper routines for the Genred module.

3. :doc:`kernel_product <kernel-product>`: a specific syntax with **convenient aliases** for operations in kernel spaces. It is only compatible with PyTorch.


.. toctree::
   :maxdepth: 2

   installation
   Genred
   generic-solver
   generic-reduction
   kernel-product
   sparsity


