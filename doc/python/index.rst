
.. figure:: ../_static/logo/pykeops_logo.png
   :width: 100%
   :alt: KeopsLab logo

=========================
Python bindings for Keops
=========================

We now fully document the public interface of the pykeops module. In a nutshell, pykeops is a numpy or pytorch frontend to the KeOps C++/Cuda routine. It contains two sets of instruction:

1. :doc:`Generic reduction <generic-syntax>`: the generic syntax compatible with both numpy and pytorch. This syntax allows you to make everything that is possible with pyKeops.

2. :doc:`Kernel-product <kernel-product>`: a specific syntax with convenient aliases for formula corresponding to operations in kernel spaces. It is compatible with pytorch only.


.. toctree::
   :maxdepth: 2
   :caption: PyKeops

   installation
   generic-syntax
   kernel-product

