Changelog and Road map
=======================

Changelog
---------

Our `Changelog <https://github.com/getkeops/keops/blob/master/CHANGELOG.md>`_
can be found on the `KeOps Github repository <https://github.com/getkeops/keops/>`_.

To-do list
----------

* Add support for **tensor** (and not just vector) variables.
* Allow users to backprop through the ``.min()`` or ``.max()`` reduction
  of a :mod:`LazyTensor <pykeops.torch.LazyTensor>`.
* Add support for the **advanced indexing** of 
  :mod:`LazyTensors <pykeops.torch.LazyTensor>`. 
  Users should be able to **extract sub-matrices** as :mod:`LazyTensors <pykeops.torch.LazyTensor>` or genuine NumPy arrays / PyTorch tensors
  to perform e.g. Nyström approximation without having to
  implement twice the same kernel formula.
* Add support for the **block construction** of 
  :mod:`LazyTensors <pykeops.torch.LazyTensor>`,
  using a ``BlockLazyTensor([[A, B], [C, D]])`` syntax.
* Write new :meth:`.tensor()` and :meth:`.array()` methods
  for :mod:`LazyTensors <pykeops.torch.LazyTensor>`,
  allowing users to cast their symbolic operators as
  explicit matrices whenever possible.
* Add support for the `Fast and Furious Method <https://gargantua.polytechnique.fr/siatel-web/linkto/mICYYYT(myY6>`_ and other
  `Multipole <https://en.wikipedia.org/wiki/Fast_multipole_method>`_ 
  or `Nyström-like <https://en.wikipedia.org/wiki/Low-rank_matrix_approximations>`_ **approximations**.
  Long-term, we hope to provide a simple
  ``K.tol = 1e-3`` syntax for :mod:`LazyTensors <pykeops.torch.LazyTensor>`.
