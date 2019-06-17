Road map
========

Our `Changelog <https://github.com/getkeops/keops/blob/master/CHANGELOG.md>`_
can be found on the `KeOps Github repository <https://github.com/getkeops/keops/>`_.

To-do list
-------------

* Put **reference paper** on Arxiv.
* Fully document the **inner C++ API** and algorithms.
* Provide **R bindings**.
* Add support for the `Fast and Furious Method <https://gargantua.polytechnique.fr/siatel-web/linkto/mICYYYT(myY6>`_ and other
  `Multipole <https://en.wikipedia.org/wiki/Fast_multipole_method>`_ 
  or `Nyström-like <https://en.wikipedia.org/wiki/Low-rank_matrix_approximations>`_ **approximations**.
  By the start of 2020, we hope to provide a simple
  ``K.tol = 1e-3`` syntax for :mod:`LazyTensors <pykeops.common.lazy_tensor.LazyTensor>`.
