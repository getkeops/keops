==========
Benchmarks
==========

These benchmarks showcase the performances of the KeOps routines as the number of samples/points varies (typical use cases should be from 100 to 1,000,000).

Comparison with other related projects
--------------------------------------

KeOps is **fast**! You may find `here <https://github.com/getkeops/keops/tree/master/benchmarks>`_ a benchmark that compare the performances of pyKeOps with other related projects.

pyKeOps benchmarks
------------------

.. note::
    If you run a KeOps script for the first time, 
    the internal engine may take a **few minutes** to compile all the 
    relevant formulas.  This work is done **once** 
    as KeOps stores the resulting *shared object* files (`*.so`) in a 
    :ref:`cache directory <part.cache>`.

