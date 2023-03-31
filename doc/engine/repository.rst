Structure of the repository
================================

`KeOps repo <https://github.com/getkeops/keops>`_ structure may be summarized as follows:

-  The `pykeops/ <(https://github.com/getkeops/keops/tree/main/pykeops>`_ folder, 
   with `common/ <https://github.com/getkeops/keops/tree/main/pykeops/common>`_, 
   `numpy/ <https://github.com/getkeops/keops/tree/main/pykeops/numpy>`_ and 
   `torch/ <https://github.com/getkeops/keops/tree/main/pykeops/torch>`_
   subfolders contains our Python wrappers and relies on the
   fantastic `PyBind11 <https://pybind11.readthedocs.io/en/stable/>`_ library.

-  The `rkeops/ <https://github.com/getkeops/keops/tree/main/rkeops>`_
   folder contains the R package sources deployed on CRAN.

-  The `keops/ <https://github.com/getkeops/keops/tree/main/keops>`_
   folder contains the meta-programming engine written in Python. It may
   be loaded as a Python module. There is no dependencies except built-in
   functions.
