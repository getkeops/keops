Python install
##############

PyKeOps is a **Python 3 wrapper** around the low-level KeOps library, which is written in **C++/CUDA**. 
It provides functions that can be used in any **NumPy** or **PyTorch** script.

Requirements
============

- **Python 3** with packages **numpy**.
- A C++ compiler compatible with ``std=c++11``: **g++** version >=7 or **clang++** version >=8.
- The **Cuda** toolkit: version >=10.0 is recommended.
- **PyTorch** (optional): version >= 1.5.


Using pip (recommended)
=======================

1. Just in case: in a terminal, check the **consistency** of the outputs of the commands ``which python``, ``python --version``, ``which pip`` and ``pip --version``. 

2. In a terminal, type:

  .. code-block:: bash

    pip install pykeops

  Note that compiled shared objects (``.so`` files on Unix, ``.dylib`` on macOS) will be stored in the folder  ``~/.cache/keops/``, where ``~`` is the path to your home folder.

3. Test your installation, as described in the :ref:`next section <part.checkPython>`.

On Google Colab
===============

Google provides free virtual machines where KeOps runs
out-of-the-box. 
In a new `Colab notebook <https://colab.research.google.com>`_, typing:

.. code-block:: bash

    !pip install pykeops > install.log

should allow you to get a working version of KeOps in less than twenty seconds.


From source using git
=====================


The simplest way of installing a specific version
of KeOps is to use `some advanced pip syntax <https://pip.pypa.io/en/stable/reference/pip_install/#git>`_:


.. code-block:: bash

    pip install git+https://github.com/getkeops/keops.git@main#subdirectory=keopscore
    pip install git+https://github.com/getkeops/keops.git@main#subdirectory=pykeops


Alternatively, you may:

1. Clone the KeOps repo at a location of your choice (denoted here as ``/path/to``):

  .. code-block:: bash

    git clone --recursive https://github.com/getkeops/keops.git /path/to/libkeops

  Note that compiled **.so** routines will be stored in the folder ``/path/to/libkeops/pykeops/build``: this directory must have **write permission**. 


2. Install via pip in editable mode as follows :
           
    .. code-block:: bash

      pip install -e /path/to/libkeops/keopscore -e /path/to/libkeops/pykeops

  + Otherwise you may add the directories ``/path/to/libkeops/keopscore`` and ``/path/to/libkeops/pykeops`` to your python path. This can be done once and for all, by adding the path to to your ``~/.bashrc``. In a terminal, type:
        
    .. code-block:: bash

      echo "export PYTHONPATH=$PYTHONPATH:/path/to/libkeops/keopscore:/path/to/libkeops/pykeops" >> ~/.bashrc

  + Alternatively, you may add the following line to the beginning of your python scripts:
    
    .. code-block:: python

      import os.path
      import sys
      sys.path.append('/path/to/libkeops/keopscore')
			sys.path.append('/path/to/libkeops/pykeops')

3. Test your installation, as described in the :ref:`next section. <part.checkPython>`


.. _`part.checkPython`:

Testing your installation
=========================

You can use the following test functions to compile and run simple KeOps formulas. If the compilation fails, it returns the full log.

1.  In a python terminal, 

  .. code-block:: python

    import pykeops
    pykeops.test_numpy_bindings()    # perform the compilation
        
  should return:

  .. code-block:: bash

    pyKeOps with numpy bindings is working!

2. If you use PyTorch, the following code:

  .. code-block:: python

    import pykeops
    pykeops.test_torch_bindings()    # perform the compilation
  
  should return:

  .. code-block:: bash

    pyKeOps with torch bindings is working!


Troubleshooting
===============

Compilation issues
------------------

First of all, make sure that you are using a C++ compiler which is compatible with the **C++11 revision**. Otherwise, compilation of formulas may fail in unexpected ways. Depending on your system, you can:

1. Install a compiler **system-wide**: for instance, on Debian-based Linux distributions, you can install g++ with apt and then use `update-alternatives <https://askubuntu.com/questions/26498/choose-gcc-and-g-version>`_ to choose a suitable compiler as default. Don't forget to pick compatible versions for both **gcc** and **g++**.  

2. Install a compiler **locally**: if you are using a conda environment, you can install a new instance of gcc and g++ by following the `documentation of conda <https://conda.io/docs/user-guide/tasks/build-packages/compiler-tools.html>`_.


.. _`part.cache`:

Cache directory
---------------

If you experience problems with compilation, it may be a good idea to **flush the build folder** that KeOps uses as a cache for already-compiled formulas. To do this, just type:

.. code-block:: python

  import pykeops
  pykeops.clean_pykeops()

You can change the build folder by using the ``set_build_folder()`` function:

.. code-block:: python

  import pykeops
  print(pykeops.get_build_folder())  # display current build_folder
  pykeops.set_build_folder("/my/new/location")  # change the build folder
  print(pykeops.get_build_folder())  # display new build_folder

Note that the command ``set_build_folder()`` without any argument will reset the location to the default one (``~/.keops/build`` on unix-like systems)

Verbosity level
---------------

You can deactivate all messages and warnings by setting the environment variable `PYKEOPS_VERBOSE` to 0. In a terminal, type:

.. code-block:: bash

  export PYKEOPS_VERBOSE=0
  python my_script_calling_pykeops.py

Alternatively, you can disable verbose compilation from your python script using the function ``pykeops.set_verbose``. In a python shell, type:

.. code-block:: python

  import pykeops
  pykeops.set_verbose(False)

