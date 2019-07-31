Python install
##############

PyKeOps is a **Python 3 wrapper** around the low-level KeOps library, which is written in **C++/CUDA**. 
It provides functions that can be used in any **NumPy** or **PyTorch** script.

Requirements
============

- **Python 3** with packages **numpy** and **GPUtil**.
- A C++ compiler: **g++** version >=5 or **clang++**.
- The **Cmake** build system, version >= 3.10.
- The **Cuda** toolkit, including the **nvcc** compiler (optional): version >=9.0 is recommended. Make sure that your C++ compiler is compatible with the installed nvcc.
- **PyTorch** (optional): version >= 1.0.0.


Using pip (recommended)
=======================

1. Just in case: in a terminal, check the **consistency** of the outputs of the commands ``which python``, ``python --version``, ``which pip`` and ``pip --version``. 

2. In a terminal

  .. code-block:: bash

    pip install pykeops

  Note that the compiled shared objects (``*.so`` files) will be stored into the folder  ``~/.cache/libkeops-$version`` where ``~`` is the path to your home folder and ``$version`` is the package version number.

3. Test your installation, as described in the :ref:`next section. <part.checkPython>`

On Google Colab
===============

Google provides free virtual machines which are able to run KeOps
out-of-the-box. 
In a new `Colab notebook <https://colab.research.google.com>`_, typing:

.. code-block:: bash

    !pip install pykeops[full] > install.log

should allow you to get a working version of KeOps in less than twenty seconds.


From source using git
=====================

1. Clone the KeOps repo at a location of your choice (denoted here as ``/path/to``)

  .. code-block:: console

    git clone --recursive https://github.com/getkeops/keops.git /path/to/libkeops

  Note that your compiled **.so** routines will be stored in the folder ``/path/to/libkeops/pykeops/build``: this directory must have **write permission**. 


2. Manually add the directory ``/path/to/libkeops`` (and **not** ``/path/to/libkeops/pykeops/``) to your python path.
   
  + This can be done once and for all, by adding the path to to your ``~/.bashrc``. In a terminal,
        
    .. code-block:: bash

      echo "export PYTHONPATH=$PYTHONPATH:/path/to/libkeops/" >> ~/.bashrc

  + Otherwise, you may add the following line to the beginning of your python scripts:
    
    .. code-block:: python

      import os.path
      import sys
      sys.path.append('/path/to/libkeops')

3. Test your installation, as described in the :ref:`next section. <part.checkPython>`


.. _`part.checkPython`:

Testing your installation
=========================

1. In a python terminal,

  .. code-block:: python

    import numpy as np
    import pykeops.numpy as pknp
    
    x = np.arange(1, 10).reshape(-1, 3).astype('float32')
    y = np.arange(3, 9).reshape(-1, 3).astype('float32')
    
    my_conv = pknp.Genred('SqNorm2(x - y)', ['x = Vi(3)', 'y = Vj(3)'])
    print(my_conv(x, y))
        
  should return:

  .. code-block:: console

    Compiling libKeOpsnumpy5ac3d464a2 in /path/to/build_dir/:
       formula: Sum_Reduction(SqNorm2(x - y),1)
       aliases: x = Vi(0,3); y = Vj(1,3); 
       dtype  : float64

  .. code-block:: python

    [[63.]
     [90.]]



2. If you use PyTorch, the following code:

  .. code-block:: python

    import torch
    import pykeops.torch as pktorch
    
    x = torch.arange(1, 10, dtype=torch.float32).view(-1, 3)
    y = torch.arange(3, 9, dtype=torch.float32).view(-1, 3)
    
    my_conv = pktorch.Genred('SqNorm2(x-y)', ['x = Vi(3)', 'y = Vj(3)'])
    print(my_conv(x, y))

  should return:

  .. code-block:: console

    Compiling libKeOpstorch91c92bd508 in /path/to/build_dir/:
       formula: Sum_Reduction(SqNorm2(x-y),1)
       aliases: x = Vi(0,3); y = Vj(1,3); 
       dtype  : float32
    ... Done. 

  .. code-block:: python

    tensor([[63.],
            [90.]])


Troubleshooting
===============

Compilation issues
------------------

First of all, make sure that you are using a C++ compiler which is compatible with the **C++11 revision** and/or your **nvcc** (CUDA) compiler. Otherwise, compilation of formulas may fail in unexpected ways. Depending on your system, you can:

1. Install a compiler **system-wide**: for instance, on Debian based Linux distros, this can be done by installing g++ with apt and then using `update-alternatives <https://askubuntu.com/questions/26498/choose-gcc-and-g-version>`_ to choose the right compiler.

2. Install a compiler **locally**: if you are using a conda environment, you can install a new instance of gcc and g++ by following the `documentation of conda <https://conda.io/docs/user-guide/tasks/build-packages/compiler-tools.html>`_.


.. _`part.cache`:

Cache directory
---------------

If you experience problems with compilation (or numerical inaccuracies after a KeOps update), it may be a good idea to **flush the build folder** (i.e. the cache of already-compiled formulas). To get the directory name, just type:

.. code-block:: python

  import pykeops
  print(pykeops.bin_folder)



Verbosity level
---------------

To help debugging, you can activate a **verbose** compilation mode. It may be done by defining the environment variable `PYKEOPS_VERBOSE` to 1. In a terminal

.. code-block:: bash

  export PYKEOPS_VERBOSE=1
  python my_script_calling_pykeops.py

Or directly in your python script by setting **after** your KeOps imports the flag pykeops.verbose to true. It gives in a python shell something like: 

.. code-block:: python

  import pykeops
  pykeops.verbose = True


Build type
----------

You can force the (re)compilation of the KeOps shared objects by changing the build type from ``Release`` (default) to ``Debug``. This may be done by defining the environment variable ``PYKEOPS_BUILD_TYPE`` , either in a terminal:

.. code-block:: bash

  export PYKEOPS_BUILD_TYPE="Debug"
  python my_script_calling_pykeops.py

Or directly in your python script, altering the value of the (string) variable ``pykeops.build_type`` right **after** your KeOps imports. In a python shell, simply type: 

.. code-block:: python

  import pykeops
  pykeops.build_type = 'Debug'

.. warning::
  Beware! The shared objects generated in debug mode are **not optimized**,
  and should thus be deleted at the end of your debugging session. 
  In order to do so, please **flush your cache directory** as described in the :ref:`previous section <part.cache>`.
