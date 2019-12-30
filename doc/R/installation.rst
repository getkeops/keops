gnu R Install
==============

Requirements
------------

- A unix-like system (typically Linux or MacOs X) with a C++ compiler compatible with std=c++14 (e.g. gcc>=7 or clang>=8)
- Cmake>=3.10
- R
- Optional: Cuda (>=10.0 is recommended)

Packaged version (recommended)
------------------------------

1. Install from the CRAN:

.. code-block:: R

    install.package("rkeops")

2. :ref:`Test your installation <part.rtest>`.

Note that binary files will be written into the ``....`` folder: this directory must have write permissions.


From source using git
---------------------

1. Clone the KeOps repository at a location of your choice (denoted here by ``/path/to``):
    

.. code-block:: bash

    git clone https://github.com/getkeops/keops.git /path/to/keops


Note that binary files will be written into the ``????/path/to/keops/rkeops/build`` folder: this directory must have write permissions.

2. :ref:`Test your installation <part.rtest>`.


.. _part.rtest:

Test that everything goes fine
------------------------------

Install RKeOps on your system and execute the following piece of code in a R terminal:

.. code-block:: R

    x = reshape(1:9,3,[]); y = reshape(3:8,3,[]);

    my_conv = keops_kernel('Sum_Reduction(SqNorm2(x-y),1)','x=Vi(0,3)','y=Vj(1,3)');
    my_conv(x,y)'

It should return:

.. code-block:: R

    ans =
        63
        90


Troubleshooting
---------------

No known bug at the moment.
