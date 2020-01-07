Installing RKeOps
==============

Requirements
------------

- A unix-like system (typically Linux or MacOs X) with a C++ compiler compatible with std=c++14 (e.g. gcc>=7 or clang>=8)
- Cmake>=3.10
- R
- Optional: Cuda (>=10.0 is recommended)

Packaged version (recommended)
------------------------------

Install from the CRAN:

.. code-block:: R

    install.package("rkeops")

From source using git
---------------------

Clone the KeOps repository at a location of your choice (denoted here by ``/path/to``):

.. code-block:: bash

    git clone --recursive https://github.com/getkeops/keops.git /path/to/keops


.. code-block:: R
    
    devtools::install("/path/to/keops/rkeops")

Troubleshooting
---------------

No known bug at the moment.
