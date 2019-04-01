* v0.1.6


* v0.1.5
    - keops: the upper bound on CudaBlockSize is now computed at compilation time (and not at runtime). So binaries have to be recompiled when the hardware is changing.
    - pykeops: Lock file for the Build dir is now done in python.

* v0.1.4
    - pyKeops compatibility with pytorch 1.0
    - some refactoring of Keopslab code (Kernel function is now called keops_kernel and Grad function is called kernel_grad)

* v0.1.3
    - Fix pickle problem with inherited class in pytorch
    - 'import pykeops' do not try to import torch any more

* v0.1.2
    - Small fixes on wheel packages

* v0.1.0
    - PyKeOps is now using Pybind11
    - Multi GPU support
    - Min, Max ... reduction added
    - doc with sphinx

* v0.0.12   
    - Some performance improvements in the formula (Square, Inv, ...)

* v0.0.10
     - First release of KeOps
