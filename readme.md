![logo](./doc/_static/logo/keops_logo.png)

[![Build Status](https://ci.inria.fr/keops/buildStatus/icon?job=keops%2Fmaster)](https://ci.inria.fr/keops/job/keops/job/master/)

# Kernel Operations on the GPU, with autodiff, without memory overflows

The KeOps library lets you compute generic reductions of **very large arrays**
whose entries are given by a mathematical formula. 
It combines a **tiled reduction scheme** with an **automatic differentiation**
engine, and can be used through **Matlab**, **NumPy** or **PyTorch** backends.
It is perfectly suited to the computation of **Kernel dot products**
and the associated gradients,
even when the full kernel matrix does *not* fit into the GPU memory.

Using the **PyTorch backend**, a typical sample of code looks like:

```python
# Create two arrays with 3 columns and a (huge) number of lines, on the GPU
import torch
x = torch.randn(1000000, 3, requires_grad=True).cuda()
y = torch.randn(2000000, 3).cuda()

# Turn our Tensors into KeOps symbolic variables:
from pykeops import LazyTensor
x_i = LazyTensor( x[:,None,:] )  # x_i.shape = (1e6, 1, 3)
y_j = LazyTensor( y[None,:,:] )  # y_j.shape = ( 1, 2e6,3)

# We can now perform large-scale computations, without memory overflows:
D_ij = ((x_i - y_j)**2).sum(dim=2)  # Symbolic (1e6,2e6,1) matrix of squared distances
K_ij = (- D_ij).exp()               # Symbolic (1e6,2e6,1) Gaussian kernel matrix

# And come back to vanilla PyTorch Tensors or NumPy arrays using
# reduction operations such as .sum(), .logsumexp() or .argmin().
# Here, the kernel density estimation   a_i = sum_j exp(-|x_i-y_j|^2)
# is computed using a CUDA online map-reduce routine that has a linear
# memory footprint and outperforms standard PyTorch implementations
# by two orders of magnitude.
a_i = K_ij.sum(dim=1)  # Genuine torch.cuda.FloatTensor, a_i.shape = (1e6, 1),
g_x = torch.autograd.grad((a_i ** 2).sum(), [x])  # KeOps supports autograd!
```

KeOps allows you to leverage your GPU without compromising on usability.
It provides:

* **Linear** (instead of quadratic) **memory footprint** for Kernel operations.
* Support for a wide range of mathematical **formulas**.
* Seamless computation of **derivatives**, up to arbitrary orders.
* Sum, LogSumExp, Min, Max but also ArgMin, ArgMax or K-min **reductions**.
* A **conjugate gradient solver** for e.g. large-scale spline interpolation or kriging, Gaussian process regression.
* An interface for **block-sparse** and coarse-to-fine strategies.
* Support for **multi GPU** configurations.

KeOps can thus be used in a wide variety of settings,
from shape analysis (LDDMM, optimal transport...)
to machine learning (kernel methods, k-means...)
or kriging (aka. Gaussian process regression).
More details are provided below:

* [Installation](http://www.kernel-operations.io/keops/introduction/installation.html)
* [Documentation](http://www.kernel-operations.io/keops/introduction/why_using_keops.html)
* [Learning KeOps with tutorials](http://www.kernel-operations.io/keops/_auto_tutorials/index.html)
* [Gallery of examples](http://www.kernel-operations.io/keops/_auto_examples/index.html)
* [Benchmarks](http://www.kernel-operations.io/keops/_auto_benchmarks/index.html)

# Projects using KeOps

As of today, KeOps provides core routines for:

* [Deformetrica](http://www.deformetrica.org), a shape analysis software
  developed by the [Aramis](https://www.inria.fr/en/teams/aramis) Inria team.
* [GeomLoss](http://www.kernel-operations.io/geomloss), a multiscale
  implementation of Kernel and **Wasserstein distances** that scales up to
  **millions of samples** on modern hardware.
* [FshapesTk](https://plmlab.math.cnrs.fr/benjamin.charlier/fshapesTk) and the
  [Shapes toolbox](https://plmlab.math.cnrs.fr/jeanfeydy/shapes_toolbox),
  two research-oriented [LDDMM](https://en.wikipedia.org/wiki/Large_deformation_diffeomorphic_metric_mapping) toolkits.

# Authors

Feel free to contact us for any bug report or feature request:

* [Benjamin Charlier](http://imag.umontpellier.fr/~charlier/)
* [Jean Feydy](https://www.math.ens.fr/~feydy/)
* [Joan Alexis Glaunès](https://www.mi.parisdescartes.fr/~glaunes/)
