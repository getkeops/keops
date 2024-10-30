![logo](./doc/_static/logo/keops_logo.png)

---------------------------------------------------------------------------------

<!-- [![Build status](https://ci.inria.fr/keops/buildStatus/icon?job=keops%2Fmain)](https://ci.inria.fr/keops/job/keops/job/main/) -->
[![PyPI version](https://img.shields.io/pypi/v/pykeops?color=blue)](https://pypi.org/project/pykeops/)
[![PyPI downloads](https://pepy.tech/badge/pykeops?color=green)](https://pypi.org/project/pykeops/)
<!-- [![CRAN version](https://img.shields.io/cran/v/rkeops?color=yellowgreen)](https://cran.r-project.org/web/packages/rkeops/index.html) -->
<!-- [![CRAN downloads](http://cranlogs.r-pkg.org/badges/grand-total/rkeops?color=yellowgreen)](https://cran.r-project.org/web/packages/rkeops/index.html) -->


Please visit our [website](https://www.kernel-operations.io/) for 
documentation, contribution guidelines and tutorials.

# Kernel Operations on the GPU, with autodiff, without memory overflows

The KeOps library lets you compute reductions of **large arrays** 
whose entries are given by a **mathematical formula** or a **neural network**. 
It combines **efficient C++ routines** with an **automatic differentiation** 
engine and can be used with **Python** (NumPy, PyTorch), **Matlab** and **R**.

It is perfectly suited to the computation of **kernel** matrix-vector products,
**K-nearest neighbors** queries, **N-body** interactions,
point cloud **convolutions** and the associated **gradients**.
Crucially, it performs well even when the corresponding kernel or distance 
matrices do *not* fit into the RAM or GPU memory.
Compared with a PyTorch GPU baseline, KeOps provides
a **x10-x100 speed-up** on a wide range of geometric applications,
from kernel methods to geometric deep learning.

![Symbolic matrices](./doc/_static/symbolic_matrix.svg)

**Why using KeOps?**
Math libraries represent most objects as matrices and tensors:

- **(a) Dense matrices.** Variables are often encoded as
  dense numerical arrays M<sub>i,j</sub> = M[i,j].
  This representation is convenient and well-supported,
  but also puts a **heavy load** on the memories of our computers.
  Unfortunately, **large arrays are expensive** to move around and
  may not even fit in RAM or GPU memories.

  In practice, this means that a majority of scientific programs are **memory-bound**.
  Run times for most neural networks and mathematical computations
  are not limited by the raw capabilities of our CPUs and CUDA cores,
  but by the **time-consuming transfers** of large arrays
  from memory circuits to arithmetic computing units.

- **(b) Sparse matrices.** To work around this problem, a common solution is to rely on
  sparse matrices: tensors that have few non-zero coefficients.
  We represent these objects using lists of indices
  (i<sub>n</sub>,j<sub>n</sub>) and values M<sub>n</sub> = M<sub>i<sub>n</sub>,j<sub>n</sub></sub>
  that correspond to a small number of non-zero entries.
  Matrix-vector operations are then implemented with
  indexing methods and scattered memory accesses.

  This method is elegant and allows us to represent large arrays 
  with a small memory footprint.
  But unfortunately, it **does not stream well on GPUs**:
  parallel computing devices are wired to perform *block-wise* memory accesses
  and have a hard time dealing with lists of *random* indices (i<sub>n</sub>,j<sub>n</sub>).
  As a consequence, when compared with dense arrays,
  sparse encodings only speed up computations for matrices
  that have **less than 1% non-zero coefficients**. 
  This restrictive condition prevents sparse matrices 
  from being very useful outside of graph and mesh processing.

- **(c) Symbolic matrices.**
  KeOps provides another solution to **speed up tensor programs**.
  Our key remark is that most of the large arrays that are used
  in machine learning and applied mathematics share a **common mathematical structure**.
  *Distance* matrices, *kernel* matrices, point cloud *convolutions*
  and *attention* layers can all be described as **symbolic** tensors:
  given two collections of vectors (x<sub>i</sub>) and (y<sub>j</sub>),
  their coefficients M<sub>i,j</sub> at location (i,j) are given by
  **mathematical formulas** F(x<sub>i</sub>,y<sub>j</sub>) 
  that are evaluated on data samples x<sub>i</sub> and y<sub>j</sub>.

  These objects are not "sparse" in the traditional sense...
  but can nevertheless be described efficiently using
  a mathematical formula F and relatively
  small data arrays (x<sub>i</sub>) and (y<sub>j</sub>).
  The main purpose of the KeOps library
  is to provide support for this abstraction
  **with all the perks of a deep learning library**:

  - A transparent [interface](https://www.kernel-operations.io/keops/python/api/index.html) with CPU and GPU integration.
  - Numerous [tutorials](https://www.kernel-operations.io/keops/_auto_tutorials/index.html) and [benchmarks](https://www.kernel-operations.io/keops/_auto_benchmarks/index.html).
  - Full support for **automatic differentiation**, **batch processing**
    and approximate computations.
  
In practice, KeOps symbolic tensors are both
**fast** and **memory-efficient**.
We take advantage of the structure of CUDA registers
to bypass costly memory transfers between
arithmetic and memory circuits. This allows us to provide a
**x10-x100 speed-up** to PyTorch GPU programs
in a wide range of settings.

Using our **Python interface**, a typical sample of code looks like:

```python
# Create two arrays with 3 columns and a (huge) number of lines, on the GPU
import torch  # NumPy, Matlab and R are also supported
M, N, D = 1000000, 2000000, 3
x = torch.randn(M, D, requires_grad=True).cuda()  # x.shape = (1e6, 3)
y = torch.randn(N, D).cuda()                      # y.shape = (2e6, 3)

# Turn our dense Tensors into KeOps symbolic variables with "virtual"
# dimensions at positions 0 and 1 (for "i" and "j" indices):
from pykeops.torch import LazyTensor
x_i = LazyTensor(x.view(M, 1, D))  # x_i.shape = (1e6, 1, 3)
y_j = LazyTensor(y.view(1, N, D))  # y_j.shape = ( 1, 2e6,3)

# We can now perform large-scale computations, without memory overflows:
D_ij = ((x_i - y_j)**2).sum(dim=2)  # Symbolic (1e6,2e6,1) matrix of squared distances
K_ij = (- D_ij).exp()               # Symbolic (1e6,2e6,1) Gaussian kernel matrix

# We come back to vanilla PyTorch Tensors or NumPy arrays using
# reduction operations such as .sum(), .logsumexp() or .argmin()
# on one of the two "symbolic" dimensions 0 and 1.
# Here, the kernel density estimation   a_i = sum_j exp(-|x_i-y_j|^2)
# is computed using a CUDA scheme that has a linear memory footprint and
# outperforms standard PyTorch implementations by two orders of magnitude.
a_i = K_ij.sum(dim=1)  # Genuine torch.cuda.FloatTensor, a_i.shape = (1e6, 1), 

# Crucially, KeOps fully supports automatic differentiation!
g_x = torch.autograd.grad((a_i ** 2).sum(), [x])
```

KeOps allows you to **get the most out of your hardware** without compromising on **usability**.
It provides:

- **Linear** (instead of quadratic) **memory footprint** for numerous types of computations.
- Support for a wide range of mathematical **formulas** that can be composed at will.
- Seamless computation of **derivatives** and **gradients**, up to arbitrary orders.
- Sum, LogSumExp, Min, Max but also ArgMin, ArgMax or K-min **reductions**.
- A **conjugate gradient solver** for large-scale spline interpolation and Gaussian process regression.
- Transparent integration with **standard packages**, such as the
  [SciPy solvers](https://docs.scipy.org/doc/scipy/reference/sparse.linalg.html) for linear algebra.
- An interface for **block-sparse** and coarse-to-fine strategies.
- Support for **multi GPU** configurations.

More details are provided below:

- [Installation](http://www.kernel-operations.io/keops/introduction/installation.html)
- [Documentation](http://www.kernel-operations.io/keops/introduction/why_using_keops.html)
- [Learning KeOps with tutorials](http://www.kernel-operations.io/keops/_auto_tutorials/index.html)
- [Gallery of examples](http://www.kernel-operations.io/keops/_auto_examples/index.html)
- [Benchmarks](http://www.kernel-operations.io/keops/_auto_benchmarks/index.html)

# Projects using KeOps

**Symbolic** matrices are to **geometric** learning what **sparse** matrices are to **graph** processing.
KeOps can thus be used in a wide range of settings,
from **shape analysis** (registration, geometric deep learning, optimal transport...)
to **machine learning** (kernel methods, k-means, UMAP...),
**Gaussian processes**, computational **biology** and **physics**.

KeOps provides core routines for the following projects and libraries:

- [GPyTorch](https://docs.gpytorch.ai/en/stable/examples/02_Scalable_Exact_GPs/KeOps_GP_Regression.html>)
  (from the universities of Cornell, Columbia, Pennsylvania) and
  [Falkon](https://falkonml.github.io/falkon/)
  (from the university of Genoa and the [Sierra](https://www.di.ens.fr/sierra/) Inria team),
  two libraries for **Gaussian Process regression** that now scale
  up to **billion-scale datasets**.

- [Deformetrica](http://www.deformetrica.org), a **computational anatomy** software
  from the [Aramis](https://www.inria.fr/en/teams/aramis) Inria team.

- The [Gudhi library](https://gudhi.inria.fr) for **topological data analysis**
  and higher dimensional geometry understanding,
  from the [DataShape](https://team.inria.fr/datashape/) Inria team.

- [GeomLoss](http://www.kernel-operations.io/geomloss), a PyTorch package for Chamfer (Hausdorff) distances, Kernel (Sobolev) divergences and **Earth Mover's (Wasserstein) distances**.
  It provides **optimal transport** solvers that scale up to
  **millions of samples** in seconds.
  
- The [deep graph matching consensus](https://github.com/rusty1s/deep-graph-matching-consensus) module,
  for learning and refining structural correspondences between graphs.

- [FshapesTk](https://plmlab.math.cnrs.fr/benjamin.charlier/fshapesTk) and the
  [Shapes toolbox](https://plmlab.math.cnrs.fr/jeanfeydy/shapes_toolbox),
  two research-oriented [LDDMM](https://en.wikipedia.org/wiki/Large_deformation_diffeomorphic_metric_mapping) toolkits.

- [HyenaDNA](https://github.com/HazyResearch/hyena-dna/) for parallel computations of the Vandermonde matrix multiplication kernel and reductions used in the S4D kernel.

# Licensing, citation, academic use

This library is licensed under the permissive [MIT license](https://en.wikipedia.org/wiki/MIT_License),
which is fully compatible with both **academic** and **commercial** applications.

If you use this code in a research paper, **please cite** our 
[original publication](https://jmlr.org/papers/v22/20-275.html):

> Charlier, B., Feydy, J., Glaunès, J. A., Collin, F.-D. & Durif, G. Kernel Operations on the GPU, with Autodiff, without Memory Overflows. Journal of Machine Learning Research 22, 1–6 (2021).

```tex
@article{JMLR:v22:20-275,
  author  = {Benjamin Charlier and Jean Feydy and Joan Alexis Glaunès and François-David Collin and Ghislain Durif},
  title   = {Kernel Operations on the GPU, with Autodiff, without Memory Overflows},
  journal = {Journal of Machine Learning Research},
  year    = {2021},
  volume  = {22},
  number  = {74},
  pages   = {1-6},
  url     = {http://jmlr.org/papers/v22/20-275.html}
}
```

For applications to **geometric (deep) learning**, 
you may also consider our [NeurIPS 2020 paper](https://www.jeanfeydy.com/Papers/KeOps_NeurIPS_2020.pdf):

```tex
@article{feydy2020fast,
    title={Fast geometric learning with symbolic matrices},
    author={Feydy, Jean and Glaun{\`e}s, Joan and Charlier, Benjamin and Bronstein, Michael},
    journal={Advances in Neural Information Processing Systems},
    volume={33},
    year={2020}
}
```

# Authors

Please contact us for any **bug report**, **question** or **feature request** by filing
a report on our [GitHub issue tracker](https://github.com/getkeops/keops/issues)!

**Core library - KeOps, PyKeOps, KeOpsLab:**

- [Benjamin Charlier](https://imag.umontpellier.fr/~charlier/), from the University of Montpellier.
- [Jean Feydy](https://www.jeanfeydy.com), from the [HeKA team](https://team.inria.fr/heka/fr/) (Inria Paris, Inserm, Université Paris Cité).
- [Joan Alexis Glaunès](http://helios.mi.parisdescartes.fr/~glaunes/), from MAP5 laboratory, Université Paris Cité.

**R bindings - RKeOps:**

- [Ghislain Durif](https://gdurif.perso.math.cnrs.fr/), from the University of Montpellier.

**Contributors:**

- [François-David Collin](https://github.com/fradav), from the University of Montpellier: Tensordot operation, CI setup.
- [Tanguy Lefort](https://github.com/tanglef), from the University of Montpellier: conjugate gradient solver.
- [Amélie Vernay](https://github.com/AmelieVernay) and [Chloé Serre-Combe](https://github.com/chloesrcb), from the University of Montpellier: support for LazyTensors in RKeOps.
- [Mauricio Diaz](https://github.com/mdiazmel), from Inria of Paris: CI setup.
- [Benoît Martin](https://github.com/benoitmartin88), from the Aramis Inria team: multi-GPU support.
- [Francis Williams](https://www.fwilliams.info), from New York University: maths operations.
- [Kshiteej Kalambarkar](https://github.com/kshitij12345), from Quansight: maths operations.
- [Hugo Aguettaz](https://github.com/haguettaz), from ETH Zürich: trigonometric functions.
- [D. J. Sutherland](https://djsutherland.ml), from the TTI-Chicago: bug fix in the Python package.
- [David Völgyes](https://scholar.google.no/citations?user=ngT2GvMAAAAJ&hl=en), from the Norwegian Institute of Science and Technology: bug fix in the formula parser.
- [Jean-Baptiste Keck](https://www.keckj.fr/), from the Univeristy Grenoble-Alpes: bug fix in the Python package.


Beyond explicit code contributions, KeOps has grown out of numerous discussions with applied mathematicians and machine learning experts. We would especially like to thank 
[Alain Trouvé](https://atrouve.perso.math.cnrs.fr/), 
[Stanley Durrleman](https://who.rocq.inria.fr/Stanley.Durrleman/), 
[Gabriel Peyré](http://www.gpeyre.com/) and 
[Michael Bronstein](https://people.lu.usi.ch/bronstem/)
for their valuable suggestions and financial support.

KeOps was awarded an [open science prize](https://www.enseignementsup-recherche.gouv.fr/fr/remise-des-prix-science-ouverte-du-logiciel-libre-de-la-recherche-2023-93732) by the French Ministry of Higher Education and Research in 2023 ("Espoir - Documentation").
