Introduction to RKeOps
======================

.. raw:: html

   <div id="TOC">

-  `Authors <#authors>`__
-  `Citation <#citation>`__
-  `What is RKeOps? <#what-is-rkeops>`__

   -  `KeOps <#keops>`__
   -  `RKeOps <#rkeops>`__
   -  `Why using RKeOps? <#why-using-rkeops>`__

-  `Matrix reduction and kernel
   operator <#matrix-reduction-and-kernel-operator>`__

   -  `What you need to do <#what-you-need-to-do>`__
   -  `Example in R <#example-in-r>`__
   -  `Generic kernel function <#generic-kernel-function>`__
   -  `CPU and GPU computing <#cpu-and-gpu-computing>`__

-  `Installing and using RKeOps <#installing-and-using-rkeops>`__

.. raw:: html

   </div>

-  URL: https://www.kernel-operations.io/
-  Source: https://github.com/getkeops/keops
-  Licence and Copyright: see
   https://github.com/getkeops/keops/blob/main/licence.txt

.. raw:: html

   <div id="authors" class="section level1">

.. rubric:: Authors
   :name: authors

Please contact us for any **bug report**, **question** or **feature
request** by filing a report on our `GitHub issue
tracker <https://github.com/getkeops/keops/issues>`__!

**Core library - KeOps, PyKeOps, KeOpsLab:**

-  `Benjamin Charlier <https://imag.umontpellier.fr/~charlier/>`__, from
   the University of Montpellier.
-  `Jean Feydy <https://www.jeanfeydy.com>`__, from Imperial College
   London.
-  `Joan Alexis
   Glaunès <http://helios.mi.parisdescartes.fr/~glaunes/>`__, from the
   University of Paris.

**R bindings - RKeOps:**

-  `Ghislain Durif <https://gdurif.perso.math.cnrs.fr/>`__, from the
   University of Montpellier.

**Contributors:**

-  `François-David Collin <https://github.com/fradav>`__, from the
   University of Montpellier: Tensordot operation, CI setup.
-  `Tanguy Lefort <https://github.com/tanglef>`__, from the University
   of Montpellier: conjugate gradient solver.
-  `Mauricio Diaz <https://github.com/mdiazmel>`__, from Inria 
   of Paris: CI setup.
-  `Benoît Martin <https://github.com/benoitmartin88>`__, from the
   Aramis Inria team: multi-GPU support.
-  `Francis Williams <https://www.fwilliams.info>`__, from New York
   University: maths operations.
-  `Kshiteej Kalambarkar <https://github.com/kshitij12345>`__, from
   Quansight: maths operations.
-  `D. J. Sutherland <https://djsutherland.ml>`__, from the TTI-Chicago:
   bug fix in the Python package.
-  `David
   Völgyes <https://scholar.google.no/citations?user=ngT2GvMAAAAJ&hl=en>`__,
   from the Norwegian Institute of Science and Technology: bug fix in
   the formula parser.

Beyond explicit code contributions, KeOps has grown out of numerous
discussions with applied mathematicians and machine learning experts. We
would especially like to thank `Alain
Trouvé <https://atrouve.perso.math.cnrs.fr/>`__, `Stanley
Durrleman <https://who.rocq.inria.fr/Stanley.Durrleman/>`__, `Gabriel
Peyré <http://www.gpeyre.com/>`__ and `Michael
Bronstein <https://people.lu.usi.ch/bronstem/>`__ for their valuable
suggestions and financial support.

--------------

.. raw:: html

   </div>

.. raw:: html

   <div id="citation" class="section level1">

.. rubric:: Citation
   :name: citation

If you use this code in a research paper, **please cite** our `original
publication <https://jmlr.org/papers/v22/20-275.html>`__:

    Charlier, B., Feydy, J., Glaunès, J. A., Collin, F.-D. & Durif, G.
    Kernel Operations on the GPU, with Autodiff, without Memory
    Overflows. Journal of Machine Learning Research 22, 1–6 (2021).

.. raw:: html

   <div class="sourceCode">

.. code:: tex

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

.. raw:: html

   </div>

For applications to **geometric (deep) learning**, you may also consider
our `NeurIPS 2020
paper <https://www.jeanfeydy.com/Papers/KeOps_NeurIPS_2020.pdf>`__:

.. raw:: html

   <div class="sourceCode">

.. code:: tex

    @article{feydy2020fast,
        title={Fast geometric learning with symbolic matrices},
        author={Feydy, Jean and Glaun{\`e}s, Joan and Charlier, Benjamin and Bronstein, Michael},
        journal={Advances in Neural Information Processing Systems},
        volume={33},
        year={2020}
    }

.. raw:: html

   </div>

--------------

.. raw:: html

   </div>

.. raw:: html

   <div id="what-is-rkeops" class="section level1">

.. rubric:: What is RKeOps?
   :name: what-is-rkeops

RKeOps is the R package interfacing the KeOps library.
`Here <https://gdurif.perso.math.cnrs.fr/files/material/slides_Toulouse_2019_Durif_KeOps.pdf>`__
you can find a few slides explaining functionalities of the KeOps
library.

.. raw:: html

   <div id="keops" class="section level2">

.. rubric:: KeOps
   :name: keops

    Seamless Kernel Operations on GPU (or CPU), with
    auto-differentiation and without memory overflows

The KeOps library (http://www.kernel-operations.io) provides routines to
compute generic reductions of large 2d arrays whose entries are given by
a mathematical formula. Using a C++/CUDA-based implementation with GPU
support, it combines a tiled reduction scheme with an automatic
differentiation engine. Relying on online map-reduce schemes, it is
perfectly suited to the scalable computation of kernel dot products and
the associated gradients, even when the full kernel matrix does not fit
into the GPU memory.

KeOps is all about breaking through this memory bottleneck and making
GPU power available for seamless standard mathematical routine
computations. As of 2019, this effort has been mostly restricted to the
operations needed to implement Convolutional Neural Networks: linear
algebra routines and convolutions on grids, images and volumes. KeOps
provides CPU and GPU support without the cost of developing a specific
CUDA implementation of your custom mathematical operators.

To ensure its versatility, KeOps can be used through Matlab, Python
(NumPy or PyTorch) and R back-ends.

.. raw:: html

   </div>

.. raw:: html

   <div id="rkeops" class="section level2">

.. rubric:: RKeOps
   :name: rkeops

| RKeOps is a library that can

-  | Compute **generic reduction** (row-wise or column-wise) of very
     large array/matrices, i.e. \\[\\sum\_{i=1}^M a\_{ij} \\ \\ \\ \\
     \\text{or}\\ \\ \\ \\ \\sum\_{j=1}^N a\_{ij}\\] for some matrix
     \\(A = [a\_{ij}]\_{M \\times N}\\) with \\(M\\) rows and \\(N\\)
     columns, whose entries \\(a\_{ij}\\) can be defined with basic math
     formulae or matrix operators.

-  | Compute **kernel dot products**, i.e. \\[\\sum\_{i=1}^M K(\\mathbf
     x\_i, \\mathbf y\_j)\\ \\ \\ \\ \\text{or}\\ \\ \\ \\
     \\sum\_{j=1}^N K(\\mathbf x\_i, \\mathbf y\_j)\\] for a kernel
     function \\(K\\) and some vectors \\(\\mathbf x\_i\\), \\(\\mathbf
     y\_j\\in \\mathbb{R}^D\\) that are generally rows of some data
     matrices \\(\\mathbf X = [x\_{ik}]\_{M \\times D}\\) and
     \\(\\mathbf Y = [y\_{jk}]\_{N \\times D}\\) respectively.

-  | Compute the **associated gradients**

    ***Applications***: RKeOps can be used to implement a wide range of
    problems encountered in ***machine learning***, ***statistics*** and
    more: such as \\(k\\)-nearest neighbor classification, \\(k\\)-means
    clustering, Gaussian-kernel-based problems (e.g. linear system with
    Ridge regularization), etc.

.. raw:: html

   </div>

.. raw:: html

   <div id="why-using-rkeops" class="section level2">

.. rubric:: Why using RKeOps?
   :name: why-using-rkeops

| RKeOps provides

-  | an API to create **user-defined operators** based on generic
     mathematical formulae, that can be applied to data matrices such as
     \\(\\mathbf X = [x\_{ik}]\_{M \\times D}\\) and \\(\\mathbf Y =
     [y\_{jk}]\_{N \\times D}\\).

-  | fast computation on **GPU** without memory overflow, especially to
     process **very large dimensions** \\(M\\) and \\(N\\) (e.g.
     \\(\\approx 10^4\\) or \\(10^6\\)) over indexes \\(i\\) and
     \\(j\\).

-  | automatic differentiation and **gradient computations** for
     user-defined operators.

--------------

.. raw:: html

   </div>

.. raw:: html

   </div>

.. raw:: html

   <div id="matrix-reduction-and-kernel-operator"
   class="section level1">

.. rubric:: Matrix reduction and kernel operator
   :name: matrix-reduction-and-kernel-operator

| The general framework of RKeOps (and KeOps) is to provide fast and
  scalable matrix operations on GPU, in particular kernel-based
  computations of the form \\[\\underset{i=1,...,M}{\\text{reduction}}\\
  G(\\boldsymbol{\\sigma}, \\mathbf x\_i, \\mathbf y\_j) \\ \\ \\ \\
  \\text{or}\\ \\ \\ \\ \\underset{j=1,...,N}{\\text{reduction}}\\
  G(\\boldsymbol{\\sigma}, \\mathbf x\_i, \\mathbf y\_j)\\] where

-  | \\(\\boldsymbol{\\sigma}\\in\\mathbb R^L\\) is a vector of
     parameters

-  | \\(\\mathbf x\_i\\in \\mathbb{R}^D\\) and \\(\\mathbf y\_j\\in
     \\mathbb{R}^{D'}\\) are two vectors of data (potentially with
     different dimensions)

-  | \\(G: \\mathbb R^L \\times \\mathbb R^D \\times \\mathbb R^{D'}
     \\to \\mathbb R\\) is a function of the data and the parameters,
     that can be expressed through a composition of generic operators

-  | \\(\\text{reduction}\\) is a generic reduction operation over the
     index \\(i\\) or \\(j\\) (e.g. sum)

| RKeOps creates (and compiles on the fly) an operator implementing your
  formula. You can apply it to your data, or compute its gradient
  regarding some data points.

    ***Note:*** You can use a wide range of reduction such as ``sum``,
    ``min``, ``argmin``, ``max``, ``argmax``, etc.

.. raw:: html

   <div id="what-you-need-to-do" class="section level2">

.. rubric:: What you need to do
   :name: what-you-need-to-do

| To use RKeOps you only need to express your computations as a formula
  with the previous form.

| RKeOps allows to use a wide range of mathematical functions to define
  your operators (see
  https://www.kernel-operations.io/keops/api/math-operations.html).

| You can use two type of input matrices with RKeOps:

-  | ones whose rows (or columns) are indexed by \\(i=1,...,M\\) such as
     \\(\\mathbf X = [x\_{ik}]\_{M \\times D}\\)

-  | others whose rows (or columns) are indexed by \\(j=1,...,N\\) such
     as \\(\\mathbf Y = [y\_{ik'}]\_{N \\times D'}\\)

More details about input matrices (size, storage order) are given in the
vignette 'Using RKeOps'.

.. raw:: html

   </div>

.. raw:: html

   <div id="example-in-r" class="section level2">

.. rubric:: Example in R
   :name: example-in-r

We want to implement with RKeOps the following mathematical formula
\\[\\sum\_{j=1}^{N} \\exp\\Big(-\\sigma \|\| \\mathbf x\_i - \\mathbf
y\_j \|\|\_2^{\\,2}\\Big)\\,\\mathbf b\_j\\] with

-  | parameter: \\(\\sigma\\in\\mathbb R\\)

-  | \\(i\\)-indexed variables \\(\\mathbf X = [\\mathbf
     x\_i]\_{i=1,...,M} \\in\\mathbb R^{M\\times 3}\\)

-  | \\(j\\)-indexed variables \\(\\mathbf Y = [\\mathbf
     y\_j]\_{j=1,...,N} \\in\\mathbb R^{N\\times 3}\\) and \\(\\mathbf B
     = [\\mathbf b\_j]\_{j=1,...,N} \\in\\mathbb R^{N\\times 6}\\)

In R, we can define the corresponding KeOps formula as a **simple text
string**:

.. raw:: html

   <div class="sourceCode">

.. code:: r

    formula = "Sum_Reduction(Exp(-s * SqNorm2(x - y)) * b, 0)"

.. raw:: html

   </div>

-  ``SqNorm2`` = squared \\(\\ell\_2\\) norm
-  ``Exp`` = exponential
-  ``Sum_reduction(..., 0)`` = sum reduction over the dimension 0 i.e.
   sum on the \\(j\\)'s (1 to sum over the \\(i\\)'s)

and the corresponding arguments of the formula, i.e. parameters or
variables indexed by \\(i\\) or \\(j\\) with their corresponding inner
dimensions:

.. raw:: html

   <div class="sourceCode">

.. code:: r

    args = c("x = Vi(3)",      # vector indexed by i (of dim 3)
             "y = Vj(3)",      # vector indexed by j (of dim 3)
             "b = Vj(6)",      # vector indexed by j (of dim 6)
             "s = Pm(1)")      # parameter (scalar) 

.. raw:: html

   </div>

Then we just compile the corresponding operator and apply it to some
data

.. raw:: html

   <div class="sourceCode">

.. code:: r

    # compilation
    op <- keops_kernel(formula, args)
    # data and parameter values
    nx <- 100
    ny <- 150
    X <- matrix(runif(nx*3), nrow=nx)   # matrix 100 x 3
    Y <- matrix(runif(ny*3), nrow=ny)   # matrix 150 x 3
    B <- matrix(runif(ny*6), nrow=ny)   # matrix 150 x 6
    s <- 0.2
    # computation (order of the input arguments should be similar to `args`)
    res <- op(list(X, Y, B, s))

.. raw:: html

   </div>

.. raw:: html

   </div>

.. raw:: html

   <div id="generic-kernel-function" class="section level2">

.. rubric:: Generic kernel function
   :name: generic-kernel-function

| With RKeOps, you can define kernel functions \\(K: \\mathbb R^D
  \\times \\mathbb R^D \\to \\mathbb R\\) such as, for some vectors
  \\(\\mathbf x\_i\\), \\(\\mathbf y\_j\\in \\mathbb{R}^D\\)

-  | the linear kernel (standard scalar product) \\(K(\\mathbf x\_i,
     \\mathbf y\_j) = \\big\\langle \\mathbf x\_i \\, , \\, \\mathbf
     y\_j \\big\\rangle\\)

-  | the Gaussian kernel \\(K(\\mathbf x\_i, \\mathbf y\_j) =
     \\exp\\left(-\\frac{1}{2\\sigma^2} \|\| \\mathbf x\_i - \\mathbf
     y\_j \|\|\_2^{\\,2}\\right)\\) with \\(\\sigma>0\\)

-  | and more...

| Then you can compute reductions based on such functions, especially
  when the \\(M \\times N\\) matrix \\(\\mathbf K = [K(\\mathbf x\_i,
  \\mathbf y\_j)]\\) is too large to fit into memory, such as

-  Kernel reduction: \\[\\sum\_{i=1}^M K(\\mathbf x\_i, \\mathbf y\_j)\\
   \\ \\ \\ \\text{or}\\ \\ \\ \\ \\sum\_{j=1}^N K(\\mathbf x\_i,
   \\mathbf y\_j)\\]

-  | Convolution-like operations: \\[\\sum\_{i=1}^M K(\\mathbf x\_i,
     \\mathbf y\_j)\\boldsymbol\\beta\_j\\ \\ \\ \\ \\text{or}\\ \\ \\
     \\ \\sum\_{j=1}^N K(\\mathbf x\_i, \\mathbf
     y\_j)\\boldsymbol\\beta\_j\\] for some vectors
     \\((\\boldsymbol\\beta\_j)\_{j=1,...,N} \\in \\mathbb R^{N\\times
     D}\\)

-  More complex operations: \\[\\sum\_{i=1}^{M}\\, K\_1(\\mathbf x\_i,
   \\mathbf y\_j)\\, K\_2(\\mathbf u\_i, \\mathbf v\_j)\\,\\langle
   \\boldsymbol\\alpha\_i\\, ,\\,\\boldsymbol\\beta\_j\\rangle \\ \\ \\
   \\ \\text{or}\\ \\ \\ \\ \\sum\_{j=1}^{N}\\, K\_1(\\mathbf x\_i,
   \\mathbf y\_j)\\, K\_2(\\mathbf u\_i, \\mathbf v\_j)\\,\\langle
   \\boldsymbol\\alpha\_i\\, ,\\,\\boldsymbol\\beta\_j\\rangle\\] for
   some kernel \\(K\_1\\) and \\(K\_2\\), and some \\(D\\)-vectors
   \\((\\mathbf x\_i)\_{i=1,...,M}, (\\mathbf u\_i)\_{i=1,...,M},
   (\\boldsymbol\\alpha\_i)\_{i=1,...,M} \\in \\mathbb R^{M\\times D}\\)
   and \\((\\mathbf y\_j)\_{j=1,...,N}, (\\mathbf v\_j)\_{j=1,...,N},
   (\\boldsymbol\\beta\_j)\_{j=1,...,N} \\in \\mathbb R^{N\\times D}\\)

.. raw:: html

   </div>

.. raw:: html

   <div id="cpu-and-gpu-computing" class="section level2">

.. rubric:: CPU and GPU computing
   :name: cpu-and-gpu-computing

Based on your formulae, RKeOps compile on the fly operators that can be
used to run the corresponding computations on CPU or GPU, it uses a
tiling scheme to decompose the data and avoid (i) useless and costly
memory transfers between host and GPU (performance gain) and (ii) memory
overflow.

    ***Note:*** You can use the same code (i.e. define the same
    operators) for CPU or GPU computing. The only difference will be the
    compiler used for the compilation of your operators (upon the
    availability of CUDA on your system).

To use CPU computing mode, you can call ``use_cpu()`` (with an optional
argument ``ncore`` specifying the number of cores used to run parallel
computations).

To use GPU computing mode, you can call ``use_gpu()`` (with an optional
argument ``device`` to choose a specific GPU id to run computations).

--------------

.. raw:: html

   </div>

.. raw:: html

   </div>

.. raw:: html

   <div id="installing-and-using-rkeops" class="section level1">

.. rubric:: Installing and using RKeOps
   :name: installing-and-using-rkeops

See the specific vignette **Using RKeOps**.

.. raw:: html

   </div>
