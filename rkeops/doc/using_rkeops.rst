Using RKeOps
============

Warning !!
----------

RKeOps is currently based on KeOps versions up to 1.5 only. The following instructions allow you to install
RKeOps either from CRAN (v1.4.2.2) or from the KeOps repository, v1.5.

.. raw:: html

   <div id="TOC">

-  `Installing RKeOps <#installing-rkeops>`__

   -  `Requirements <#requirements>`__
   -  `Install from CRAN <#install-from-cran>`__
   -  `Install from Github sources <#install-from-github-sources>`__
   -  `Get sources and install from local
      repository <#get-sources-and-install-from-local-repository>`__

-  `How to use RKeOps <#how-to-use-rkeops>`__

   -  `Example <#example>`__
   -  `Formula <#formula>`__
   -  `Arguments <#arguments>`__

      -  `Input matrix <#input-matrix>`__
      -  `Notations <#notations>`__

   -  `Creating a new operator <#creating-a-new-operator>`__
   -  `Run computations <#run-computations>`__
   -  `Computing gradients <#computing-gradients>`__
   -  `RKeOps options <#rkeops-options>`__

      -  `Compile options <#compile-options>`__
      -  `Choosing CPU or GPU computing at
         runtime <#choosing-cpu-or-gpu-computing-at-runtime>`__
      -  `Other runtime options <#other-runtime-options>`__

   -  `Advanced use <#advanced-use>`__

      -  `Precision <#precision>`__
      -  `Data storage orientation <#data-storage-orientation>`__
      -  `Compilation files and
         cleaning <#compilation-files-and-cleaning>`__

.. raw:: html

   </div>

| RKeOps is a R front-end for the KeOps C++/Cuda library. It provides
  standard functions that can be used in any R code.

Thanks to RKeOps, you can use **GPU computing directly inside R**
without the cost of developing a specific CUDA implementation of your
custom mathematical operators.

.. raw:: html

   <div id="installing-rkeops" class="section level1">

.. rubric:: Installing RKeOps
   :name: installing-rkeops

.. raw:: html

   <div id="requirements" class="section level2">

.. rubric:: Requirements
   :name: requirements

-  R (tested with R >= 3.5)
-  Cmake (>=3.10)
-  C++ compiler (g++ >=7 or clang) for CPU computing or CUDA compiler
   (nvcc >=10) and CUDA libs for GPU computing

**Disclaimer:** KeOps (including RKeOps) is not functional on Windows,
it was only tested on Linux and MacOS.

.. raw:: html

   </div>

.. raw:: html

   <div id="install-from-cran" class="section level2">

.. rubric:: Install from CRAN
   :name: install-from-cran

    **Note:** RKeOps is avaible on CRAN but only for UNIX environment
    (GNU/Linux and MacOS) and not for Windows.

.. raw:: html

   <div class="sourceCode">

.. code:: r

    install.packages("rkeops")

.. raw:: html

   </div>

.. raw:: html

   </div>

.. raw:: html

   <div id="install-from-github-sources" class="section level2">

.. rubric:: Install from Github sources
   :name: install-from-github-sources

    !! In most recent version of devtools, the ``args`` argument is not
    available anymore and it is not possible to use
    ``devtools::install_git``. Please check next section to install from
    sources.

-  Install directly from Github (requires ``git``)

.. raw:: html

   <div class="sourceCode">

.. code:: r

    devtools::install_git("https://github.com/getkeops/keops",
                          ref = "v1.5",
                          subdir = "rkeops",
                          args="--recursive")
    # not possible to use `devtools::intall_github()` because of the required submodule

.. raw:: html

   </div>

.. raw:: html

   </div>

.. raw:: html

   <div id="get-sources-and-install-from-local-repository"
   class="section level2">

.. rubric:: Get sources and install from local repository
   :name: get-sources-and-install-from-local-repository

-  Get KeOps sources (bash command)

   .. raw:: html

      <div class="sourceCode">

   .. code:: bash

       git clone --recurse-submodules="keops/lib/sequences" -b v1.5 https://github.com/getkeops/keops
       # or
       git clone -b v1.5 https://github.com/getkeops/keops
       cd keops
       git submodule update --init -- keops/lib/sequences
       # other submodules are not necessary for RKeOps

   .. raw:: html

      </div>

-  Install from local source in R (assuming you are in the ``keops``
   directory)

.. raw:: html

   <div class="sourceCode">

.. code:: r

    devtools::install("rkeops")

.. raw:: html

   </div>

--------------

.. raw:: html

   </div>

.. raw:: html

   </div>

.. raw:: html

   <div id="how-to-use-rkeops" class="section level1">

.. rubric:: How to use RKeOps
   :name: how-to-use-rkeops

Load RKeOps in R:

.. raw:: html

   <div class="sourceCode">

.. code:: r

    library(rkeops)
    ##
    ## You are using rkeops version 1.4.2

.. raw:: html

   </div>

RKeOps allows to define and compile new operators that run computations
on GPU.

.. raw:: html

   <div id="example" class="section level2">

.. rubric:: Example
   :name: example

.. raw:: html

   <div class="sourceCode">

.. code:: r

    # implementation of a convolution with a Gaussian kernel
    formula = "Sum_Reduction(Exp(-s * SqNorm2(x - y)) * b, 0)"
    # input arguments
    args = c("x = Vi(3)",      # vector indexed by i (of dim 3)
             "y = Vj(3)",      # vector indexed by j (of dim 3)
             "b = Vj(6)",      # vector indexed by j (of dim 6)
             "s = Pm(1)")      # parameter (scalar)
    # compilation
    op <- keops_kernel(formula, args)
    # data and parameter values
    nx <- 100
    ny <- 150
    X <- matrix(runif(nx*3), nrow=nx)   # matrix 100 x 3
    Y <- matrix(runif(ny*3), nrow=ny)   # matrix 150 x 3
    B <- matrix(runif(ny*6), nrow=ny)   # matrix 150 x 6
    s <- 0.2

    # to run computation on CPU (default mode)
    use_cpu()
    # to run computations on GPU (to be used only if relevant)
    use_gpu()

    # computation (order of the input arguments should be similar to `args`)
    res <- op(list(X, Y, B, s))

.. raw:: html

   </div>

The different elements (formula, arguments, compilation, computation) in
the previous example will be detailled in the next sections.

.. raw:: html

   </div>

.. raw:: html

   <div id="formula" class="section level2">

.. rubric:: Formula
   :name: formula

To use RKeOps and define new operators, you need to write the
corresponding *formula* which is a text string defining a composition of
mathematical operations. It should be characterized by two elements:

1. a composition of generic functions applied to some input matrices,
   whose one of their dimensions is either indexed by \\(i=1,...,M\\) or
   \\(j=1,...,N\\)

2. a reduction over indexes \\(i=1,...,M\\) (row-wise) or
   \\(j=1,...,N\\) (column-wise) of the \\(M \\times N\\) matrix whose
   entries are defined by 1.

| RKeOps implements a wide range of mathematical operators and
  reduction: please refer to
  https://www.kernel-operations.io/keops/api/math-operations.html for
  more details.

**Example:** We want to implement the following kernel-based reduction
(convolution with a Gaussian kernel): \\[\\sum\_{j=1}^{N}
\\exp\\Big(-\\sigma \|\| \\mathbf x\_i - \\mathbf y\_j
\|\|\_2^{\\,2}\\Big)\\,\\mathbf b\_j\\] with

-  | parameter: \\(\\sigma\\in\\mathbb R\\)

-  | \\(i\\)-indexed variables \\([\\mathbf x\_i]\_{i=1,...,M}
     \\in\\mathbb R^{M\\times 3}\\)

-  | \\(j\\)-indexed variables \\([\\mathbf y\_j]\_{j=1,...,N}
     \\in\\mathbb R^{N\\times 3}\\) and \\([\\mathbf b\_j]\_{j=1,...,N}
     \\in\\mathbb R^{N\\times 6}\\)

In R, we can define the corresponding KeOps formula as a simple **text
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

.. raw:: html

   </div>

.. raw:: html

   <div id="arguments" class="section level2">

.. rubric:: Arguments
   :name: arguments

The formula describing your computation can take several input
arguments: variables and parameters. The input variables will generally
correspond to rows or columns of your data matrices, you need to be
cautious with their dimensions.

.. raw:: html

   <div id="input-matrix" class="section level3">

.. rubric:: Input matrix
   :name: input-matrix

| You can use two types of input matrices with RKeOps:

-  | ones whose rows (or columns) are indexed by \\(i=1,...,M\\) such as
     \\(\\mathbf X = [x\_{ik}]\_{M \\times D}\\)

-  | others whose rows (or columns) are indexed by \\(j=1,...,N\\) such
     as \\(\\mathbf Y = [y\_{ik'}]\_{N \\times D'}\\)

| The dimensions over indexes \\(i\\) or \\(j\\) are called the **outer
  dimensions** (i.e. \\(M\\) or \\(N\\)). The other dimensions (i.e.
  \\(D\\) or \\(D'\\)) are called the **inner dimensions**. These terms
  refer to the contiguity of the data in memory:

-  | **Outer dimensions** \\(M\\) and \\(N\\) (over indexes \\(i\\) and
     \\(j\\) respectively) can be **very large**, even to large for GPU
     memory.

-  | **Inner dimensions** \\(D\\) and \\(D'\\) should be **small**
     enough to fit in GPU memory, in particular to ensure data
     colocality and avoid useless memory transfers. Corresponding
     columns (or rows) should be contiguous in memory (this point is
     handled for you in RKeOps, see this
     `section <#data-storage-orientation>`__).

    ***Note 1:*** The outer dimension can correspond to the rows or the
    columns of the input matrices (and vice-versa for the inner
    dimension). The optimal orientation of input matrices is discussed
    in this `section <#data-storage-orientation>`__ .

    | ***Note 2:*** All matrices indexed by \\(i\\) should have the same
      outer dimension \\(M\\) over \\(i\\), same for all matrices
      indexed by \\(j\\) (outer dimension \\(N\\)). Only the inner
      dimensions \\(D\\) and \\(D'\\) should be known for the
      compilation of your operators. The respective outer dimensions
      \\(M\\) and \\(N\\) are set at runtime (and can change from one
      run to another).

.. raw:: html

   </div>

.. raw:: html

   <div id="notations" class="section level3">

.. rubric:: Notations
   :name: notations

Input arguments of the formula are defined by using keywords, they can
be of different types:

+-----------+-----------------------------+
| keyword   | meaning                     |
+===========+=============================+
| ``Vi``    | variable indexed by ``i``   |
+-----------+-----------------------------+
| ``Vj``    | variable indexed by ``j``   |
+-----------+-----------------------------+
| ``Pm``    | parameter                   |
+-----------+-----------------------------+

You should provide a vector of text string specifying the name and the
type of all arguments in your formula.

| Each keyword takes as parameter the inner dimension of the
  corresponding object. For instance, to define an input variable
  indexed by \\(i\\) corresponding to a \\(D\\)-dimensional vector, you
  can use ``"Vi(D)"``, same for a \\(D\\)-dimensional variable indexed
  by \\(j\\) being ``"Vj(D)"`` or a \\(D\\)-dimensional parameter
  ``"Pm(D)"``.

The vector of arguments should be

.. raw:: html

   <div class="sourceCode">

.. code:: r

    args = c("<name1>=<type1>(dim1)", "<name2>=<type2>(dim2)", "<nameX>=<typeX>(dimX)")

.. raw:: html

   </div>

where

-  ``<nameX>`` is the name
-  ``<type1>`` is the type (among ``Vi``, ``Vj`` or ``Pm``)
-  ``<dimX>`` is the **inner dimension**

| of the ``X``\ \\(^\\text{th}\\) variable in the formula.

    ***Important:*** The names should correspond to the ones used in the
    formula. The input parameter order will be the one used when calling
    the compiled operator.

**Example:** We define the corresponding arguments of the previous
`formula <#formula>`__, i.e. parameters or variables indexed by \\(i\\)
or \\(j\\) with their corresponding inner dimensions:

.. raw:: html

   <div class="sourceCode">

.. code:: r

    args = c("x = Vi(3)",      # vector indexed by i (of dim 3)
             "y = Vj(3)",      # vector indexed by j (of dim 3)
             "b = Vj(6)",      # vector indexed by j (of dim 6)
             "s = Pm(1)")      # parameter (scalar)

.. raw:: html

   </div>

.. raw:: html

   </div>

.. raw:: html

   </div>

.. raw:: html

   <div id="creating-a-new-operator" class="section level2">

.. rubric:: Creating a new operator
   :name: creating-a-new-operator

By using the function ``keops_kernel``, based on the formula and its
arguments that we previously defined, we can compile and load into R the
corresponding operator:

.. raw:: html

   <div class="sourceCode">

.. code:: r

    # compilation
    op <- keops_kernel(formula, args)

.. raw:: html

   </div>

| Calling ``keops_kernel(formula, args)`` returns a function that can be
  later used to run computations on your data with your value of
  parameters. You should only be cautious with the similarity of each
  argument inner dimension.

The returned function (here ``op``) expects a list of input values in
the order specified in the vector ``args``.

The result of compilation (shared library file) is stored on the system
and will be reused when calling again the function ``keops_kernel`` on
the same formula with the same arguments and the same conditions (e.g.
precision), to avoid useless recompilation.

.. raw:: html

   </div>

.. raw:: html

   <div id="run-computations" class="section level2">

.. rubric:: Run computations
   :name: run-computations

We generate data with inner dimensions (number of columns) corresponding
to each argument expected by the operator ``op``. The function ``op``
takes in input a list of input arguments. If the list if named, ``op``
checks the association between the supplied names and the names of the
formula arguments. In this case only, it can also correct the order of
the input list to match the expected order of arguments.

.. raw:: html

   <div class="sourceCode">

.. code:: r

    # data and parameter values
    nx <- 100
    ny <- 150
    X <- matrix(runif(nx*3), nrow=nx)   # matrix 100 x 3
    Y <- matrix(runif(ny*3), nrow=ny)   # matrix 150 x 3
    B <- matrix(runif(ny*6), nrow=ny)   # matrix 150 x 6
    s <- 0.2

    # to run computation on CPU (default mode)
    use_cpu()
    # to run computations on GPU (to be used only if relevant)
    use_gpu()

    # computation (order of the input arguments should be similar to `args`)
    res <- op(list(x, y, beta, s))

.. raw:: html

   </div>

.. raw:: html

   </div>

.. raw:: html

   <div id="computing-gradients" class="section level2">

.. rubric:: Computing gradients
   :name: computing-gradients

You can define gradients directly in the formula, e.g.

.. raw:: html

   <div class="sourceCode">

.. code:: r

    # defining a formula with a Gradient
    formula <- "Grad(Sum_Reduction(SqNorm2(x-y), 0), x, eta)"
    args <- c("x=Vi(0,3)", "y=Vj(1,3)", "eta=Vi(2,1)")
    # compiling the corresponding operator
    op <- keops_kernel(formula, args)

    # data
    nx <- 100
    ny <- 150
    x <- matrix(runif(nx*3), nrow=nx, ncol=3)     # matrix 100 x 3
    y <- matrix(runif(ny*3), nrow=ny, ncol=3)     # matrix 150 x 3
    eta <- matrix(runif(nx*1), nrow=nx, ncol=1)   # matrix 100 x 1

    # computation
    input <- list(x, y, eta)
    res <- op(input)

.. raw:: html

   </div>

where ``eta`` is the new variable at which the gradient is computed, its
dimension should correspond to the output dimension of the operation
inside the gradient (here ``SqNorm2(x-y)`` is of dimension 1).

You can also use the function ``keops_grad`` to derive existing KeOps
operators.

.. raw:: html

   <div class="sourceCode">

.. code:: r

    # defining an operator (reduction on squared distance)
    formula <- "Sum_Reduction(SqNorm2(x-y), 0)"
    args <- c("x=Vi(0,3)", "y=Vj(1,3)")
    op <- keops_kernel(formula, args)
    # defining its gradient regarding x
    grad_op <- keops_grad(op, var="x")

    # data
    nx <- 100
    ny <- 150
    x <- matrix(runif(nx*3), nrow=nx, ncol=3)     # matrix 100 x 3
    y <- matrix(runif(ny*3), nrow=ny, ncol=3)     # matrix 150 x 3
    eta <- matrix(runif(nx*1), nrow=nx, ncol=1)   # matrix 100 x 1

    # computation
    input <- list(x, y, eta)
    res <- grad_op(input)

.. raw:: html

   </div>

**Note:** when defining a gradient, the operator created by
``keops_grad``\ requires an additional variable whose inner dimension
corresponds to the output dimension of the derived formula (here
``SqNorm2(x-y)`` is a real-valued function, hence dimension 1) and outer
dimension corresponds to the outer dimension of the variable regarding
which the gradient is taken (here ``x``).

.. raw:: html

   </div>

.. raw:: html

   <div id="rkeops-options" class="section level2">

.. rubric:: RKeOps options
   :name: rkeops-options

RKeOps behavior is driven by specific options in ``R`` global options
scope. Such options are set up when loading RKeOps (i.e. by calling
``library(rkeops)``).

You can get the current values of RKeOps options with

.. raw:: html

   <div class="sourceCode">

.. code:: r

    get_rkeops_options()

.. raw:: html

   </div>

To (re)set RKeOps options to default values, run:

.. raw:: html

   <div class="sourceCode">

.. code:: r

    set_rkeops_options()

.. raw:: html

   </div>

To set a specific option with a given value, you can do:

.. raw:: html

   <div class="sourceCode">

.. code:: r

    set_rkeops_option(option, value)
    # `option` = text string, name of the option to set up
    # `value` = whatever value to assign to the chosen option

.. raw:: html

   </div>

Check ``?set_rkeops_option`` for more details.

.. raw:: html

   <div id="compile-options" class="section level3">

.. rubric:: Compile options
   :name: compile-options

-  ``use_cuda_if_possible``: by default, user-defined operators are
   compiled for GPU if CUDA is available (and compiled for CPU
   otherwise).

.. raw:: html

   <div class="sourceCode">

.. code:: r

    # enable compiling for GPU if available (not necessary if using default options)
    compile4gpu()
    # or equivalently
    set_rkeops_option("use_cuda_if_possible", 1)
    # disable compiling for GPU
    set_rkeops_option("use_cuda_if_possible", 0)

.. raw:: html

   </div>

-  ``precision``: by default, user-defined operators are compiled to use
   float 32bits for computations (faster than float 64bits or double,
   compensated sum is available to reduce errors inherent to float
   32bits operations)

.. raw:: html

   <div class="sourceCode">

.. code:: r

    set_rkeops_option("precision", "float")    # float 32bits (default)
    set_rkeops_option("precision", "double")   # float 64bits

.. raw:: html

   </div>

You can directly change the precision used in compiled operators with
the functions ``compile4float32`` and ``compile4float64`` which
respectively enable float 32bits precision (default) and float 64bits
(or double) precision.

-  other compile options (including boolean value to enable verbosity or
   to add debugging flag), see ``?compile_options``

.. raw:: html

   </div>

.. raw:: html

   <div id="choosing-cpu-or-gpu-computing-at-runtime"
   class="section level3">

.. rubric:: Choosing CPU or GPU computing at runtime
   :name: choosing-cpu-or-gpu-computing-at-runtime

By default, RKeOps runs computations on CPU (even for GPU-compiled
operators). To enable GPU computing, you can run (before calling your
operator):

.. raw:: html

   <div class="sourceCode">

.. code:: r

    use_gpu()
    # see `?runtime_options` for a more advanced use of GPU inside RKeOps

.. raw:: html

   </div>

You can also specify the GPU id that you want to use, e.g.
``use_gpu(device=0)`` to use GPU 0 (default) for instance.

To deactivate GPU computations, you can run ``use_cpu()``.

    In CPU mode, you can control the number of CPU cores used by RKeOps
    for computations, e.g. with ``use_cpu(ncore = 2)`` to run on 2
    cores.

.. raw:: html

   </div>

.. raw:: html

   <div id="other-runtime-options" class="section level3">

.. rubric:: Other runtime options
   :name: other-runtime-options

-  ``device_id``: choose on which GPU the computations will be done,
   default is 0.

.. raw:: html

   <div class="sourceCode">

.. code:: r

    set_rkeops_option("device_id", 0)

.. raw:: html

   </div>

***Note***: We recommend to handle GPU assignation outside RKeOps, for
instance by setting the environment variable ``CUDA_VISIBLE_DEVICES``.
Thus, you can keep the default GPU device id = 0 in RKeOps.

-  Other runtime options, see ``?runtime_options``

.. raw:: html

   </div>

.. raw:: html

   </div>

.. raw:: html

   <div id="advanced-use" class="section level2">

.. rubric:: Advanced use
   :name: advanced-use

.. raw:: html

   <div id="precision" class="section level3">

.. rubric:: Precision
   :name: precision

By default, RKeOps uses float 32bits precision for computations. Since R
only considers 64bits floating point numbers, if you want to use float
32bits, input data and output results will be casted before and after
computations respectively in your RKeOps operator. If your application
requires to use float 64bits (double) precision, keep in mind that you
will suffer a performance loss (potentially not an issue on high-end
GPUs). In any case, compensated summation reduction is available in
KeOps to correct for 32bits floating point arithmetic errors.

.. raw:: html

   </div>

.. raw:: html

   <div id="data-storage-orientation" class="section level3">

.. rubric:: Data storage orientation
   :name: data-storage-orientation

| In R, matrices are stored using a column-major order, meaning that an
  \\(M \\times D\\) matrix is stored in memory as a succession of
  \\(D\\) vectors of length \\(M\\) representing each of its columns. A
  consequence is that two successive entries of a column are contiguous
  in memory, but two successive entries of a row are separated by
  \\(M\\) elements. See this
  `page <https://en.wikipedia.org/wiki/Row-_and_column-major_order>`__
  for more details.

For RKeOps to be computationnally efficient, it is important that
elements of the input matrices are contiguous along the inner dimensions
\\(D\\) (or \\(D'\\)). Thus, it is recommended to use input matrices
where the outer dimensions (i.e. indexes \\(i\\) or \\(j\\)) are the
columns, and inner dimensions the rows, e.g. transpose matrices
\\(\\mathbf X^{t} = [x\_{ki}]\_{D \\times M}\\) or \\(\\mathbf Y^{t} =
[y\_{k'i}]\_{D' \\times N}\\).

    | ***Important:*** In machine learning and statistics, we generally
      use data matrices where each sample/observation/individual is a
      row, i.e. matrices where the outer dimensions correspond to rows,
      e.g. \\(\\mathbf X = [x\_{ik}]\_{M \\times D}\\), \\(\\mathbf Y =
      [y\_{ik'}]\_{N \\times D'}\\).
    | This is the default using case of RKeOps. RKeOps will then
      automatically convert your matrices to their transpose, where the
      outer dimensions correspond to columns.
    | If you want to use data where the inner dimension directly
      corresponds to rows of your matrices, i.e. \\(\\mathbf X^{t} =
      [x\_{ki}]\_{D \\times M}\\) or \\(\\mathbf Y^{t} = [y\_{k'i}]\_{D'
      \\times N}\\), you just need to specify the input parameter
      ``inner_dim=0`` when calling your operator.

Example:

.. raw:: html

   <div class="sourceCode">

.. code:: r

    # standard column reduction of a matrix product
    op <- keops_kernel(formula = "Sum_Reduction((x|y), 1)",
                       args = c("x=Vi(3)", "y=Vj(3)"))

    # data (inner dimension = columns)
    nx <- 10
    ny <- 15
    # x_i = rows of the matrix X
    X <- matrix(runif(nx*3), nrow=nx, ncol=3)
    # y_j = rows of the matrix Y
    Y <- matrix(runif(ny*3), nrow=ny, ncol=3)
    # computing the result (here, by default `inner_dim=1` and columns correspond
    # to the inner dimension)
    res <- op(list(X,Y))

    # data (inner dimension = rows)
    nx <- 10
    ny <- 15
    # x_i = columns of the matrix X
    X <- matrix(runif(nx*3), nrow=3, ncol=nx)
    # y_j = columns of the matrix Y
    Y <- matrix(runif(ny*3), nrow=3, ncol=ny)
    # computing the result (we specify `inner_dim=0` to indicate that rows
    # correspond to the inner dimension)
    res <- op(list(X,Y), inner_dim=0)

.. raw:: html

   </div>

.. raw:: html

   </div>

.. raw:: html

   <div id="compilation-files-and-cleaning" class="section level3">

.. rubric:: Compilation files and cleaning
   :name: compilation-files-and-cleaning

The compilation of new operators produces shared library (or share
object ``.so``) files stored in a ``build`` sub-directory of the package
installation directory, to be reused and avoid recompilation of already
defined operators.

You can check where your compiled operators are stored by running
``get_build_dir()``. To clean RKeOps install and remove all shared
library files, you can run ``clean_rkeops()``.

.. raw:: html

   </div>

.. raw:: html

   </div>

.. raw:: html

   </div>
