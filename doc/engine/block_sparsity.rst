Block-sparsity
================================


**Complexity of the KeOps routines.**
Notwithstanding their finely grained management of memory, the **GPU_1D** and
**GPU_2D** schemes have a **quadratic time complexity**: if :math:`\mathrm{M}`
and :math:`\mathrm{N}` denote the number of :math:`i` and :math:`j` variables,
the time needed to perform a generic reduction scales asymptotically in
:math:`O(\mathrm{M}\mathrm{N})`. This is most evident in 
our :doc:`convolution benchmarks <../_auto_benchmarks/plot_benchmarks_convolutions_3D>`, 
where all the kernel
coefficients

.. math::

   \begin{aligned}
   K_{ij} ~=~ k(x_i,y_j) ~=~ \exp(-\|x_i-y_j\|^2\,/\,2\sigma^2)\end{aligned}

are computed to implement a discrete Gaussian convolution:

.. math::

   \begin{aligned}
       (a_i)~=~ (K_{ij}) \cdot (b_i) \qquad\text{i.e.}\qquad
       a_i ~=~ \sum_{j=1}^\mathrm{N} k(x_i,y_j)\cdot b_j \qquad \text{for $i\in\left[\!\left[ 1,\mathrm{M} \right]\!\right] $.}
       \end{aligned}

Can we do better?
------------------------

To break through this quadratic lower bound, a simple idea is to **skip
some computations**, using a *sparsity prior* on the kernel matrix. For
instance, we could decide to skip kernel computations when points
:math:`x_i` and :math:`y_j` are far away from each other. But can we do
so *efficiently*?

**Sparsity on the CPU.**
On CPUs, a standard strategy is to use **sparse matrices**, encoding our
operators through lists of non-zero coefficients and indices.
Schematically, this comes down to endowing each index
:math:`i\in\left[\!\left[ 1,\mathrm{M}\right]\!\right]` with a set :math:`J_i\subset\left[\!\left[ 1,\mathrm{N}\right]\!\right]` of
:math:`j`-neighbors and to restrict ourselves to the computation of

.. math::

   \begin{aligned}
       a_i ~=~ \sum_{j\in J_i} k(x_i,y_j)\cdot b_j, \qquad \text{for $i\in\left[\!\left[ 1,\mathrm{M}\right]\!\right]$.}\end{aligned}

This approach is very well suited to matrices which only have a handful
of nonzero coefficients per line, such as the **intrinsic Laplacian** of a
3D mesh. But on large, densely connected problems, sparse encoding runs
into a major issue: since it relies on **non-contiguous** memory
accesses, it scales poorly on GPUs.

Block-sparse reductions
----------------------------

As explained in :doc:`our introduction <../autodiff_gpus/what_is_a_gpu>`,
GPU chips are wired to rely on **coalesced memory
operations** which load blocks of dozens of contiguous bytes at once.
Instead of allowing the use of arbitrary indexing sets :math:`J_i` for
all lines of our sparse kernel matrix, we should thus restrict ourselves
to computations of the form:

.. math::

   \begin{aligned}
       a_i ~=~ \sum_{l=1}^{S_q} \sum_{j=\text{start}^q_l}^{\text{end}^q_l-1} k(x_i,y_j)\cdot b_j, \qquad 
       \text{for $i \in \left[\!\left[\text{start}_q, \text{end}_q\right[\!\right[$~~ and~~ $q\in \left[\!\left[ 1,\mathrm{Q}\right]\!\right]$,}\end{aligned}

where:

#. The
   :math:`\left[\!\left[\text{start}_q, \text{end}_q\right[\!\right[`
   intervals form a **partition** of the set of
   :math:`i`-indices  :math:`\left[\!\left[ 1,\mathrm{M}\right]\!\right]`:

   .. math::

      \begin{aligned}
          \left[\!\left[ 1,\mathrm{M}\right]\!\right] ~=~ \bigsqcup_{q=1}^{\mathrm{Q}} 
          \,\left[\!\left[\text{start}_q, \text{end}_q\right[\!\right[.
        \end{aligned}

#. For every segment :math:`q\in\left[\!\left[ 1,\mathrm{Q}\right]\!\right]`, the :math:`S_q`
   intervals :math:`[\![\text{start}^q_l, \text{end}^q_l[\![` encode a
   set of *neighbors* as a **finite collection of contiguous ranges of
   indices**:

   .. math::

      \begin{aligned}
          \forall~i\in \left[\!\left[\text{start}_q, \text{end}_q\right[\!\right[, ~ 
          J_i~=~ \bigsqcup_{l=1}^{S_q} \,[\![\text{start}^q_l, \text{end}^q_l[\![.
        \end{aligned}

By encoding our sparsity patterns as **block-wise binary masks** made up
of tiles

.. math::

   \begin{aligned}
   T^q_l~=~\left[\!\left[\text{start}_q, \text{end}_q\right[\!\right[ 
         \times 
         [\![\text{start}^q_l, \text{end}^q_l[\![ ~~
         \subset ~
         \left[\!\left[ 1,\mathrm{M}\right]\!\right]\times\left[\!\left[ 1,\mathrm{N}\right]\!\right]~,\end{aligned}

we can leverage coalesced memory operations for **maximum efficiency on
the GPU**. As long as our index ranges are wider than the CUDA blocks, we
should get close to optimal performances.

**Going further.**
This scheme can be generalized to *generic* formulas and reductions. For
reductions with respect to the :math:`i` axis, we simply have to define
*transposed* tiles

.. math::

   \begin{aligned}
   U^q_l~=~[\![\text{start}^q_l, \text{end}^q_l[\![ 
   \times 
   \left[\!\left[\text{start}_q, \text{end}_q\right[\!\right[ ~~
   \subset ~
   \left[\!\left[ 1,\mathrm{M}\right]\!\right]\times\left[\!\left[ 1,\mathrm{N}\right]\!\right]\end{aligned}

and restrict ourselves to computations of the form:

.. math::

   \begin{aligned}
       b_j 
       ~=~ 
       \sum_{l=1}^{S_q} \sum_{i=\text{start}^q_l}^{\text{end}^q_l-1} k(x_i,y_j)\cdot a_i, \qquad 
       \text{for $j \in \left[\!\left[\text{start}_q, \text{end}_q\right[\!\right[$ 
       and $q\in \left[\!\left[ 1,\mathrm{Q}\right]\!\right]$.}\end{aligned}


A decent trade-off
------------------------

This block-wise approach to sparse reductions may seem a bit too coarse,
as some negligible coefficients get computed with little to no impact on
the final result... But in practice, the **GPU speed-ups** on contiguous
memory operations more than make up for it: implemented in the
`GpuConv1D_ranges.cu <https://github.com/getkeops/keops/blob/master/keops/core/mapreduce/GpuConv1D_ranges.cu>`_ 
CUDA file, our block-sparse Map-Reduce scheme is
the workhorse of the **multiscale Sinkhorn algorithm** showcased in
the `GeomLoss library <https://www.kernel-operations.io/geomloss/>`_.

As explained :doc:`in our documentation <../python/sparsity>`, 
the main user interface for KeOps
block-sparse reduction is an optional “**.ranges**” attribute for
:mod:`LazyTensors` which encodes arbitrary block-sparsity masks. In
practice, as illustrated in the Figure below, helper
routines allow users to specify tiled sparsity patterns from **clustered**
arrays of samples :math:`x_i`, :math:`y_j` and coarse
**cluster-to-cluster** boolean matrices. Implementing 
`Barnes-Hut-like strategies <https://en.wikipedia.org/wiki/Barnes%E2%80%93Hut_simulation>`_ 
and other approximation rules
is thus relatively easy, up to a preliminary sorting pass which ensures
that **all clusters are stored contiguously in memory**.


.. list-table::

  * - .. figure:: images/block_sparse_2.png
         :alt: Spiral and Gaussian.

         ..

         |br| **(a)** Point clouds.

    - .. figure:: images/block_sparse_1.png
         :alt: Coarse boolean mask.

         ..

         |br| **(b)** Coarse boolean mask. 


**Figure.**
Illustrating **block-sparse reductions** with 2D point clouds. When
using an :math:`\mathrm{M}`-by-:math:`\mathrm{N}` “kernel” matrix to compute an
interaction term between two datasets, a common approximation strategy
is to **skip terms which correspond to clusters of points that are far
away from each other**. Through a set of helper routines and optional
arguments, KeOps allows users to implement these pruning strategies
efficiently, on the GPU. **(a)** Putting our points in square bins, we
compute the centroid of each cluster. Simple thresholds on
centroid-to-centroid distances allow us to decide that the 43rd “cyan”
cluster of target points :math:`(x_i)` in the spiral should only interact with
neighboring cells of source points :math:`(y_j)` in the Gaussian sample, highlighted in
magenta, etc. **(b)** In practice, this decision is encoded in a **coarse
boolean matrix** that is processed by KeOps, with each line (*resp.*
column) corresponding to a cluster of :math:`x` (*resp.* :math:`y`)
variables. Here, we higlight the 43rd line of our mask which corresponds
to the cyan-magenta points of **(a)**.


.. |br| raw:: html

  <br/><br/>
