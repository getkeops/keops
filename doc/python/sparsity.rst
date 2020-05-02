Block-sparse reductions
#######################
.. _`part.sparsity`:


Complexity of the KeOps routines
================================

KeOps is built around online map-reduce routines
that have a **quadratic time complexity**: if :math:`M` and
:math:`N` denote the number of :math:`i` and :math:`j` variables,
the cost of a generic reduction scales asymptotically in :math:`O(MN)`.
This is most evident in our :doc:`convolution benchmark <../_auto_benchmarks/plot_benchmark_convolutions>`,
where we compute all the kernel coefficients :math:`K(x_i,y_j) = \exp(-\|x_i-y_j\|^2)`
to implement a discrete convolution:

.. math::
    a~=~ K_{xy} \,b \qquad\text{i.e.}\qquad
    a_i ~=~ \sum_{j=1}^N K(x_i,y_j)\cdot b_j \qquad \text{for $i\in[1,M]$.}

**Can we do better?**
To go beyond this quadratic lower bound,
a simple idea is to use a **sparsity prior**
on the kernel matrix to **skip some computations**. 
For instance, we could decide to skip kernel computations
when **points** :math:`x_i` **and** :math:`y_j` **are far apart from each other**.
But can we do so **efficiently**?

Sparsity on the CPU
-------------------

On CPUs, a standard strategy is to use `sparse matrices <https://en.wikipedia.org/wiki/Sparse_matrix>`_
and encode our operators with **lists of non-zero coefficients and indices**.
Schematically, this comes down to endowing each index :math:`i\in[1,M]`
with a set :math:`J_i\subset[1,N]` of :math:`j`-neighbors,
and to restrict ourselves to the computation of


.. math::
    a_i ~=~ \sum_{j\in J_i} K(x_i,y_j)\cdot b_j, \qquad \text{for $i\in[1,M]$.}

This approach is very well suited to matrices with a handful of nonzero coefficients per line, 
e.g. the intrinsic Laplacian of a 3D mesh.
But on large, densely connected problems, sparse encoding
runs into a major issue: as it relies on **non-contiguous** memory accesses,
it scales **very poorly** on parallel hardware.

Block-sparsity on the GPU
-------------------------

As explained on the `NVidia devblog <https://devblogs.nvidia.com/how-access-global-memory-efficiently-cuda-c-kernels/>`_,
GPUs rely on **coalesced memory operations** which load blocks
of **dozens of contiguous bytes** at once. Instead of allowing the
use of arbitrary index sets :math:`J_i` for all lines of our sparse kernel matrix,
we should thus restrict ourselves to computations of the form:

.. math::
    a_i ~=~ \sum_{l=1}^{S_k} \sum_{j=\text{start}^k_l}^{\text{end}^k_l-1} K(x_i,y_j)\cdot b_j, \qquad 
    \text{for $i \in [\text{start}_k, \text{end}_k)$ and $k\in [1,K]$,}

where:

- The :math:`[\text{start}_k, \text{end}_k)` intervals form a **partition** of the
  set of :math:`i`-indices :math:`[1,M]`:

  .. math::
    [1,M]~=~ \bigsqcup_{k=1}^K \,[\text{start}_k, \text{end}_k).

- For every segment :math:`k\in[1,K]`, the :math:`S_k` intervals
  :math:`[\text{start}^k_l, \text{end}^k_l)` encode a set of **neighbors**
  as a finite collection of contiguous index ranges:

  .. math::
    \forall~i\in[\text{start}_k, \text{end}_k), ~ 
    J_i~=~ \bigsqcup_{l=1}^{S_k} \,[\text{start}^k_l, \text{end}^k_l).

By encoding our sparsity patterns as **block-wise binary masks**
made up of tiles :math:`T^k_l~=~[\text{start}_k, \text{end}_k) \times [\text{start}^k_l, \text{end}^k_l) \subset [1,M]\times[1,N]`,
we can leverage coalesced memory operations for maximum efficiency on the GPU. 
As long as our index ranges are **wider than the buffer size**,
we should get close to **optimal performances**.

**Going further.** This scheme can be generalized to **generic**
formulas and reductions. For reductions with respect to the :math:`i` axis,
we'd simply have to define *transposed* tiles 
:math:`U^k_l~=~[\text{start}^k_l, \text{end}^k_l) \times [\text{start}_k, \text{end}_k) \subset [1,M]\times[1,N]`
and restrict ourselves to computations of the form:

.. math::
    b_j ~=~ \sum_{l=1}^{S_k} \sum_{i=\text{start}^k_l}^{\text{end}^k_l-1} K(x_i,y_j)\cdot a_i, \qquad 
    \text{for $j \in [\text{start}_k, \text{end}_k)$ and $k\in [1,K]$.}


**A decent trade-off.** This **block-wise** approach to sparse reductions may seem a bit
too **coarse**, as negligible coefficients get computed to no avail...
But in practice, the **GPU speed-ups** on contiguous memory loads
more than make up for it.

Documentation
=============

See the :doc:`pytorch <api/pytorch/Cluster>` or :doc:`numpy <api/numpy/Cluster>`  api documentation for the syntax.

Examples
========

As documented in e.g. the :class:`numpy.Genred <pykeops.numpy.Genred>` or :class:`torch.Genred <pykeops.torch.Genred>` docstring,
all KeOps reductions accept an optional **ranges** argument,
which can be either ``None`` (i.e. dense, quadratic reduction)
or a 6-uple of integer arrays, which encode
a set of :math:`[\text{start}_k, \text{end}_k)` and
:math:`[\text{start}^k_l, \text{end}^k_l)` intervals
for reductions with respect to the :math:`j` and :math:`i` indices.


A **full tutorial** on block-sparse reductions
is provided in the :doc:`gallery <../_auto_examples/index>`,
for both :doc:`NumPy <../_auto_examples/numpy/plot_grid_cluster_numpy>`
and :doc:`PyTorch <../_auto_examples/pytorch/plot_grid_cluster_pytorch>` APIs.
As you go through these notebooks, you will learn how to:

1. Cluster and sort your data to enforce **contiguity**.
2. Define **coarse binary masks** that encode block-sparse reduction schemes.
3. Turn this information at cluster level into a **ranges** argument that can be used with KeOps' **generic reductions**.
4. **Test** these block-sparse algorithms, and **benchmark** them vs. simpler, dense implementations.


The :mod:`pykeops.numpy.cluster` and :mod:`pykeops.torch.cluster` modules
provide a set of **helper functions** whose interface is described below.
Feel free to use and adapt them to **your own setting**,
beyond the simple case of **Sum** reductions and Gaussian **convolutions**!
