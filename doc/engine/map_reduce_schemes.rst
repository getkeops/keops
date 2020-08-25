Map-Reduce schemes
================================


The **most important piece of code** in the KeOps package is the
one-dimensional, heavily templated Map-Reduce scheme that can be found
in the 
`GpuConv1D.cu <https://github.com/getkeops/keops/blob/master/keops/core/mapreduce/GpuConv1D.cu>`_ 
CUDA file. Used as a **default backend** by the
:mod:`Genred` operator, this standard distributed algorithm relies on
principles that are exposed in the reference 
`CUDA programming guide <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory>`_

In a nutshell, this scheme may be described as a 
**tiled “for” loop on the reduction index**  :math:`j`, 
parallelized over the sole
index :math:`i` – hence the “1D” denomination – which reduces the
computed values of :math:`F(p^1,\dots, x^1_i, \dots, y^1_j, \dots)`
**on-the-fly, without ever storing or sending them to the Device
memory.**

The 1D algorithm 
----------------------

More precisely, as illustrated in the Figure below – with the
standard C++ convention of indexes that range “from :math:`0` to
:math:`\mathrm{N} - 1`” – we may decompose the instructions executed by our CUDA
blocks as follows:

#. Each block of :math:`\mathrm{K}` threads is attributed an index
   :math:`\mathrm{A}` that ranges between :math:`0` and
   :math:`\lceil \mathrm{M} / \mathrm{K} \rceil - 1`. This number may exceed the
   physical number of blocks that can run simultaneously on the GPU
   chip, but the **nvcc** compiler abstracts these technicalities away.
   |br|


#. In every block, the :math:`\mathrm{K}` threads are indexed by an
   integer
   :math:`k \in [\![0, \mathrm{K}[\![~~ = [\![0, \mathrm{K}-1]\!]`. The
   :math:`k`-th thread is assigned to a fixed value of
   :math:`i = k + \mathrm{AK}`. It loads the relevant values of
   :math:`p^1, \dots, p^P` and :math:`x^1_i, \dots, x^X_i` **from the
   Device to the Thread memory** or *register*, taking advantage of the
   speed-up for contiguous memory accesses: threads in the same block
   read neighboring memory adresses. Once again, the compiler handles
   the distribution of :math:`\mathrm{K}` virtual workers on a fixed
   number of physical CUDA cores.
   |br|


#. Each thread is instructed to compute a single value
   :math:`a_i = a_{k+\mathrm{AK}}` through a “for” loop on the
   values of :math:`j` in :math:`\left[\!\left[ 1,\mathrm{N} \right]\!\right]`. To minimize the
   transfers between the Device and Shared memories while maximizing the
   amount of contiguous memory accesses (as discussed page ), this
   :math:`j`-loop is cut in blocks of size :math:`\mathrm{K}`: **the
   large** :math:`\mathrm{M}`\ **-by-**\ :math:`\mathrm{N}` **plane 
   of** :math:`(i,j)` **indices is
   effectively cut in small** :math:`\mathrm{K}`\ **-by-**\ :math:`\mathrm{K}` **tiles**, following
   a standard CUDA procedure. Having initialized a temporary buffer
   “:math:`a`” (in the Thread memory) to the neutral element of the
   :math:`\operatorname{Reduction}` – :math:`0` if it is a sum,
   :math:`+\infty` if it is a minimum, etc. – the :math:`k`-th thread of
   the block **loops** over values of the **tile index**
   :math:`\mathrm{B}` in
   :math:`\left[\!\left[0, \lceil \mathrm{N} / \mathrm{K} \rceil - 1\right]\!\right]`:
   |br|


   #. Being assigned to an index :math:`j_k = k + \mathrm{BK}`, the
      worker loads the relevant values of
      :math:`y^1_{j_k}, \dots, y^Y_{j_k}` **from the Device to the
      Shared memory**. This task is performed in conjunction with the
      other threads of the block and comes down to a contiguous transfer
      of a slice “:math:`j \in [\![ \mathrm{BK}, \mathrm{BK + K} [\![`”
      of the :math:`y`-data arrays from the “library” of the State
      department to the shared “office shelf”.
      |br|


   #. The thread waits for latecomers and **synchronizes** with all
      workers in the same block: we don’t want to start the computing
      job if some of the :math:`y_j`\ ’s have not yet been loaded
      properly in the Shared memory!
      |br|


   #. **Making a loop** over the reduction index :math:`j` in
      :math:`[\![\mathrm{BK}, \mathrm{BK + K} [\![`, the worker:
      |br|


      #. Loads the relevant values of the :math:`y_j`\ ’s **from the
         Shared to the Thread memory**.


      #. **Computes** the value of
         :math:`F(p^1,\dots, x^1_i, \dots, y^1_j, \dots)`, with all
         variables standing close to the computing core in the Thread
         memory.


      #. **Reduces** this value onto the running buffer :math:`a`, in
         the Thread memory.


   #. Once again, the thread **synchronizes** with the other workers.


#. Once this large outer loop has been completed, the buffer :math:`a`
   associated to the :math:`k`-th thread contains our final value
   :math:`a_{k+\mathrm{AK}}`. It is then saved **from the Thread to the
   Device memory** in an appropriate “output” array, alongside the other
   values in the “:math:`i \in [\![\mathrm{AK}, \mathrm{AK + K} [\![`”
   range that have been computed by the block.



.. list-table::

  * - .. figure:: images/matmat.svg
         :alt: Naive scheme.
         :width: 80%

         ..

         |br| **(a)** Simple, ideal scheme.

    - .. figure:: images/scheme_1D.svg
         :alt: Tiled scheme.

         ..

         |br| **(b)** Optimized GPU scheme. 


**Figure.** The default 1D Map-Reduce scheme used by the KeOps :mod:`Genred` engine
can be described as a simple **loop over the reduction index** :math:`j`
that is **optimized for GPU chips**.
**(a)** Each thread :math:`i` 
computes one of the :math:`a_i`\ 's by looping
over the reduction index :math:`j` and eating-up the values of :math:`F` 
**on-the-fly**.
**(b)** Due to the importance
of the **Shared memory and block-wise memory accesses**, **(a)** is cut in 
:math:`\mathrm{K}`-by-:math:`\mathrm{K}` tiles to ensure an optimal
management of the :math:`y_j`\ 's.


Performances
----------------

As most efficient CUDA programs, the algorithm presented above is pretty
verbose: a full page of tedious memory transfers surrounds what is, at
heart, a good old **“for-for” loop**. Crucially though, our efforts pay
off: as evidenced by our :doc:`benchmarks <../_auto_benchmarks/plot_benchmarks_convolutions_3D>`, 
KeOps typically provides a
**x30/x10,000 speed-up** when compared with tensorized
PyTorch-GPU/NumPy-CPU implementations of the same kernel dot
product, while keeping a **linear (instead of quadratic) memory
footprint**.

This efficiency mostly comes down to the fact that instead of storing
the :math:`\mathrm{M}`-by-:math:`\mathrm{N}` computed values of
:math:`F(p^1,\dots, x^1_i, \dots, y^1_j, \dots)` in **superfluous
quadratic buffers** (such as the “kernel matrix”), generating at least
:math:`2\mathrm{M}\mathrm{N}` high-latency transfers between **the Thread and the Device
memories**, **KeOps maximizes the use of the Shared memory** and
consumes the relevant values of
:math:`F(p^1,\dots, x^1_i, \dots, y^1_j, \dots)` on-the-spot, in the
registers of the CUDA cores.

Note that **this level of performance could not have been achieved with
high-level Python code**: PyTorch and TensorFlow variables always refer
to arrays that are stored in the **Device memory**. Writing C++
CUDA programs is the only way of getting an explicit access to the
**Shared and Thread memories**. As discussed in
the next sections, supporting **generic** formulas and reductions
with KeOps thus required the implementation of a fully-fledged
symbolic math engine, **within the CUDA framework**, that could be
executed *inside* our loop at steps 3.3.2-3.


The 2D algorithm
-------------------

The “**GPU_1D**” algorithm that we just presented is efficient whenever
:math:`\mathrm{M}` is larger than the number of CUDA cores available on the
chip: no thread stays idle. This is generally the case in shape analysis
and data sciences, where the support of **batch processing** by KeOps
allows programmers to fully **saturate their GPUs** with large input
tensors.

Nevertheless, to provide cover for cases where the number of “indexing
lines” :math:`\mathrm{M}` is much smaller than the size of the “reduction range”
:math:`\mathrm{N}`, KeOps also implements a **2D Map-Reduce scheme** in the
`GpuConv2D.cu <https://github.com/getkeops/keops/blob/master/keops/core/mapreduce/GpuConv2D.cu>`_ 
CUDA file. Assigning the
:math:`\mathrm{K}`-by-:math:`\mathrm{K}` tiles of
the computation plan **one-by-one to the CUDA blocks** – instead of
using a line-wise grouping method – this algorithm requires the
allocation of intermediate buffers but makes sure that no block stays
idle during the computation.


.. |br| raw:: html

  <br/><br/>