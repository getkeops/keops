Kernel MMDs, Optimal Transport
------------------------------------------

Thanks to its support of the **Sum** and **LogSumExp** reductions,
KeOps is perfectly suited to the large-scale computation of
`Kernel norms <https://en.wikipedia.org/wiki/Reproducing_kernel_Hilbert_space>`_
and `Sinkhorn divergences <https://arxiv.org/abs/1803.00567>`_.
Going further, the :ref:`block-sparse routines <part.sparsity>` allow us
to implement **genuine coarse-to-fine strategies** that scale
(almost) linearly with the number of samples,
as advocated in `(Schmitzer, 2016) <https://arxiv.org/abs/1610.06519>`_.

Relying on the KeOps routines :func:`generic_sum() <pykeops.torch.generic_sum>` and
:func:`generic_logsumexp() <pykeops.torch.generic_logsumexp>`,
the `GeomLoss library <https://www.kernel-operations.io/geomloss>`_
provides **Geometric Loss functions** as simple PyTorch layers,
with a fully-fledged `gallery of examples <https://www.kernel-operations.io/geomloss/_auto_examples/index.html>`_.
Implemented on the GPU for the very first time, these routines 
**outperform the standard Sinkhorn algorithm by a factor 50-100** 
and redefine the state-of-the-art
for discrete Optimal Transport: on modern hardware, 
**Wasserstein distances between clouds of 100,000 points can now be
computed in less than a second**.

