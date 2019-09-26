A blind spot in the literature
================================

**Memory usage, performances.**
Out of the box, the tensor-centric interfaces of PyTorch,
TensorFlow, Matlab and NumPy strike a **good balance between
power and simplicity**: explicit matrices allow users to implement
standard engineering tools with a code that stays close to the maths.

But there is a limit to what full matrices can handle: **whenever the
operators involved present some structure, baseline matrix-vector
products can be vastly outperformed by domain-specific
implementations.** Some examples of this rule are well-known and
supported by major frameworks through dedicated methods and “layers”:

-  In Image Processing, **convolutions**, **Fourier** and **Wavelet
   transforms** rely on ad hoc schemes that do not involve circulant or
   Vandermonde matrices.

-  On graph or mesh data, **sparse matrices** are encoded as lists of
   indices plus coefficients and provide support for local operators:
   graph Laplacians, divergences, etc.

**KeOps: adding support for symbolic tensors.**
Surprisingly, though, little to no effort has been made to support
generic **mathematical** or **“symbolic” matrices**, which are not
*sparse* in the traditional sense but can nevertheless be described
compactly in memory using a *symbolic formula* and some small data
arrays.

Allowing the users of **kernel or distance matrices** to bypass the
*transfer* and *storage* of superfluous quadratic buffers is the main
purpose of the KeOps library. As a bite-sized example of our
interface, the program below is a revision of the script presented 
in the previous section that **scales up to clouds of** :math:`\mathrm{N}\,=\,1,000,000` 
**samples in less than a second on modern hardware**, 
with a linear memory footprint – remark the
absence of any :math:`\mathrm{N}`-by-:math:`\mathrm{N}` buffer in the graph.

.. code-block:: python

    from pykeops.torch import LazyTensor  # Semi-symbolic wrapper for torch Tensors
    q_i  = LazyTensor( q[:,None,:] )  # (N,D) Tensor -> (N,1,D) Symbolic Tensor
    q_j  = LazyTensor( q[None,:,:] )  # (N,D) Tensor -> (1,N,D) Symbolic Tensor

    D_ij = ((q_i - q_j) ** 2).sum(dim=2)  # Symbolic matrix of squared distances
    K_ij = (- D_ij / (2 * s**2) ).exp()   # Symbolic Gaussian kernel matrix
    v    = K_ij@p  # Genuine torch Tensor. (N,N)@(N,D) = (N,D)

    # Finally, compute the kernel norm H(q,p):
    H = .5 * torch.dot( p.view(-1), v.view(-1) ) # .5 * <p,v>
    # Display the computational graph in the figure below, annotated by hand:
    make_dot(H, {'q':q, 'p':p}).render(view=True)


.. figure:: images/hamiltonian_KP.svg
    :width: 70% 
    :alt: Computational graph
    :align: center