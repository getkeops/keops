import torch

def from_matrix( ranges_i, ranges_j, keep ) :
    r"""Turns a boolean matrix into a KeOps-friendly **ranges** argument.

    This routine is a helper for the **block-sparse** reduction mode of KeOps,
    allowing you to turn clustering information (**ranges_i**,
    **ranges_j**) and a cluster-to-cluster boolean mask (**keep**) 
    into integer tensors of indices that can be used to schedule the KeOps routines.

    Suppose that you're working with variables :math:`x_i`  (:math:`i \in [0,10^6)`),
    :math:`y_j`  (:math:`j \in [0,10^7)`), and that you want to compute a KeOps reduction
    over indices :math:`i` or :math:`j`: Instead of performing the full 
    kernel dot product (:math:`10^6 \cdot 10^7 = 10^{13}` operations!), 
    you may want to restrict yourself to
    interactions between points :math:`x_i` and :math:`y_j` that are "close" to each other.

    With KeOps, the simplest way of doing so is to:
    
    1. Compute cluster labels for the :math:`x_i`'s and :math:`y_j`'s, using e.g. 
       the :func:`grid_cluster` method.
    2. Compute the ranges (**ranges_i**, **ranges_j**) and centroids associated 
       to each cluster, using e.g. the :func:`cluster_ranges_centroids` method.
    3. Sort the tensors ``x_i`` and ``y_j`` with :func:`sort_clusters` to make sure that the
       clusters are stored contiguously in memory (this step is **critical** for performance on GPUs).

    At this point:
        - the :math:`k`-th cluster of :math:`x_i`'s is given by ``x_i[ ranges_i[k,0]:ranges_i[k,1], : ]``, for :math:`k \in [0,M)`, 
        - the :math:`\ell`-th cluster of :math:`y_j`'s is given by ``y_j[ ranges_j[l,0]:ranges_j[l,1], : ]``, for :math:`\ell \in [0,N)`.
    
    4. Compute the :math:`(M,N)` matrix **dist** of pairwise distances between cluster centroids.
    5. Apply a threshold on **dist** to generate a boolean matrix ``keep = dist < threshold``.
    6. Define a KeOps reduction ``my_genred = Genred(..., axis = 0 or 1)``, as usual.
    7. Compute the block-sparse reduction through
       ``result = my_genred(x_i, y_j, ranges = from_matrix(ranges_i,ranges_j,keep) )``

    :func:`from_matrix` is thus the routine that turns a **high-level description**
    of your block-sparse computation (cluster ranges + boolean matrix)
    into a set of **integer tensors** (the **ranges** optional argument), 
    used by KeOps to schedule computations on the GPU.

    Args:
        ranges_i ((M,2) IntTensor): List of :math:`[\text{start}_k,\text{end}_k)` indices.
            For :math:`k \in [0,M)`, the :math:`k`-th cluster of ":math:`i`" variables is
            given by ``x_i[ ranges_i[k,0]:ranges_i[k,1], : ]``, etc.
        ranges_j ((N,2) IntTensor): List of :math:`[\text{start}_l,\text{end}_l)` indices.
            For :math:`\ell \in [0,N)`, the :math:`\ell`-th cluster of ":math:`j`" variables is
            given by ``y_j[ ranges_j[l,0]:ranges_j[l,1], : ]``, etc.
        keep ((M,N) BoolTensor): 
            If the output ``ranges`` of :func:`from_matrix` is used in a KeOps reduction,
            we will only compute and reduce the terms associated to pairs of "points"
            :math:`x_i`, :math:`y_j` in clusters :math:`k` and :math:`\ell`
            if ``keep[k,l] == 1``.

    Returns:
        A 6-uple of LongTensors that can be used as an optional **ranges**
        argument of :func:`Genred <pykeops.torch.Genred>`. See the documentation of :func:`Genred <pykeops.torch.Genred>` for reference.

    Example:
        >>> r_i = torch.IntTensor( [ [2,5], [7,12] ] )          # 2 clusters: X[0] = x_i[2:5], X[1] = x_i[7:12]
        >>> r_j = torch.IntTensor( [ [1,4], [4,9], [20,30] ] )  # 3 clusters: Y[0] = y_j[1:4], Y[1] = y_j[4:9], Y[2] = y_j[20:30]
        >>> x,y = torch.Tensor([1., 0.]), torch.Tensor([1.5, .5, 2.5])  # dummy "centroids"
        >>> dist = (x[:,None] - y[None,:])**2
        >>> keep = (dist <= 1)                                  # (2,3) matrix
        >>> print(keep)
        tensor([[1, 1, 0],
                [0, 1, 0]], dtype=torch.uint8)
        --> X[0] interacts with Y[0] and Y[1], X[1] interacts with Y[1]
        >>> (ranges_i,slices_i,redranges_j, ranges_j,slices_j,redranges_i) = from_matrix(r_i,r_j,keep)
        --> (ranges_i,slices_i,redranges_j) will be used for reductions with respect to "j" (axis=1)
        --> (ranges_j,slices_j,redranges_i) will be used for reductions with respect to "i" (axis=0)

        Information relevant if **axis** = 1:

        >>> print(ranges_i)  # = r_i
        tensor([[ 2,  5],
                [ 7, 12]], dtype=torch.int32)
        --> Two "target" clusters in a reduction wrt. j
        >>> print(slices_i)  
        tensor([2, 3], dtype=torch.int32)
        --> X[0] is associated to redranges_j[0:2]
        --> X[1] is associated to redranges_j[2:3]
        >>> print(redranges_j)
        tensor([[1, 4],
                [4, 9],
                [4, 9]], dtype=torch.int32)
        --> For X[0], i in [2,3,4],       we'll reduce over j in [1,2,3] and [4,5,6,7,8]
        --> For X[1], i in [7,8,9,10,11], we'll reduce over j in [4,5,6,7,8]


        Information relevant if **axis** = 0:

        >>> print(ranges_j)
        tensor([[ 1,  4],
                [ 4,  9],
                [20, 30]], dtype=torch.int32)
        --> Three "target" clusters in a reduction wrt. i
        >>> print(slices_j)
        tensor([1, 3, 3], dtype=torch.int32)
        --> Y[0] is associated to redranges_i[0:1]
        --> Y[1] is associated to redranges_i[1:3]
        --> Y[2] is associated to redranges_i[3:3] = no one...
        >>> print(redranges_i)
        tensor([[ 2,  5],
                [ 2,  5],
                [ 7, 12]], dtype=torch.int32)
        --> For Y[0], j in [1,2,3],     we'll reduce over i in [2,3,4]
        --> For Y[1], j in [4,5,6,7,8], we'll reduce over i in [2,3,4] and [7,8,9,10,11]
        --> For Y[2], j in [20,21,...,29], there is no reduction to be done
    """
    I, J = torch.meshgrid( (torch.arange(0, keep.shape[0]), torch.arange(0,keep.shape[1])) )
    redranges_i = ranges_i[ I.t()[keep.t()] ]  # Use PyTorch indexing to "stack" copies of ranges_i[...]
    redranges_j = ranges_j[ J[keep] ]
    slices_i = keep.sum(1).cumsum(0).int()  # slice indices in the "stacked" array redranges_j
    slices_j = keep.sum(0).cumsum(0).int()  # slice indices in the "stacked" array redranges_i
    return (ranges_i, slices_i, redranges_j, ranges_j, slices_j, redranges_i)


if __name__=="__main__" :
    r_i = torch.IntTensor( [ [2,5], [7,12] ] )
    r_j = torch.IntTensor( [ [1,4], [4,9], [20,30] ] )
    x, y = torch.Tensor( [ 0., 1. ] ), torch.Tensor([ 0., .7, 2. ])
    dist = (x[:,None]-y[None,:])**2
    keep = (dist <= .8)
    print(keep)
    for item in from_matrix(r_i,r_j,keep) :
        print(item)