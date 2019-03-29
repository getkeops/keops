import torch

def sort_clusters(x, lab) :
    """Sorts a list of points and labels to make sure that the clusters are contiguous in memory.

    On the GPU, **contiguous memory accesses** are key to high performances.
    By making sure that points in the same cluster are stored next
    to each other in memory, this pre-processing routine allows
    KeOps to compute block-sparse reductions with maximum efficiency.

    Warning:
        For unknown reasons, ``torch.bincount`` is much more efficient
        on *unsorted* arrays of labels... so make sure not to call ``bincount``
        on the output of this routine!

    Args:
        x ((M,D) Tensor or tuple/list of (M,..) Tensors): List of points :math:`x_i \in \mathbb{R}^D`.
        lab ((M,) IntTensor): Vector of class labels :math:`\ell_i\in\mathbb{N}`.

    Returns:
        (M,D) Tensor or tuple/list of (M,..) Tensors, (M,) IntTensor:
        
        Sorted **point cloud(s)** and **vector of labels**.

    Example:
        >>> x   = torch.Tensor(   [ [0.], [5.], [.4], [.3], [2.] ])
        >>> lab = torch.IntTensor([  0,    2,    0,    0,    1   ])
        >>> x_sorted, lab_sorted = sort_clusters(x, lab)
        >>> print(x_sorted)
        tensor([[0.0000],
                [0.4000],
                [0.3000],
                [2.0000],
                [5.0000]])
        >>> print(lab_sorted)
        tensor([0, 0, 0, 1, 2], dtype=torch.int32)
    """
    lab, perm = torch.sort(lab.view(-1))
    if type(x) is tuple:
        x_sorted = tuple( a[perm] for a in x )
    elif type(x) is list:
        x_sorted =  list( a[perm] for a in x )
    else:
        x_sorted = x[perm]

    return x_sorted, lab

def cluster_ranges(lab, Nlab=None) :
    """Computes the ``[start,end)`` indices that specify clusters in a sorted point cloud.

    If **lab** denotes a vector of labels :math:`\ell_i\in[0,C)`,
    :func:`sort_clusters` allows us to sort our point clouds and make sure
    that points that share the same label are stored next to each other in memory.
    :func:`cluster_ranges` is simply there to give you the **slice indices**
    that correspond to each of those :math:`C` classes.

    Args:
        x ((M,D) Tensor): List of points :math:`x_i \in \mathbb{R}^D`.
        lab ((M,) IntTensor): Vector of class labels :math:`\ell_i\in\mathbb{N}`.

    Keyword Args:
        Nlab ((C,) IntTensor, optional): If you have computed it already,
            you may specify the number of points per class through
            this integer vector of length :math:`C`.

    Returns:
        (C,2) IntTensor:
        
        Stacked array of :math:`[\\text{start}_k,\\text{end}_k)` indices in :math:`[0,M]`,
        for :math:`k\in[0,C)`.

    Example:
        >>> x   = torch.Tensor(   [ [0.], [5.], [.4], [.3], [2.] ])
        >>> lab = torch.IntTensor([  0,    2,    0,    0,    1   ])
        >>> x_sorted, lab_sorted = sort_clusters(x, lab)
        >>> print(x_sorted)
        tensor([[0.0000],
                [0.4000],
                [0.3000],
                [2.0000],
                [5.0000]])
        >>> print(lab_sorted)
        tensor([0, 0, 0, 1, 2], dtype=torch.int32)
        >>> ranges_i = cluster_ranges(lab)
        >>> print( ranges_i )
        tensor([[0, 3],
                [3, 4],
                [4, 5]], dtype=torch.int32)
        --> cluster 0 = x_sorted[0:3, :]
        --> cluster 1 = x_sorted[3:4, :]
        --> cluster 2 = x_sorted[4:5, :]
    """
    if Nlab is None : Nlab = torch.bincount(lab).float()
    pivots = torch.cat( (torch.Tensor([0.]).to(Nlab.device), Nlab.cumsum(0)) )
    return torch.stack( (pivots[:-1], pivots[1:]) ).t().int()

def cluster_centroids(x, lab, Nlab=None, weights=None, weights_c=None) :
    """Computes the (weighted) centroids of classes specified by a vector of labels.
    
    If points :math:`x_i \in\mathbb{R}^D` are assigned to :math:`C` different classes
    by the vector of integer labels :math:`\ell_i \in [0,C)`,
    this function returns a collection of :math:`C` centroids

    .. math::
        c_k ~=~ \\frac{\sum_{i, \ell_i = k} w_i\cdot x_i}{\sum_{i, \ell_i=k} w_i},
    
    where the weights :math:`w_i` are set to 1 by default.

    Args:
        x ((M,D) Tensor): List of points :math:`x_i \in \mathbb{R}^D`.
        lab ((M,) IntTensor): Vector of class labels :math:`\ell_i\in\mathbb{N}`.

    Keyword Args:
        Nlab ((C,) IntTensor): Number of points per class. Recomputed if None.
        weights ((N,) Tensor): Positive weights :math:`w_i` of each point.
        weights_c ((C,) Tensor): Total weight of each class. Recomputed if None.

    Returns:
        (C,D) Tensor:
        
        List of centroids :math:`c_k \in \mathbb{R}^D`.

    Example:
        >>> x = torch.Tensor([     [0.], [1.], [4.], [5.], [6.] ])
        >>> lab = torch.IntTensor([ 0,    0,    1,    1,    1   ])
        >>> weights = torch.Tensor([ .5,  .5,   2.,   1.,   1.  ])
        >>> centroids = cluster_centroids(x, lab, weights=weights)
        >>> print(centroids)
        tensor([[0.5000],
                [4.7500]])
    """
    if Nlab is None : Nlab = torch.bincount(lab).float()
    if weights is not None and weights_c is None:
        weights_c = torch.bincount(lab, weights=weights).view(-1,1)

    c = torch.zeros( (len(Nlab), x.shape[1]), dtype=x.dtype,device=x.device)
    for d in range(x.shape[1]):
        if weights is None:
            c[:,d] = torch.bincount(lab,weights=x[:,d]) / Nlab
        else :
            c[:,d] = torch.bincount(lab,weights=x[:,d]*weights.view(-1)) / weights_c.view(-1)
    return c

def cluster_ranges_centroids(x, lab, weights=None) :
    """Computes the cluster indices and centroids of a (weighted) point cloud with labels.
    
    If **x** and **lab** encode a cloud of points :math:`x_i\in\mathbb{R}^D`
    with labels :math:`\ell_i\in[0,C)`, for :math:`i\in[0,M)`, this routine returns:
      
    - Ranges :math:`[\\text{start}_k,\\text{end}_k)` compatible with
      :func:`sort_clusters` for :math:`k\in[0,C)`.
    - Centroids :math:`c_k` for each cluster :math:`k`, computed as barycenters
      using the weights :math:`w_i \in \mathbb{R}_{>0}`:

    .. math::
        c_k = \\frac{\sum_{i, \ell_i=k} w_i\cdot \ell_i}{\sum_{i, \ell_i=k} w_i}

    - Total weights :math:`\sum_{i, \ell_i=k} w_i`, for :math:`k\in[0,C)`.

    The weights :math:`w_i` can be given through a vector **weights**
    of size :math:`M`, and are set by default to 1 for all points in the cloud.

    Args:
        x ((M,D) Tensor): List of points :math:`x_i \in \mathbb{R}^D`.
        lab ((M,) IntTensor): Vector of class labels :math:`\ell_i\in\mathbb{N}`.

    Keyword Args:
        weights ((M,) Tensor): Positive weights :math:`w_i` that can be used to compute
            our barycenters.

    Returns:
        (C,2) IntTensor, (C,D) Tensor, (C,) Tensor:
        
        **ranges** - Stacked array of :math:`[\\text{start}_k,\\text{end}_k)` indices in :math:`[0,M]`,
        for :math:`k\in[0,C)`, compatible with the :func:`sort_clusters` routine.

        **centroids** - List of centroids :math:`c_k \in \mathbb{R}^D`.

        **weights_c** - Total weight of each cluster.

    Example:
        >>> x   = torch.Tensor([  [0.], [.5], [1.], [2.], [3.] ])
        >>> lab = torch.IntTensor([ 0,    0,   1,    1,    1   ])
        >>> ranges, centroids, weights_c = cluster_ranges_centroids(x, lab)
        >>> print(ranges)
        tensor([[0, 2],
                [2, 5]], dtype=torch.int32)
        --> cluster 0 = x[0:2, :]
        --> cluster 1 = x[2:5, :]
        >>> print(centroids)
        tensor([[0.2500],
                [2.0000]])
        >>> print(weights_c)
        tensor([2., 3.])

        >>> weights = torch.Tensor([ 1.,  .5,  1.,  1.,  10. ])
        >>> ranges, centroids, weights_c = cluster_ranges_centroids(x, lab, weights=weights)
        >>> print(ranges)
        tensor([[0, 2],
                [2, 5]], dtype=torch.int32)
        --> cluster 0 = x[0:2, :]
        --> cluster 1 = x[2:5, :]
        >>> print(centroids)
        tensor([[0.1667],
                [2.7500]])
        >>> print(weights_c)
        tensor([1.5000, 12.0000])
    """
    Nlab = torch.bincount(lab).float()
    if weights is not None :
        w_c = torch.bincount(lab, weights=weights).view(-1)
        return cluster_ranges(lab, Nlab), cluster_centroids(x, lab, Nlab, weights=weights, weights_c=w_c), w_c
    else :
        return cluster_ranges(lab, Nlab), cluster_centroids(x, lab, Nlab), Nlab

def swap_axes(ranges) :
    """Swaps the ":math:`i`" and ":math:`j`" axes of a reduction's optional **ranges** parameter.
    
    This function returns **None** if **ranges** is **None**,
    and swaps the :math:`i` and :math:`j` arrays of indices otherwise."""
    if ranges is None :
        return None
    else :
        return (*ranges[3:6],*ranges[0:3])