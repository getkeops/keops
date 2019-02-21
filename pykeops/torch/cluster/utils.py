import torch

def sort_clusters(x, lab) :
    """Sorts a list of points with class labels to make clusters contiguous in memory.

    On the GPU, **contiguous memory accesses** are key to high performances.
    By making sure that points in the same cluster are stored next
    to each other in memory, this pre-processing routine allows
    KeOps to compute block-sparse reductions with maximum efficiency.

    Warning:
        For unknown reasons, ``torch.bincount`` is much more efficient
        on *unsorted* arrays of labels... so make sure not to call ``bincount``
        on the output of this routine!

    Args:
        x ((M,D) Tensor): List of points :math:`x_i \in \mathbb{R}^D`.
        lab ((M,) IntTensor): Vector of class labels :math:`l_i\in\mathbb{N}`.

    Returns:
        (M,D) Tensor, (M,) IntTensor:
        
        Sorted **point cloud** and **vector of labels**.

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
    return x[perm,:], lab

def cluster_ranges(lab, Nlab=None) :
    """Computes the ``[start,end)`` indices that specify clusters in a sorted point cloud.

    If ``lab`` denotes a vector of labels :math:`l_i\in[0,C)`,
    :func:`sort_clusters` allows us to sort our point clouds and make sure
    that points that share the same label are stored next to each other in memory.
    :func:`cluster_ranges` is simply there to give you the **slice indices**
    that correspond to each of those :math:`C` classes.

    Args:
        x ((M,D) Tensor): List of points :math:`x_i \in \mathbb{R}^D`.
        lab ((M,) IntTensor): Vector of class labels :math:`l_i\in\mathbb{N}`.

    Keyword Args:
        Nlab ((C,) IntTensor, optional): If you have computed it already,
            you may specify the number of points per class through
            this integer vector of length :math:`C`.

    Returns:
        (C,2) IntTensor:
        
        Stacked array of :math:`[start_k,end_k)` indices in :math:`[0,M]`,
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

def cluster_centroids(x, lab, Nlab=None, w=None, w_c=None) :
    """Computes the (weighted) centroids of classes specified by a vector of labels.
    
    If points :math:`x_i \in\mathbb{R}^D` are assigned to :math:`C` different classes
    by the vector of integer labels :math:`l_i \in [0,C)`,
    this function returns a collection of :math:`C` centroids

    .. math::
        c_k ~=~ \tfrac{1}{W_k}\sum_{l_i = k} w_i\cdot x_i,
    
    where:


    Args:
        x ((M,D) Tensor): List of points :math:`x_i \in \mathbb{R}^D`.
        lab ((M,) IntTensor): Vector of class labels :math:`l_i\in\mathbb{N}`.

    Keyword Args:
        Nlab ((C,) Tensor): 
        size (float or (D,) Tensor): Dimensions of the cubic cells ("voxels").

    Returns:
        (C,D) Tensor:
        
        List of centroids :math:`c_k \in \mathbb{R}^D`.
    """
    if Nlab is None : Nlab = torch.bincount(lab).float()

    c = torch.zeros( (len(Nlab), x.shape[1]), dtype=x.dtype,device=x.device)
    for d in range(x.shape[1]):
        if w is None : c[:,d] = torch.bincount(lab,weights=x[:,d]) / Nlab
        else :         c[:,d] = torch.bincount(lab,weights=x[:,d]*w.view(-1)) / w_c.view(-1)
    return c

def cluster_ranges_centroids(x, lab, weights=None) :
    """Computes the cluster indices and centroids of a (weighted) point cloud with class labels.
    
    If 


    Args:
        x ((M,D) Tensor): List of points :math:`x_i \in \mathbb{R}^D`.
        lab ((M,) IntTensor): Vector of class labels :math:`l_i\in\mathbb{N}`.

    Keyword Args:
        weights ((M,) Tensor): Positive weights that can be used to compute
            our barycenters.

    Returns:
        (C,2) IntTensor, (C,D) Tensor:
        
        **ranges** - Stacked array of :math:`[start_k,end_k)` indices in :math:`[0,M]`,
        for :math:`k\in[0,C)`.

        **centroids** - List of centroids :math:`c_k \in \mathbb{R}^D`.
    """
    Nlab = torch.bincount(lab).float()
    if weights is not None :
        w_c = torch.bincount(lab, weights=weights).view(-1,1)
        return cluster_ranges(lab, Nlab), cluster_centroids(x, lab, Nlab, w=weights, w_c=w_c), w_c
    else :
        return cluster_ranges(lab, Nlab), cluster_centroids(x, lab, Nlab)

def swap_axes(ranges) :
    """
    
    This function returns ``None`` if ``ranges`` is ``None``,
    and swaps the :math:`i` and :math:`j` arrays of indices otherwise."""
    if ranges is None :
        return None
    else :
        return (*ranges[3:6],*ranges[0:3])