import torch

def grid_cluster( x, size ) :
    """Simplistic clustering algorithm which distributes points into cubic bins.

    Args:
        x ((M,D) Tensor): List of points :math:`x_i \in \mathbb{R}^D`.
        size (float or (D,) Tensor): Dimensions of the cubic cells ("voxels").

    Returns:
        (M,) IntTensor:

        Vector of integer **labels**. Two points ``x[i]`` and ``x[j]`` are 
        in the same cluster if and only if ``labels[i] == labels[j]``.
        Labels are sorted in a compact range :math:`[0,C)`,
        where :math:`C` is the number of non-empty cubic cells.

    Example:
        >>> x = torch.Tensor([ [0.], [.1], [.9], [.05], [.5] ])  # points in the unit interval
        >>> labels = grid_cluster(x, .2)  # bins of size .2
        >>> print( labels )
        tensor([0, 0, 2, 0, 1], dtype=torch.int32)

    """
    with torch.no_grad() :
        # Quantize the points' positions
        if   x.shape[1]==1 : weights = torch.IntTensor( [ 1 ] ,            ).to(x.device)
        elif x.shape[1]==2 : weights = torch.IntTensor( [ 2**10, 1] ,      ).to(x.device)
        elif x.shape[1]==3 : weights = torch.IntTensor( [ 2**20, 2**10, 1] ).to(x.device)
        else : raise NotImplementedError()
        x_  = ( x / size ).floor().int()
        x_ *= weights
        lab = x_.sum(1).abs() # labels

        # Replace arbitrary labels with unique identifiers in a compact arange
        u_lab = torch.unique(lab).sort()[0]
        N_lab = len(u_lab)
        foo  = torch.empty(u_lab.max()+1, dtype=torch.int32, device=x.device)
        foo[u_lab] = torch.arange(N_lab,  dtype=torch.int32, device=x.device)
        lab  = foo[lab]

    return lab
