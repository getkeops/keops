import numpy as np


def grid_cluster(x, size):
    r"""Simplistic clustering algorithm which distributes points into cubic bins.

    Args:
        x ((M,D) array): List of points :math:`x_i \in \mathbb{R}^D`.
        size (float or (D,) array): Dimensions of the cubic cells ("voxels").

    Returns:
        (M,) integer array:

        Vector of integer **labels**. Two points ``x[i]`` and ``x[j]`` are
        in the same cluster if and only if ``labels[i] == labels[j]``.
        Labels are sorted in a compact range :math:`[0,C)`,
        where :math:`C` is the number of non-empty cubic cells.

    Example:
        >>> x = np.array([ [0.], [.1], [.9], [.05], [.5] ])  # points in the unit interval
        >>> labels = grid_cluster(x, .2)  # bins of size .2
        >>> print( labels )
        [0, 0, 2, 0, 1]

    """
    # Quantize the points' positions:
    if x.shape[1] == 1:
        weights = np.array([1], dtype=int)
    elif x.shape[1] == 2:
        weights = np.array([2**10, 1], dtype=int)
    elif x.shape[1] == 3:
        weights = np.array([2**20, 2**10, 1], dtype=int)
    else:
        raise NotImplementedError()
    x_ = np.floor(x / size).astype(int)
    x_ *= weights
    lab = np.sum(x_, axis=1)  # labels
    lab = lab - np.min(lab)

    # Replace arbitrary labels with unique identifiers in a compact arange:
    u_lab = np.sort(np.unique(lab))
    N_lab = len(u_lab)
    foo = np.empty(np.max(u_lab) + 1, dtype=int)
    foo[u_lab] = np.arange(N_lab, dtype=int)
    lab = foo[lab]

    return lab
