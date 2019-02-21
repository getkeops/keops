import torch

def sort_clusters(x, lab) :
    """
    Beware! For an unknown reason, torch.bincount is much more efficient
    on *unsorted* arrays of labels, so make sure not to call bincount
    on the output of this routine!
    """
    lab, perm = torch.sort(lab)
    return x[perm,:], lab

def cluster_ranges(lab, Nlab=None) :
    """Blablabla"""
    if Nlab is None : Nlab = torch.bincount(lab).float()
    pivots = torch.cat( (torch.Tensor([0.]).to(Nlab.device), Nlab.cumsum(0)) )
    return torch.stack( (pivots[:-1], pivots[1:]) ).t().int()

def cluster_centroids(x, lab, Nlab=None, w=None, w_c=None) :
    """Blablabla"""
    if Nlab is None : Nlab = torch.bincount(lab).float()

    c = torch.zeros( (len(Nlab), x.shape[1]), dtype=x.dtype,device=x.device)
    for d in range(x.shape[1]):
        if w is None : c[:,d] = torch.bincount(lab,weights=x[:,d]) / Nlab
        else :         c[:,d] = torch.bincount(lab,weights=x[:,d]*w.view(-1)) / w_c.view(-1)
    return c

def cluster_ranges_centroids(x, lab, weights=None) :
    """Blablabla"""
    Nlab = torch.bincount(lab).float()
    if weights is not None :
        w_c = torch.bincount(lab, weights=weights).view(-1,1)
        return cluster_ranges(lab, Nlab), cluster_centroids(x, lab, Nlab, w=weights, w_c=w_c), w_c
    else :
        return cluster_ranges(lab, Nlab), cluster_centroids(x, lab, Nlab)

def swap_axes(ranges) :
    """Blablabla"""
    if ranges is None :
        return None
    else :
        return (*ranges[3:6],*ranges[0:3])