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
    if Nlab is None : Nlab = torch.bincount(lab).float()
    pivots = torch.cat( (torch.Tensor([0.]).to(Nlab.device), Nlab.cumsum(0)) )
    return torch.stack( (pivots[:-1], pivots[1:]) ).t().int()

def cluster_centroids(x, lab, Nlab=None) :
    if Nlab is None : Nlab = torch.bincount(lab).float()

    c = torch.zeros( (len(Nlab), x.shape[1]), dtype=x.dtype,device=x.device)
    for d in range(x.shape[1]):
        c[:,d] = torch.bincount(lab,weights=x[:,d]) / Nlab
    return c

def cluster_ranges_centroids(x, lab) :
    Nlab = torch.bincount(lab).float()
    return cluster_ranges(lab, Nlab), cluster_centroids(x, lab, Nlab)