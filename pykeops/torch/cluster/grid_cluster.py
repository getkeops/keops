import torch

def grid_cluster( x, size ) :
    """
    Simplistic clustering algorithm, which distributes points
    into cubic bins of size L.
    """
    with torch.no_grad() :

        # Quantize the points' positions
        if   x.shape[1]==2 : weights = torch.LongTensor( [ 2**10, 1] ,      ).to(x.device)
        elif x.shape[1]==3 : weights = torch.LongTensor( [ 2**20, 2**10, 1] ).to(x.device)
        x_  = ( x / size ).floor().long()
        x_ *= weights
        lab = x_.sum(1).abs() # labels

        # Replace arbitrary labels with unique identifiers in a compact arange
        u_lab = torch.unique(lab)
        N_lab = len(u_lab)
        foo  = torch.empty(u_lab.max()+1, dtype=torch.long, device=x.device)
        foo[u_lab] = torch.arange(N_lab,  dtype=torch.long, device=x.device)
        lab  = foo[lab]

    return lab