
import torch

def _squared_distances(x, y) :
    x_i = x.unsqueeze(1)         # Shape (N,D) -> Shape (N,1,D)
    y_j = y.unsqueeze(0)         # Shape (M,D) -> Shape (1,M,D)
    return ((x_i-y_j)**2).sum(2) # N-by-M matrix, xmy[i,j] = |x_i-y_j|^2

def _log_sum_exp(mat, dim):
    """
    Computes the log-sum-exp of a matrix with a numerically stable scheme, 
    in the user-defined summation dimension: exp is never applied
    to a number >= 0, and in each summation row, there is at least
    one "exp(0)" to stabilize the sum.
    
    For instance, if dim = 1 and mat is a 2d array, we output
                log( sum_j exp( mat[i,j] )) 
    by factoring out the row-wise maximas.
    """
    max_rc = torch.max(mat, dim)[0]
    return max_rc + torch.log(torch.sum(torch.exp(mat - max_rc.unsqueeze(dim)), dim))
