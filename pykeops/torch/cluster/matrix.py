import torch

def from_matrix( ranges_i, ranges_j, keep ) :
    I, J = torch.meshgrid( (torch.arange(0, keep.shape[0]), torch.arange(0,keep.shape[1])) )
    redranges_i = ranges_i[ I.t()[keep.t()] ]
    redranges_j = ranges_j[ J[keep] ]
    slices_i = keep.sum(1).cumsum(0)
    slices_j = keep.sum(0).cumsum(0)
    return (ranges_i, slices_i, redranges_j, ranges_j, slices_j, redranges_i)


if __name__=="__main__" :
    r_i = torch.LongTensor( [ [2,5], [7,12], [1,3], [8,10] ] )
    r_j = torch.LongTensor( [ [1,4], [5,10], [4,9], [1,11], [6,9], [20,30] ] )
    x = torch.arange(0.,4.)
    y = torch.arange(-2.,4.)
    C = (x[:,None]-y[None,:])**2
    B = (C <= 1)
    print(r_i)
    print(r_j)
    print(B)
    for item in from_matrix(r_i,r_j,B) :
        print(item)