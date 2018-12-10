import torch

from pykeops import default_cuda_type
from pykeops.common.parse_type import get_type, get_sizes, complete_aliases
from pykeops.common.get_options import get_tag_backend
from pykeops.common.gridcluster_io import load_gridcluster

include_dirs = torch.utils.cpp_extension.include_paths()[0]

def grid_label(points, voxelsize, backend='auto', device_id=-1, cuda_type=default_cuda_type):

    gridcluster = load_gridcluster(points.shape[1], cuda_type, 'torch', ['-DPYTORCH_INCLUDE_DIR=' + include_dirs])

    args = (voxelsize, points) # order expected by the CPP code

    tagCPUGPU, tag1D2D, tagHostDevice = get_tag_backend(backend, args)

    if tagCPUGPU==1 & tagHostDevice==1:
        device_id = args[0].device.index
        for i in range(1,len(args)):
            if args[i].device.index != device_id:
                raise ValueError("[KeOps] Input arrays must be all located on the same device.")
    
    return gridcluster.grid_label( tagCPUGPU, tagHostDevice, device_id, *args )






def grid_cluster( x, size ) :
    """
    Simplistic clustering algorithm, which distributes points
    into cubic bins of size L.
    """
    with torch.no_grad() :
        if False :
            lab = grid_label(x, size)
        else :
            # Quantize the points' positions
            if   x.shape[1]==2 : weights = torch.IntTensor( [ 2**10, 1] ,      ).to(x.device)
            elif x.shape[1]==3 : weights = torch.IntTensor( [ 2**20, 2**10, 1] ).to(x.device)
            x_  = ( x / size ).floor().int()
            x_ *= weights
            lab = x_.sum(1).abs() # labels

        # Replace arbitrary labels with unique identifiers in a compact arange
        u_lab = torch.unique(lab)
        N_lab = len(u_lab)
        foo  = torch.empty(u_lab.max()+1, dtype=torch.int32, device=x.device)
        foo[u_lab] = torch.arange(N_lab,  dtype=torch.int32, device=x.device)
        lab  = foo[lab]

    return lab