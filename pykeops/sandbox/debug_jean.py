"""
Debug CUDA illegal memory errors
===================================

"""


##############################################
# Setup
# ---------------------

import torch
from geomloss import SamplesLoss
from time import time

# torch.manual_seed(0)
torch.manual_seed(1)

use_cuda = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

##############################################
# Sample points on the unit sphere:
#

N, M = (100, 100) if not use_cuda else (100000, 100000)
x, y = torch.randn(N, 3).type(dtype), torch.randn(M, 3).type(dtype)
x, y = x / (2 * x.norm(dim=1, keepdim=True)), y / (2 * y.norm(dim=1, keepdim=True))
x.requires_grad = True

##########################################################
# Use the PyTorch profiler to output Chrome trace files:

for loss in ["gaussian"]:#, "sinkhorn"]:
    for backend in ["multiscale"]:
        Loss = SamplesLoss(
            loss, blur=0.05, backend=backend, truncate=3, verbose=True
        )
        L_xy = Loss(x, y)
        #L_xy.backward()
        print("cost = {:.6f}".format(L_xy.item()))
