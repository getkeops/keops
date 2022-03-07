"""
Profile the GeomLoss routines
===================================

This example explains how to **profile** the geometric losses
to select the backend and truncation/scaling values that
are best suited to your data.
"""


##############################################
# Setup
# ---------------------

import torch
from geomloss import SamplesLoss
from time import time

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

for loss in ["sinkhorn"]:
    for backend in ["multiscale"]:
        Loss = SamplesLoss(
                loss, blur=0.05, backend=backend, truncate=3, verbose=True
            )
        t_0 = time()
        L_xy = Loss(x, y)
        #torch.cuda.synchronize()
        t_1 = time()
        print("{:.2f}s, cost = {:.6f}".format(t_1 - t_0, L_xy.item()))    


######################################################################
# Now, all you have to do is to open the "Easter egg" address
# ``chrome://tracing`` in Google Chrome/Chromium,
# and load the ``profile_*`` files one after
# another. Enjoy :-)

print("Done.")