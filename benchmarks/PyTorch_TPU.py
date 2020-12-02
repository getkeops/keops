"""
PyTorch, on a TPU
============================
"""

################################################################
# This code should be run on a Google Colab session, with TPU acceleration.
#


import os

assert os.environ[
    "COLAB_TPU_ADDR"
], "Make sure to select TPU from Edit > Notebook settings > Hardware accelerator"

###################################################
#

DIST_BUCKET = "gs://tpu-pytorch/wheels"
TORCH_WHEEL = "torch-1.15-cp36-cp36m-linux_x86_64.whl"
TORCH_XLA_WHEEL = "torch_xla-1.15-cp36-cp36m-linux_x86_64.whl"
TORCHVISION_WHEEL = "torchvision-0.3.0-cp36-cp36m-linux_x86_64.whl"

# Install Colab TPU compat PyTorch/TPU wheels and dependencies
"""
!pip uninstall -y torch torchvision
!gsutil cp "$DIST_BUCKET/$TORCH_WHEEL" .
!gsutil cp "$DIST_BUCKET/$TORCH_XLA_WHEEL" .
!gsutil cp "$DIST_BUCKET/$TORCHVISION_WHEEL" .
!pip install "$TORCH_WHEEL"
!pip install "$TORCH_XLA_WHEEL"
!pip install "$TORCHVISION_WHEEL"
!sudo apt-get install libomp5
"""

###################################################
#


import torch
import torch_xla
import torch_xla.core.xla_model as xm

t = torch.randn(2, 2, device=xm.xla_device())
print(t.device)
print(t)


###################################################
#
# Run the cell below two times: once for the compilation, once for profiling!
#

nits = 100
Ns, D = [10000, 100000, 1000000], 3

for N in Ns:

    x = torch.randn(N, D, device=xm.xla_device())
    y = torch.randn(N, D, device=xm.xla_device())
    p = torch.randn(N, 1, device=xm.xla_device())

    def KP(x, y, p):
        D_xx = (x * x).sum(-1).unsqueeze(1)  # (N,1)
        D_xy = torch.matmul(x, y.permute(1, 0))  # (N,D) @ (D,M) = (N,M)
        D_yy = (y * y).sum(-1).unsqueeze(0)  # (1,M)
        D_xy = D_xx - 2 * D_xy + D_yy
        K_xy = (-D_xy).exp()

        return K_xy @ p

    import time

    start = time.time()

    for _ in range(nits):
        p = KP(x, y, p)

    print(p)
    end = time.time()
    print("Timing with {} points: {} x {:.4f}s".format(N, nits, (end - start) / nits))
