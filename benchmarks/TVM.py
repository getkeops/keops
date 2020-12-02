"""
TVM
===============
"""

##########################
# If you're running this script on Google Colab, use the following lines to install TVM on your session:


#%matplotlib inline
#
# try:
#  import google.colab
#  IN_COLAB = True
# except:
#  IN_COLAB = False
#
# if IN_COLAB:
#    ! gsutil cp "gs://tvm-fcrc-binaries-7f775516ff9dfab922c304049f294cec/tvm.tar.gz" /tmp/tvm.tar.gz
#    ! mkdir -p /tvm
#    ! tar -xf /tmp/tvm.tar.gz --strip-components=4 --directory /tvm
#    ! ls -la /tvm
#    ! bash /tvm/package.sh
#    # Add TVM to the Python path.
#    import sys
#    sys.path.append('/tvm/python')
#    sys.path.append('/tvm/topi/python')
#    sys.path.append('/tvm/nnvm/python')
#    sys.path.append('/tvm/vta/python')
# else:
#    print("Notebook executing locally, skipping Colab setup ...")


###################################################################
# Actual benchmark:


from __future__ import absolute_import, print_function

import tvm
import numpy as np
from time import time


# Global declarations of environment.
tgt_host = "llvm"
tgt = "cuda"
ctx = tvm.context(tgt, 0)

# Declare axis and reduction indices
n = tvm.var("n")
j = tvm.reduce_axis((0, n), "j")

# Declare Variable
A0 = tvm.placeholder((n,), name="A0", dtype="float32")
A1 = tvm.placeholder((n,), name="A1", dtype="float32")
A2 = tvm.placeholder((n,), name="A2", dtype="float32")

B0 = tvm.placeholder((n,), name="B0", dtype="float32")
B1 = tvm.placeholder((n,), name="B1", dtype="float32")
B2 = tvm.placeholder((n,), name="B2", dtype="float32")

D = tvm.placeholder((n,), name="D", dtype="float32")

D_ij = (
    lambda i: (A0[i] - B0[j]) * (B0[j] - A0[i])
    + (A1[i] - B1[j]) * (B1[j] - A1[i])
    + (A2[i] - B2[j]) * (B2[j] - A2[i])
)
K_ij = lambda i: tvm.call_pure_extern("float32", "__expf", D_ij(i))

C0 = tvm.compute((n,), lambda i: tvm.sum(K_ij(i) * D[j], axis=j), name="C0")

# Scheduled the computation
s0 = tvm.create_schedule(C0.op)
bx, tx = s0[C0].split(C0.op.axis[0], factor=192)
s0[C0].bind(bx, tvm.thread_axis("blockIdx.x"))
s0[C0].bind(tx, tvm.thread_axis("threadIdx.x"))

# Actually build the binary
fconv0 = tvm.build(
    s0, [A0, A1, A2, B0, B1, B2, D, C0], tgt, target_host=tgt_host, name="myconv0"
)

# Benchmark
nits = 10
Ns = [10000, 100000, 1000000]

for n in Ns:
    a_np = np.random.randn(n, 3).astype(A0.dtype)
    a0 = tvm.nd.array(a_np[:, 0], ctx)
    a1 = tvm.nd.array(a_np[:, 1], ctx)
    a2 = tvm.nd.array(a_np[:, 2], ctx)

    b_np = np.random.randn(n, 3).astype(B0.dtype)
    b0 = tvm.nd.array(b_np[:, 0], ctx)
    b1 = tvm.nd.array(b_np[:, 1], ctx)
    b2 = tvm.nd.array(b_np[:, 2], ctx)

    d_np = np.random.randn(
        n,
    ).astype(D.dtype)
    d = tvm.nd.array(d_np, ctx)

    c_np = np.random.randn(n, 3).astype(C0.dtype)
    c = tvm.nd.array(c_np[:, 0], ctx)

    start = time()

    for _ in range(nits):
        fconv0(a0, a1, a2, b0, b1, b2, d, c)  # Evaluations
    ctx.sync()

    end = time()

    print("Timing with {} points: {} x {:.4f}s".format(n, nits, (end - start) / nits))
