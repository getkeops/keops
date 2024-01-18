import pykeops
import numpy as np
from pykeops.numpy import Genred

# pykeops.clean_pykeops()          # just in case old build files are still present
np.random.seed(0)

N = 64
M = 100
shape = (N, N, N)

use_fast_math = False

# Generate non uniform points in [-0.5, 0.5]
samples = np.random.rand(M, len(shape))
samples /= samples.max()
samples -= 0.5

samples *= 2 * np.pi  # Scaling to match exp(-i*(2*pi*k)*x)

# Generate Uniform points (image voxel grid)
# final shape is (1,N*N*N)
locs = np.ascontiguousarray(
    np.array(np.meshgrid(*[np.arange(s) for s in shape], indexing="ij"))
    .reshape(len(shape), -1)
    .T.astype(np.float32)
)

# Create Fake Image data
image = np.random.randn(*shape) + 1j * np.random.randn(*shape)
image = image.astype(np.complex64)

# Create Fake Coeffs data (in k-space)
kspace = np.random.randn(M) + 1j * np.random.randn(M)
kspace = kspace.astype(np.complex64)

# Initialize pykeops kernels

variables = ["x_j = Vj(1,{dim})", "nu_i = Vi(0,{dim})", "b_j = Vj(2,2)"]
aliases = [s.format(dim=len(shape)) for s in variables]
forward_op = Genred(
    "ComplexMult(ComplexExp1j(- nu_i | x_j),  b_j)",
    aliases,
    reduction_op="Sum",
    axis=1,
    use_fast_math=use_fast_math,
)

samples = samples.astype(np.float32)
image = image.flatten().view("(2,)float32")

# Calls Forward
coeff_cpu = forward_op(samples, locs, image, backend="CPU").view(np.complex64)

coeff_gpu = forward_op(samples, locs, image, backend="GPU").view(np.complex64)

print(np.linalg.norm(coeff_gpu - coeff_cpu))
print(np.linalg.norm(coeff_gpu - coeff_cpu) / np.linalg.norm(coeff_cpu))


# assert np.allclose(coeff_gpu, coeff_cpu)
