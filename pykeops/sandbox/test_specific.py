import pykeops
import numpy as np
from pykeops.numpy.shape_distance import FshapeScp
from pykeops.numpy.utils import np_kernel, np_kernel_sphere

M, N, D, E = 100, 100, 3, 3
x = np.random.rand(M, D)
a = np.random.rand(M, E)
f = np.random.rand(M, 1)
y = np.random.rand(N, D)
b = np.random.rand(N, E)
g = np.random.rand(N, 1)
sigma_geom = 1.0
sigma_sig = 1.0
sigma_sphere = np.pi / 2
kgeom = "gaussian"
ksig = "gaussian"
ksphere = "gaussian_oriented"
myconv = FshapeScp(kernel_geom=kgeom, kernel_sig=ksig, kernel_sphere=ksphere)
gamma = myconv(
    x,
    y,
    f,
    g,
    a,
    b,
    sigma_geom=sigma_geom,
    sigma_sig=sigma_sig,
    sigma_sphere=sigma_sphere,
).ravel()

areaa = np.linalg.norm(a, axis=1)
areab = np.linalg.norm(b, axis=1)

nalpha = a / areaa[:, np.newaxis]
nbeta = b / areab[:, np.newaxis]

gamma_py = np.sum(
    (areaa[:, np.newaxis] * areab[np.newaxis, :])
    * np_kernel(x, y, sigma_geom, kgeom)
    * np_kernel(f, g, sigma_sig, ksig)
    * np_kernel_sphere(nalpha, nbeta, sigma_sphere, ksphere),
    axis=1,
)

# compare output
print(np.allclose(gamma, gamma_py, atol=1e-6))
