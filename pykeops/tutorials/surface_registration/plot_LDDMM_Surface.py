"""
====================
Surface registration
====================

Example of a diffeomorphic matching of surfaces using varifolds metrics:
We perform an LDDMM matching of two meshes using the geodesic shooting algorithm.

Note that this minimimalistic tutorial is intended to showcase,
within a single python script, the KeOps syntax in a complex use-case.

Going further, you may thus be interested in the cleaner and modular
`"shape" toolbox <https://plmlab.math.cnrs.fr/jeanfeydy/shapes_toolbox>`_.
"""

####################################################################
# Define our dataset
# ------------------
#
# Standard imports

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import torch
from torch.autograd import grad

import time

from pykeops.torch import Kernel, kernel_product
from pykeops.torch.kernel_product.formula import *


# torch type and device
use_cuda = torch.cuda.is_available()
torchdeviceId = torch.device('cuda:0') if use_cuda else 'cpu'
torchdtype = torch.float32

# PyKeOps counterpart
KeOpsdeviceId = torchdeviceId.index  # id of Gpu device (in case Gpu is  used)
KeOpsdtype = torchdtype.__str__().split('.')[1]  # 'float32'


####################################################################
# Import data file, one of :
#
# *  "hippos.pt" : original data (6611 vertices),
# *  "hippos_red.pt" : reduced size (1654 vertices),
# *  "hippos_reduc.pt" : further reduced (662 vertices),
# *  "hippos_reduc_reduc.pt" : further reduced (68 vertices)


if use_cuda:
    datafile = 'data/hippos.pt'
else:
    datafile = 'data/hippos_reduc_reduc.pt'

##################################################################
# Define the kernels
# ------------------
#
# Define Gaussian kernel :math:`(K(x,y)b)_i = \sum_j \exp(-\|x_i-y_j\|^2)b_j`

def GaussKernel(sigma):
    def K(x, y, b):
        params = {
            'id': Kernel('gaussian(x,y)'),
            'gamma': 1 / (sigma * sigma),
            'backend': 'auto'
        }
        return kernel_product(params, x, y, b)
    return K


###################################################################
# Define "Gaussian-CauchyBinet" kernel :math:`(K(x,y,u,v)b)_i = \sum_j \exp(-\|x_i-y_j\|^2) \langle u_i,v_j\rangle^2 b_j`

def GaussLinKernel(sigma):
    def K(x, y, u, v, b):
        params = {
            'id': Kernel('gaussian(x,y) * linear(u,v)**2'),
            'gamma': (1 / (sigma * sigma), None),
            'backend': 'auto'
        }
        return kernel_product(params, (x, u), (y, v), b)
    return K


####################################################################
# Custom ODE solver, for ODE systems which are defined on tuples
def RalstonIntegrator(nt=10):
    def f(ODESystem, x0, deltat=1.0):
        x = tuple(map(lambda x: x.clone(), x0))
        dt = deltat / nt
        for i in range(nt):
            xdot = ODESystem(*x)
            xi = tuple(map(lambda x, xdot: x + (2 * dt / 3) * xdot, x, xdot))
            xdoti = ODESystem(*xi)
            x = tuple(map(lambda x, xdot, xdoti: x + (.25 * dt) * (xdot + 3 * xdoti), x, xdot, xdoti))
        return x
    
    return f

####################################################################
# LDDMM implementation
# --------------------

#####################################################################
# Deformations: diffeomorphism 
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^

#####################################################################
# Hamiltonian system

def Hamiltonian(K):
    def H(p, q):
        return .5 * (p * K(q, q, p)).sum()
    return H


def HamiltonianSystem(K):
    H = Hamiltonian(K)
    def HS(p, q):
        Gp, Gq = grad(H(p, q), (p, q), create_graph=True)
        return -Gq, Gp
    return HS


#####################################################################
# Shooting approach

def Shooting(p0, q0, K, deltat=1.0, Integrator=RalstonIntegrator()):
    return Integrator(HamiltonianSystem(K), (p0, q0), deltat)


def Flow(x0, p0, q0, K, deltat=1.0, Integrator=RalstonIntegrator()):
    HS = HamiltonianSystem(K)
    def FlowEq(x, p, q):
        return (K(x, q, p),) + HS(p, q)
    return Integrator(FlowEq, (x0, p0, q0), deltat)[0]


def LDDMMloss(K, dataloss, gamma=0):
    def loss(p0, q0):
        p,q = Shooting(p0, q0, K)
        return gamma * Hamiltonian(K)(p0, q0) + dataloss(q)
    return loss

####################################################################
# Data attachment term
# ^^^^^^^^^^^^^^^^^^^^

#####################################################################
# Varifold data attachment loss for surfaces

# VT: vertices coordinates of target surface, 
# FS,FT : Face connectivity of source and target surfaces
# K kernel
def lossVarifoldSurf(FS, VT, FT, K):
    def CompCLNn(F, V):
        V0, V1, V2 = V.index_select(0, F[:, 0]), V.index_select(0, F[:, 1]), V.index_select(0, F[:, 2])
        C, N = .5 * (V0 + V1 + V2), .5 * torch.cross(V1 - V0, V2 - V0)
        L = (N ** 2).sum(dim=1)[:, None].sqrt()
        return C, L, N / L
    
    CT, LT, NTn = CompCLNn(FT, VT)
    cst = (LT * K(CT, CT, NTn, NTn, LT)).sum()
    
    def loss(VS):
        CS, LS, NSn = CompCLNn(FS, VS)
        return cst + (LS * K(CS, CS, NSn, NSn, LS)).sum() - 2 * (LS * K(CS, CT, NSn, NTn, LT)).sum()

    return loss

####################################################################
# Registration
# ------------

####################################################################
# load dataset

VS, FS, VT, FT = torch.load(datafile)
q0 = torch.tensor(VS, dtype=torchdtype, device=torchdeviceId, requires_grad=True)
VT = torch.tensor(VT, dtype=torchdtype, device=torchdeviceId)
FS = torch.tensor(FS, dtype=torch.long, device=torchdeviceId)
FT = torch.tensor(FT, dtype=torch.long, device=torchdeviceId)

sigma = torch.tensor([20], dtype=torchdtype, device=torchdeviceId)

#####################################################################
# define data attachment and LDDMM functional

dataloss = lossVarifoldSurf(FS, VT, FT, GaussLinKernel(sigma=sigma))
Kv = GaussKernel(sigma=sigma)
loss = LDDMMloss(Kv, dataloss)

######################################################################
# perform optimization

# initialize momentum vectors
p0 = torch.zeros(q0.shape, dtype=torchdtype, device=torchdeviceId, requires_grad=True)

optimizer = torch.optim.LBFGS([p0])
print('performing optimization...')
start = time.time()

def closure():
    optimizer.zero_grad()
    L = loss(p0, q0)
    L.backward()
    return L


optimizer.step(closure)
print('Optimization time (one L-BFGS step): ', round(time.time() - start, 2), ' seconds')

####################################################################
# display output

fig = plt.figure()
plt.title('LDDMM matching example')
p, q = Shooting(p0, q0, Kv)

q0np, qnp, FSnp = q0.detach().cpu().numpy(), q.detach().cpu().numpy(), FS.detach().cpu().numpy()
VTnp, FTnp = VT.detach().cpu().numpy(), FT.detach().cpu().numpy()
ax = Axes3D(fig)
ax.plot_trisurf(q0np[:, 0], q0np[:, 1], q0np[:, 2], triangles=FSnp, color=(1, 0, 0, .5), edgecolor=(1, 1, 1, .3), linewidth=1)
ax.plot_trisurf(qnp[:, 0], qnp[:, 1], qnp[:, 2], triangles=FSnp, color=(1, 1, 0, .5), edgecolor=(1, 1, 1, .3), linewidth=1)
ax.plot_trisurf(VTnp[:, 0], VTnp[:, 1], VTnp[:, 2], triangles=FTnp, color=(0, 0, 0, 0), edgecolor=(0, 0, 1, .3), linewidth=1)
ax.axis('equal')

blue_proxy = plt.Rectangle((0, 0), 1, 1, fc="r")
red_proxy = plt.Rectangle((0, 0), 1, 1, fc="y")
yellow_proxy = plt.Rectangle((0, 0), 1, 1, fc="b")
ax.legend([blue_proxy, red_proxy, yellow_proxy],['source', 'deformed', 'target'])
plt.show()