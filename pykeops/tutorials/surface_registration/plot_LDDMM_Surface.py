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
import os
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

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
def RalstonIntegrator():
    def f(ODESystem, x0, nt, deltat=1.0):
        x = tuple(map(lambda x: x.clone(), x0))
        dt = deltat / nt
        l = [x]
        for i in range(nt):
            xdot = ODESystem(*x)
            xi = tuple(map(lambda x, xdot: x + (2 * dt / 3) * xdot, x, xdot))
            xdoti = ODESystem(*xi)
            x = tuple(map(lambda x, xdot, xdoti: x + (.25 * dt) * (xdot + 3 * xdoti), x, xdot, xdoti))
            l.append(x)
        return l
    
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

def Shooting(p0, q0, K, nt=10, Integrator=RalstonIntegrator()):
    return Integrator(HamiltonianSystem(K), (p0, q0), nt)


def Flow(x0, p0, q0, K, deltat=1.0, Integrator=RalstonIntegrator()):
    HS = HamiltonianSystem(K)
    def FlowEq(x, p, q):
        return (K(x, q, p),) + HS(p, q)
    return Integrator(FlowEq, (x0, p0, q0), deltat)[0]


def LDDMMloss(K, dataloss, gamma=0):
    def loss(p0, q0):
        p,q = Shooting(p0, q0, K)[-1]
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
# Load the dataset

VS, FS, VT, FT = torch.load(datafile)
q0 = VS.clone().detach().to(dtype=torchdtype, device=torchdeviceId).requires_grad_(True)
VT = VT.clone().detach().to(dtype=torchdtype, device=torchdeviceId)
FS = FS.clone().detach().to(dtype=torch.long, device=torchdeviceId)
FT = FT.clone().detach().to(dtype=torch.long, device=torchdeviceId)
sigma = torch.tensor([20], dtype=torchdtype, device=torchdeviceId)

#####################################################################
# Define data attachment and LDDMM functional

dataloss = lossVarifoldSurf(FS, VT, FT, GaussLinKernel(sigma=sigma))
Kv = GaussKernel(sigma=sigma)
loss = LDDMMloss(Kv, dataloss)

######################################################################
# Perform optimization

# initialize momentum vectors
p0 = torch.zeros(q0.shape, dtype=torchdtype, device=torchdeviceId, requires_grad=True)

optimizer = torch.optim.LBFGS([p0], max_eval=6)
print('performing optimization...')
start = time.time()

def closure():
    optimizer.zero_grad()
    L = loss(p0, q0)
    print('loss', L.detach().cpu().numpy())
    L.backward()
    return L

for i in range(10):
    print('it ', i, ': ', end='')
    optimizer.step(closure)
    
print('Optimization (L-BFGS) time: ', round(time.time() - start, 2), ' seconds')

####################################################################
# Display output
# --------------
# The animated version of the deformation:

listpq = Shooting(p0, q0, Kv, nt=15)

################################################################################
# .. raw:: html
#
#     <img class='sphx-glr-single-img' src='../../_images/surface_matching.gif'/>
#


################################################################################
# .. raw:: html
#
#     <iframe src="../../_images/temp-plot.html" height="600px" width="100%"></iframe>
#


####################################################################
# The code to generate the .gif:

import imageio
import io
from PIL import Image

VTnp, FTnp = VT.detach().cpu().numpy(), FT.detach().cpu().numpy()
q0np, FSnp = q0.detach().cpu().numpy(), FS.detach().cpu().numpy()

images = []
for t in range(15):
    fig = plt.figure()
    qnp = listpq[t][1].detach().cpu().numpy()
    ax = Axes3D(fig)
    ax.axis('equal')
    ax.plot_trisurf(q0np[:, 0], q0np[:, 1], q0np[:, 2], triangles=FSnp, color=(0, 0, 0, 0),  edgecolor=(1, 0, 0, .08), linewidth=1)
    ax.plot_trisurf(qnp[:, 0],  qnp[:, 1],  qnp[:, 2],  triangles=FSnp, color=(1, 1, 0, .5), edgecolor=(1, 1, 1, .3),  linewidth=1)
    ax.plot_trisurf(VTnp[:, 0], VTnp[:, 1], VTnp[:, 2], triangles=FTnp, color=(0, 0, 0, 0),  edgecolor=(0, 0, 1, .3),  linewidth=1)
    blue_proxy   = plt.Rectangle((0, 0), 1, 1, fc="r")
    red_proxy    = plt.Rectangle((0, 0), 1, 1, fc="y")
    yellow_proxy = plt.Rectangle((0, 0), 1, 1, fc="b")
    ax.legend([blue_proxy, red_proxy, yellow_proxy], ['source', 'deformed', 'target'])
    ax.set_title('LDDMM matching example, step ' + str(t))
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    images.append(np.array(Image.open(buf)))
    buf.close()

save_folder = '../../../doc/_build/html/_images/'
os.makedirs(save_folder, exist_ok=True)
imageio.mimsave(save_folder + 'surface_matching.gif', images, duration=.5)

####################################################################
# The code to generate the plotly figure

listpq = Shooting(p0, q0, Kv, nt=50)

import plotly
import plotly.plotly as py
import plotly.graph_objs as go
import numpy as np
fig = go.Figure()
fig.add_mesh3d(x=qnp[:,0], y=qnp[:,1], z=qnp[:,2],
                    i=FSnp[:,0], j=FSnp[:,1], k=FSnp[:,2])

frames = []
for i in range(15):
    qnp = listpq[i][1].detach().cpu().numpy()
    frames.append(go.Frame(data=[{'type': 'mesh3d', 'x':qnp[:,0], 'y':qnp[:,1], 'z':qnp[:,2]}]))

fig.frames = frames

fig.layout.updatemenus = [
    {
        'buttons': [
            {
                'args': [None, {'frame': {'duration': 500, 'redraw': False},
                         'fromcurrent': True, 'transition': {'duration': 300}}],
                'label': 'Play',
                'method': 'animate'
            },
            {
                'args': [[None], {'frame': {'duration': 0, 'redraw': False}, 'mode': 'immediate',
                'transition': {'duration': 0}}],
                'label': 'Pause',
                'method': 'animate'
            }
        ],
        'direction': 'left',
        'pad': {'r': 10, 't': 87},
        'showactive': False,
        'type': 'buttons',
        'x': 0.55,
        'xanchor': 'right',
        'y': 0,
        'yanchor': 'top'
    }
]

plotly.offline.plot(fig)

