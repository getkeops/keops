"""
====================
Surface registration
====================

Example of a diffeomorphic matching of surfaces using varifolds metrics:
We perform an LDDMM matching of two meshes using the geodesic shooting algorithm.


"""

####################################################################
# Define our dataset
# ------------------
#
# Standard imports

import os
import time

import imageio
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
from torch.autograd import grad

from pykeops.torch import Kernel, kernel_product, LazyTensor, Vi, Vj
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
# Define Gaussian kernel :math:`(K(x,y)b)_i = \sum_j \exp(-\gamma\|x_i-y_j\|^2)b_j`

def GaussKernel(sigma):
    x, y, b = Vi(0,3), Vj(1,3), Vj(2,3)
    gamma = 1 / (sigma * sigma)
    D2 = x.sqdist(y)
    K = (-D2*gamma).exp()
    return (K*b).sum_reduction(axis=1)

###################################################################
# Define "Gaussian-CauchyBinet" kernel :math:`(K(x,y,u,v)b)_i = \sum_j \exp(-\gamma\|x_i-y_j\|^2) \langle u_i,v_j\rangle^2 b_j`

def GaussLinKernel(sigma):
    x, y, u, v, b = Vi(0,3), Vj(1,3), Vi(2,3), Vj(3,3), Vj(4,1)
    gamma = 1 / (sigma * sigma)
    D2 = x.sqdist(y)
    K = (-D2*gamma).exp() * (u*v).sum()**2
    return (K*b).sum_reduction(axis=1)

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
    def get_center_length_normal(F, V):
        V0, V1, V2 = V.index_select(0, F[:, 0]), V.index_select(0, F[:, 1]), V.index_select(0, F[:, 2])
        centers, normals =  (V0 + V1 + V2) / 3, .5 * torch.cross(V1 - V0, V2 - V0)
        length = (normals ** 2).sum(dim=1)[:, None].sqrt()
        return centers, length, normals/ length
    
    CT, LT, NTn = get_center_length_normal(FT, VT)
    cst = (LT * K(CT, CT, NTn, NTn, LT)).sum()
    
    def loss(VS):
        CS, LS, NSn = get_center_length_normal(FS, VS)
        return cst + (LS * K(CS, CS, NSn, NSn, LS)).sum() - 2 * (LS * K(CS, CT, NSn, NTn, LT)).sum()

    return loss

####################################################################
# Registration
# ------------

####################################################################
# Load the dataset and plot it

VS, FS, VT, FT = torch.load(datafile)
q0 = VS.clone().detach().to(dtype=torchdtype, device=torchdeviceId).requires_grad_(True)
VT = VT.clone().detach().to(dtype=torchdtype, device=torchdeviceId)
FS = FS.clone().detach().to(dtype=torch.long, device=torchdeviceId)
FT = FT.clone().detach().to(dtype=torch.long, device=torchdeviceId)
sigma = torch.tensor([20], dtype=torchdtype, device=torchdeviceId)

fig = plt.figure()
ax = Axes3D(fig)
ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.plot_trisurf(q0.detach().cpu().numpy()[:, 0],
                q0.detach().cpu().numpy()[:, 1],
                q0.detach().cpu().numpy()[:, 2],
                triangles=FS.detach().cpu().numpy(),
                color=(0, 0, 0, 0),  edgecolor=(1, 0, 0, .08), linewidth=1)
ax.plot_trisurf(VT.detach().cpu().numpy()[:, 0],
                VT.detach().cpu().numpy()[:, 1],
                VT.detach().cpu().numpy()[:, 2],
                triangles=FT.detach().cpu().numpy(),
                color=(0, 0, 0, 0),  edgecolor=(0, 0, 1, .3),  linewidth=1)
blue_proxy = plt.Rectangle((0, 0), 1, 1, fc="b")
red_proxy = plt.Rectangle((0, 0), 1, 1, fc=(1, 0, 0, .5))
ax.legend([red_proxy,  blue_proxy], ['source', 'target'])
ax.set_title('Data')
plt.show()

#####################################################################
# Define data attachment and LDDMM functional

dataloss = lossVarifoldSurf(FS, VT, FT, GaussLinKernel(sigma=sigma))
Kv = GaussKernel(sigma=sigma)
loss = LDDMMloss(Kv, dataloss)

######################################################################
# Perform optimization

# initialize momentum vectors
p0 = torch.zeros(q0.shape, dtype=torchdtype, device=torchdeviceId, requires_grad=True)

optimizer = torch.optim.LBFGS([p0], max_eval=10, max_iter=10)
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
nt = 15
listpq = Shooting(p0, q0, Kv, nt=nt)

################################################################################
# .. raw:: html
#
#     <img class='sphx-glr-single-img' src='../../_images/surface_matching.gif'/>
#


####################################################################
# The code to generate the .gif:

VTnp, FTnp = VT.detach().cpu().numpy(), FT.detach().cpu().numpy()
q0np, FSnp = q0.detach().cpu().numpy(), FS.detach().cpu().numpy()


images = []
for t in range(nt):
    qnp = listpq[t][1].detach().cpu().numpy()
    
    # create Figure
    fig = Figure(figsize=(6, 5), dpi=100)
    # Link canvas to fig
    canvas = FigureCanvasAgg(fig)
    
    # make the plot
    ax = Axes3D(fig)
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.plot_trisurf(qnp[:, 0],  qnp[:, 1],  qnp[:, 2],  triangles=FSnp, color=(1, 1, 0, .5), edgecolor=(1, 1, 1, .3),  linewidth=1)
    ax.plot_trisurf(VTnp[:, 0], VTnp[:, 1], VTnp[:, 2], triangles=FTnp, color=(0, 0, 0, 0),  edgecolor=(0, 0, 1, .3),  linewidth=1)

    yellow_proxy = plt.Rectangle((0, 0), 1, 1, fc="y")
    ax.legend([yellow_proxy, blue_proxy], ['deformed', 'target'])
    ax.set_title('LDDMM matching example, step ' + str(t))
    
    # draw it!
    canvas.draw()
    
    # save plot in a numpy array through buffer
    s, (width, height) = canvas.print_to_buffer()
    images.append(np.frombuffer(s, np.uint8).reshape((height, width, 4)))

save_folder = '../../../doc/_build/html/_images/'
os.makedirs(save_folder, exist_ok=True)
imageio.mimsave(save_folder + 'surface_matching.gif', images, duration=.5)

