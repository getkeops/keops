#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 25 00:38:18 2018

@author: glaunes
"""

#%%

import os.path
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + (os.path.sep + '..')*2)

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import torch
from torch.autograd import Variable, grad

import time

from pykeops.torch.generic_sum       import GenericSum
from pykeops.torch.kernels import Kernel, kernel_product

def GaussKernelKeopsGeneric(sigma):
    def f(x,y,b):
        dp = x.shape[1]
        dv = b.shape[1]
        aliases = ["P = Pm(0,1)","X=Vx(1,"+str(dp)+")","Y=Vy(2,"+str(dp)+")","B=Vy(3,"+str(dv)+")"]
        formula = "Exp(-P*SqDist(X,Y))*B"
        signature   =   [ (dv, 0), (1, 2), (dp, 0), (dp, 1), (dv, 1) ]
        sum_index   = 0 # the result is indexed by "i"; for "j", use "1"
        oos2 = Variable(torch.FloatTensor([1/sigma**2]))
        return GenericSum.apply("auto",aliases,formula,signature,sum_index,oos2,x,y,b)
    return f

def GaussKernelKeops(sigma):
    def f(x,y,b):
        params = {
            "id"      : Kernel("gaussian(x,y)"),
            "gamma"   : Variable(torch.FloatTensor([1/sigma**2])),
            "backend" : "auto"
        }
        return kernel_product( x,y,b, params)
    return f

def GaussKernel(sigma):
    oos2 = 1/sigma**2
    def f(x,y,b):
        return torch.exp(-oos2*torch.sum((x[:,None,:]-y[None,:,:])**2,dim=2))@b
    return f

def GaussLinKernel(sigma):
    oos2 = 1/sigma**2
    def f(x,y,vx,vy,b):
        Kxy = torch.exp(-oos2*torch.sum((x[:,None,:]-y[None,:,:])**2,dim=2))
        Sxy = torch.sum(vx[:,None,:]*vy[None,:,:],dim=2)**2
        return (Kxy*Sxy)@b
    return f

def GaussLinKernelKeops(sigma):
    def f(x,y,vx,vy,b):
        params = {
            "id"      : Kernel("gaussian(x,y) * linear(u,v)**2"),
            "gamma"   : (Variable(torch.FloatTensor([1/sigma**2])),Variable(torch.FloatTensor([0]))),
            "backend" : "auto"
        }
        return kernel_product( (x,vx),(y,vy),b, params)
    return f
#
def GaussLinKernelKeopsGeneric(sigma):
    def f(x,y,vx,vy,b):
        dp = x.shape[1]
        df = vx.shape[1]
        dv = b.shape[1]
        aliases = ["P = Pm(0,1)",
                   "X=Vx(1,"+str(dp)+")","Y=Vy(2,"+str(dp)+")",
                   "U=Vx(3,"+str(df)+")","V=Vy(4,"+str(df)+")",
                   "B=Vy(5,"+str(dv)+")"]
        formula = "Exp(-P*SqDist(X,Y))*Square((U,V))*B"
        signature   =   [ (dv, 0), (1, 2), (dp, 0), (dp, 1), (df, 0), (df, 1), (dv, 1) ]
        sum_index   = 0 # the result is indexed by "i"; for "j", use "1"
        oos2 = Variable(torch.FloatTensor([1/sigma**2]))
        return GenericSum.apply("auto",aliases,formula,signature,sum_index,oos2,x,y,vx,vy,b)
    return f

def HeunsIntegrator(nt=10):
    def f(ODESystem,x0,deltat=1.0):
        x = tuple(map(lambda x:x.clone(),x0))
        dt = deltat/nt
        for i in range(nt):
            xdot = ODESystem(*x)
            xi = tuple(map(lambda x,xdot:x+(2*dt/3)*xdot,x,xdot))
            xdoti = ODESystem(*xi)
            x = tuple(map(lambda x,xdot,xdoti:x+(.25*dt)*(xdot+3*xdoti),x,xdot,xdoti))
        return x
    return f

def Hamiltonian(K):
    def f(p,q):
        return .5*(p*K(q,q,p)).sum()
    return f

def HamiltonianSystem(K):
    H = Hamiltonian(K)
    def f(p,q):
        Gp,Gq = grad(H(p,q),(p,q), create_graph=True)
        return -Gq,Gp
    return f

def Shooting(p0,q0,K,deltat=1.0,Integrator=HeunsIntegrator()):
    return Integrator(HamiltonianSystem(K),(p0,q0),deltat)

def Flow(x0,p0,q0,K,deltat=1.0,Integrator=HeunsIntegrator()):
    HS = HamiltonianSystem(K)
    def FlowEq(x,p,q):
        return (K(x,q,p),)+HS(p,q)
    return Integrator(FlowEq,(x0,p0,q0),deltat)[0]

def lossVarifoldSurf(FS,VT,FT,K):
    def CompCLNn(F,V):
        V0, V1, V2 = V.index_select(0,F[:,0]), V.index_select(0,F[:,1]), V.index_select(0,F[:,2])
        C, N = .5*(V0+V1+V2), .5*torch.cross(V1-V0,V2-V0)
        L = (N**2).sum(dim=1)[:,None].sqrt()
        return C,L,N/L
    CT,LT,NTn = CompCLNn(FT,VT)
    cst = (LT*K(CT,CT,NTn,NTn,LT)).sum()
    def f(VS):
        CS,LS,NSn = CompCLNn(FS,VS)
        return cst + (LS*K(CS,CS,NSn,NSn,LS)).sum() - 2*(LS*K(CS,CT,NSn,NTn,LT)).sum()
    return f

def LDDMMloss(K,loss,gamma=0):
    def f(p0,q0):
        p,q = Shooting(p0,q0,K)
        return gamma * Hamiltonian(K)(p0,q0) + loss(q)
    return f

def CpuOrGpu(x):
    if torch.cuda.is_available():
        with torch.cuda.device(1):
            x = tuple(map(lambda x:x.cuda(),x))
    return x

VS,FS,VT,FT = CpuOrGpu(torch.load('data/hippos_reduc_reduc.pt'))

q0 = VS = Variable(VS,requires_grad=True)
VT, FS, FT = Variable(VT), Variable(FS), Variable(FT)

lossData = lossVarifoldSurf(FS,VT,FT,GaussLinKernelKeopsGeneric(sigma=20))

n,d = q0.shape
p0 = Variable(CpuOrGpu(torch.zeros(n,d)), requires_grad=True)

optimizer = torch.optim.LBFGS([p0])

#%%

Kv = GaussKernelKeopsGeneric(sigma=20)



loss = LDDMMloss(Kv,lossData)

N = 5

start = time.time()
for i in range(N):
    def closure():
        optimizer.zero_grad()
        L = loss(p0,q0)
        L.backward()
        return L
    optimizer.step(closure)
print('Optimization time : ',round(time.time()-start,2),' seconds')

#%%

nt = 10
Q = np.zeros((n,d,nt+1))
p,q = p0,q0
for i in range(nt):
    Q[:,:,i] = q.data.cpu()
    p,q = Shooting(p,q,Kv,1/nt,HeunsIntegrator(1))
Q[:,:,nt] = q.data.cpu()

fig = plt.figure();
plt.title('LDDMM matching example')  
q0np, qnp, FSnp = q0.data.cpu().numpy(), q.data.cpu().numpy(), FS.data.cpu().numpy()
VTnp, FTnp = VT.data.cpu().numpy(), FT.data.cpu().numpy()
#ax = fig.gca(projection='3d')
ax = Axes3D(fig)
ax.axis('equal')
ax.plot_trisurf(q0np[:,0],q0np[:,1],q0np[:,2],triangles=FSnp,alpha=.5)
ax.plot_trisurf(qnp[:,0],qnp[:,1],qnp[:,2],triangles=FSnp,alpha=.5)
ax.plot_trisurf(VTnp[:,0],VTnp[:,1],VTnp[:,2],triangles=FTnp,alpha=.5)
