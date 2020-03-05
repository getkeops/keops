#!/usr/bin/env python
# coding: utf-8

# In[58]:


import torch
from pykeops.torch import LazyTensor

def torch_patches(I,s):
    # inputs : image I, torch tensor of shape (m,n,d)
    #         size of patches s, integer
    # output : matrix of all s-by-s patches of I, torch tensor of shape ((m-s+1)*(n-s+1),d*s**2)
    m, n, d = I.shape
    P = torch.zeros((m-s+1)*(n-s+1),s**2,d)
    k = 0
    for i in range(s):
        for j in range(s):
            P[:,k,:] = I[i:i+m-s+1,j:j+n-s+1,:].flatten().view(-1,d)
            k += 1
    return P.view(-1,d*s**2)

def LazyTensor_patches(I,s,axis):
    # input : image I, torch tensor of shape (m,n,d)
    #         size of patches s, integer
    # output : LazyTensor representing all s-by-s patches of I, shape ((m-s+1)*n-s+1,d*s**2)
    # N.B. there are (m-s)*(s-1) garbage patches, corresponding to indices (i,m-s+1),...(i,m-1) 
    # for each row i except the last one. These garbage patches will not be taken into account in computations
    # but will remain in the output as zero valued rows.
    m, n, d = I.shape
    I = I.view(m*n,d)
    P = ()
    ind_last = (m-s+1)*n-s+1
    for i in range(s):
        for j in range(s):
            ind_shift = i*n + j
            P += (LazyTensor(I[ind_shift:ind_last+ind_shift,:],axis=axis),)
    P = LazyTensor.cat(P,dim=-1)
    rgm = torch.arange(m-s+1)[:,None].int()
    ranges = torch.cat((rgm*n,rgm*n+n-s+1),dim=1)
    slices = (torch.arange(m-s+1).int()+1)*(m-s+1)
    red_ranges = torch.cat((ranges,)*(m-s+1),dim=0)
    P.ranges = (ranges,slices,red_ranges,ranges,slices,red_ranges)
    return P


# In[60]:


m, n, d, s, K = 20, 20, 3, 5, 4
I = torch.rand(m,n,d)

import time

# PyTorch
start = time.time()
P = torch_patches(I,s)
D2 = ((P[:,None,:]-P[None,:,:])**2).sum(dim=2)
out_torch = D2.argsort(dim=1)[:,:4]
print("elapsed torch : ",time.time()-start)

# KeOps
start = time.time()
P_i = LazyTensor_patches(I,s,axis=0)
P_j = LazyTensor_patches(I,s,axis=1)
D2 = P_i.sqdist(P_j)
out_keops = D2.argKmin(dim=1,K=4)
# now post-processing to get rid of the garbage rows of the output
ind_keep = torch.arange(m*n).view(m,n)[:-s+1,:-s+1].flatten()
# here the output corresponds to patch indices, for the list of patches that includes
# garbage patches. So we need also to convert the indices
out_keops = out_keops[ind_keep,:]
q = out_keops // n
r = out_keops % n
out_keops = q*(n-s+1)+r
print("elapsed keops : ",time.time()-start)
print("erreur : ",(out_keops-out_torch).abs().sum().item())


# In[ ]:




