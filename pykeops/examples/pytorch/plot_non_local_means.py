
"""
=======================================
Image denoising with non local means 
=======================================

Another example of operation over image patches with KeOps.
Denoising via non local means (brute-force)
 
"""

###############################################################
# Setup
# -------------
# Standard imports:

import torch
from pykeops.torch import LazyTensor

id_device = 1
device = 'cuda:'+str(id_device) if torch.cuda.is_available() else 'cpu'

###############################################################
# Parameters (s size of block, K number of nearest neighbours)
s, K = 5, 10
sd2 = s//2

###############################################################
# Loading image
import imageio
if 'cuda' in device:
    imfile = "https://homepages.cae.wisc.edu/~ece533/images/fruits.png"
else:
    #imfile = "http://helios.math-info.univ-paris5.fr/~glaunes/pikagotmilk64.bmp"
    imfile = "http://helios.math-info.univ-paris5.fr/~glaunes/pikachu.bmp"
I = torch.tensor(imageio.imread(imfile)).float().to(device)
sigma = 30.
I += torch.randn(I.shape, device=device)*sigma
I = I.clamp(0,255)
h2 = (1.*sigma)**2

m, n, d = I.shape

###############################################################
# PyTorch implementation (works only for small images)
# ----------------------------------------------------
###############################################################
# function to build image patches with PyTorch
def torch_patches(I,s):
    # inputs : I torch tensor of shape (m,n,d), image
    #          s : integer, size of patches
    # output : torch tensor of shape ((m-s+1)*(n-s+1),d*s**2), matrix of all s-by-s patches of I
    m, n, d = I.shape
    P = torch.zeros((m-s+1)*(n-s+1),s**2,d)
    k = 0
    for i in range(s):
        for j in range(s):
            P[:,k,:] = I[i:i+m-s+1,j:j+n-s+1,:].flatten().view(-1,d)
            k += 1
    return P.view(-1,d*s**2)

###############################################################
# Testing with PyTorch if image is not too large
import time
if m*n<20000:
    start = time.time()
    P = torch_patches(I,s)
    D2 = ((P[:,None,:]-P[None,:,:])**2).sum(dim=2)
    K = (-D2/(d*s**2*h2)).exp()
    C = K.sum(dim=1)
    out_torch = (K[:,:,None]*I[sd2:-sd2,sd2:-sd2,:].reshape((1,(m-s+1)*(n-s+1),d))).sum(dim=1) / C[:,None]
    out_torch = out_torch.reshape((m-s+1,n-s+1,d))
    print("elapsed time with PyTorch : ",time.time()-start)


###############################################################
# KeOps implementation
# ----------------------------------------------------

###############################################################
# Function to represent image patches as a LazyTensor

def LazyTensor_patches(I,s,axis,use_ranges=True):
    # input : I torch tensor of shape (m,n,d), image
    #         s : integer, size of patches
    # output : LazyTensor of shape ((m-s+1)*n-s+1,d*s**2) representing all s-by-s patches of I, 
    # N.B. there are (m-s)*(s-1) "garbage" patches, corresponding to indices (i,m-s+1),...(i,m-1)
    # for each row i except the last one. These patches will not be taken into account in computations
    # if use_ranges=True, but will remain in any case in the output as zero valued rows.
    m, n, d = I.shape
    I = I.view(m*n,d)
    ind_last = (m-s+1)*n-s+1
    for i in range(s):
        for j in range(s):
            if i==0 and j==0:
                P = LazyTensor(torch.narrow(I,0,0,ind_last),axis=axis)
            else:
                ind_shift = i*n + j
                P = P.concat(LazyTensor(torch.narrow(I,0,ind_shift,ind_last),axis=axis))
    if use_ranges:
        # we define the range of computation using the range arguments of KeOps
        # This is used to avoid computing over "garbage" patches
        ranges = torch.tensor([[0,ind_last]]).int().to(I.device)
        slices = torch.tensor([m-s+1]).int().to(I.device)
        rgm = torch.arange(m-s+1)[:,None].int()
        red_ranges = torch.cat((rgm*n,rgm*n+n-s+1),dim=1).to(I.device)
        return P, (ranges,slices,red_ranges,ranges,slices,red_ranges)
    else:
        return P

###############################################################
# Testing with KeOps - using ranges
if True:
    start = time.time()
    # creating LazyTensor objects for patches and computing
    P_i, ranges = LazyTensor_patches(I,s,axis=0)
    P_j, ranges = LazyTensor_patches(I,s,axis=1)
    D2 = P_i.sqdist(P_j)
    K = (-D2/(d*s**2*h2)).exp()
    C = K.sum(dim=1)
    ind_keep = torch.arange(m*n).view(m,n)[:-s+1,:-s+1].flatten()
    I_ = torch.zeros((m-s+1)*n-s+1,d, device=device)
    I_[ind_keep,:] = I[sd2:-sd2,sd2:-sd2,:].reshape((m-s+1)*(n-s+1),d)
    out_keops = (K*I_[None,:,:]).sum(dim=1) / C
    out_keops = out_keops[ind_keep,:]
    out_keops = out_keops.reshape((m-s+1,n-s+1,d))
    print("elapsed time with KeOps (ranges) : ",time.time()-start)




###############################################################
# displaying the results 
# ----------------------------------------------------

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
def plot_image(I):
    I = I.cpu().numpy().astype(np.uint8)
    I = Image.fromarray(I, 'RGB')
    I = I.resize((256,256))
    plt.imshow(I)

plot_image(I)
plt.figure()
plot_image(out_keops)
plt.show()

