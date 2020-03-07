# Example of operation over image patches with KeOps
# Given an image I of shape (m,n,d), we compute the K nearest patches of 
# every s-by-s image patch of the image

import torch
from pykeops.torch import LazyTensor

id_device = 0
device = 'cuda:'+str(id_device) if torch.cuda.is_available() else 'cpu'

if 'cuda' in device:
    m, n = 512, 512
else:
    m, n = 30, 30

d, s, K = 3, 5, 4

torch.manual_seed(2 )

I = torch.rand(m,n,d).double().to(device)
#I = torch.arange(m*n*d).float().view(m,n,d).to(device)


# function to buill image patches with PyTorch, used as reference

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


# testing with PyTorch (only possible for small images)
import time
if m*n<20000:
    start = time.time()
    P = torch_patches(I,s)
    D2 = ((P[:,None,:]-P[None,:,:])**2).sum(dim=2)
    out_torch = D2.argsort(dim=1)[:,:4]
    print("elapsed time PyTorch : ",time.time()-start)




# function to represent image patches as a LazyTensor

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
        rgm = torch.arange(m-s+1)[:,None].int()
        ranges = torch.cat((rgm*n,rgm*n+n-s+1),dim=1).to(I.device)
        slices = ((torch.arange(m-s+1).int()+1)*(m-s+1)).to(I.device)
        red_ranges = (torch.cat((ranges,)*(m-s+1),dim=0)).to(I.device)
        return P, (ranges,slices,red_ranges,ranges,slices,red_ranges)
    else:
        return P

def LazyTensor_patches_v2(I,s,axis,use_ranges=True):
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
        ranges = torch.tensor([[0,ind_last]]).int()
        slices = torch.tensor([m-s+1]).int()
        rgm = torch.arange(m-s+1)[:,None].int()
        red_ranges = torch.cat((rgm*n,rgm*n+n-s+1),dim=1).to(I.device)
        return P, (ranges,slices,red_ranges,ranges,slices,red_ranges)
    else:
        return P


# testing with KeOps - with ranges

start = time.time()

# creating LazyTensor objects for patches and computing
P_i, ranges = LazyTensor_patches(I,s,axis=0)
P_j, ranges = LazyTensor_patches(I,s,axis=1)
D2 = P_i.sqdist(P_j)
out_keops = D2.argKmin(dim=1, K=K, ranges=ranges)

# now post-processing to get rid of the garbage rows of the output
ind_keep = torch.arange(m*n).view(m,n)[:-s+1,:-s+1].flatten()
out_keops = out_keops[ind_keep,:]
# Also here the output corresponds to patch indices, for the list of patches that includes
# garbage patches. So we need to convert the indices
q = out_keops // n
r = out_keops % n
out_keops = q*(n-s+1)+r

print("elapsed time with KeOps (ranges) : ",time.time()-start)

if m*n<20000:
    print("errors : ",(out_keops.cpu()!=out_torch).sum().item())


# testing with KeOps - with ranges v2

start = time.time()

# creating LazyTensor objects for patches and computing
P_i, ranges = LazyTensor_patches_v2(I,s,axis=0)
P_j, ranges = LazyTensor_patches_v2(I,s,axis=1)
D2 = P_i.sqdist(P_j)
out_keops = D2.argKmin(dim=1, K=K, ranges=ranges)

# now post-processing to get rid of the garbage rows of the output
ind_keep = torch.arange(m*n).view(m,n)[:-s+1,:-s+1].flatten()
out_keops = out_keops[ind_keep,:]
# Also here the output corresponds to patch indices, for the list of patches that includes
# garbage patches. So we need to convert the indices
q = out_keops // n
r = out_keops % n
out_keops = q*(n-s+1)+r

print("elapsed time with KeOps (ranges) : ",time.time()-start)

if m*n<20000:
    print("errors : ",(out_keops.cpu()!=out_torch).sum().item())



# testing with KeOps - with padding and no ranges

# padding the image with very large values on the right border
Ipad = torch.cat((I,1e20*torch.ones(m,1,d, device=device).double()),dim=1)

start = time.time()

# creating LazyTensor objects for patches and computing
P_i = LazyTensor_patches(Ipad,s,axis=0,use_ranges=False)
P_j = LazyTensor_patches(Ipad,s,axis=1,use_ranges=False)
D2 = P_i.sqdist(P_j)
out_keops = D2.argKmin(dim=1,K=K)

# now post-processing to get rid of the garbage rows of the output
ind_keep = torch.arange(m*(n+1)).view(m,n+1)[:-s+1,:-s].flatten()
out_keops = out_keops[ind_keep,:]
# Also here the output corresponds to patch indices, for the list of patches that includes
# garbage patches. So we need to convert the indices
q = out_keops // (n+1)
r = out_keops % (n+1)
out_keops = q*(n-s+1)+r

print("elapsed time with KeOps (padding) : ",time.time()-start)

if m*n<20000:
    print("error : ",(out_keops.cpu()!=out_torch).sum().item())

ind = torch.argmax((out_torch!=out_keops).sum(dim=1))
print(ind)
print(out_torch[ind,:])
print(out_keops[ind,:])
print(((P[608,:]-P[178,:])**2).sum())
print(((P[608,:]-P[543,:])**2).sum())
a = ((P[608,:]-P[178,:])**2).sum()
b = ((P[608,:]-P[543,:])**2).sum()
print(a-b)
