
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

def LazyTensor_patches(I,s,axis,use_ranges=True):
    # input : image I, torch tensor of shape (m,n,d)
    #         size of patches s, integer
    # output : LazyTensor representing all s-by-s patches of I, shape ((m-s+1)*n-s+1,d*s**2)
    # N.B. there are (m-s)*(s-1) "garbage" patches, corresponding to indices (i,m-s+1),...(i,m-1) 
    # for each row i except the last one. These garbage patches will not be taken into account in computations
    # if use_ranges=True, but will remain in any case in the output as zero valued rows.
    m, n, d = I.shape
    I = I.view(m*n,d)
    P = ()
    ind_last = (m-s+1)*n-s+1
    for i in range(s):
        for j in range(s):
            ind_shift = i*n + j
            P += (LazyTensor(torch.narrow(I,0,ind_shift,ind_last),axis=axis),)
    P = LazyTensor.cat(P,dim=-1)
    if use_ranges:
        rgm = torch.arange(m-s+1)[:,None].int()
        ranges = torch.cat((rgm*n,rgm*n+n-s+1),dim=1).to(I.device)
        slices = ((torch.arange(m-s+1).int()+1)*(m-s+1)).to(I.device)
        red_ranges = (torch.cat((ranges,)*(m-s+1),dim=0)).to(I.device)
        P.ranges = (ranges,slices,red_ranges,ranges,slices,red_ranges)
    return P




m, n, d, s, K = 30, 30, 1, 5, 4

id_device = 0
device = 'cuda:'+str(id_device) if torch.cuda.is_available() else 'cpu'

I = torch.rand(m,n,d).to(device)

if 'cuda' in device:
    torch.cuda.memory_allocated(id_device)

# testing with PyTorch
import time
if m*n<10000:
    start = time.time()
    P = torch_patches(I,s)
    D2 = ((P[:,None,:]-P[None,:,:])**2).sum(dim=2)
    out_torch = D2.argsort(dim=1)[:,:4]
    print("elapsed torch : ",time.time()-start)



# testing with KeOps - with ranges

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

if m*n<10000:
    print("erreur : ",(out_keops-out_torch).abs().sum().item())



# testing with KeOps - with padding and no ranges

Ipad = torch.cat((I,1e20*torch.ones(m,1,d)),dim=1)

start = time.time()
P_i = LazyTensor_patches(Ipad,s,axis=0,use_ranges=False)
P_j = LazyTensor_patches(Ipad,s,axis=1,use_ranges=False)
D2 = P_i.sqdist(P_j)
out_keops = D2.argKmin(dim=1,K=4)
# now post-processing to get rid of the garbage rows of the output
ind_keep = torch.arange(m*(n+1)).view(m,n+1)[:-s+1,:-s].flatten()
# here the output corresponds to patch indices, for the list of patches that includes
# garbage patches. So we need also to convert the indices
out_keops = out_keops[ind_keep,:]
q = out_keops // (n+1)
r = out_keops % (n+1)
out_keops = q*(n-s+1)+r
print("elapsed keops : ",time.time()-start)

if m*n<10000:
    print("erreur : ",(out_keops-out_torch).abs().sum().item())


