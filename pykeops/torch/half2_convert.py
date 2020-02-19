

import torch
from pykeops.common.parse_type import get_type

def make_odd_cat(y):
    bdims = y.shape[:-2]
    N, D = y.shape[-2:]
    if N%2==0:
        ycut = y[...,:-1,:].view(bdims+(N-1,D))
        yend = y[...,-1,:].view(bdims+(1,D))
        y = torch.cat((y,yend,ycut),dim=-2)
    else:
        y = torch.cat((y,y),dim=-2)
    return y, N
 
def make_even_size(x):
    bdims = x.shape[:-2]
    M, D = x.shape[-2:]
    if M%2==1:
        xend = x[...,-1,:].view(bdims+(1,D)) # used as dummy row to insert into x
        x = torch.cat((x,xend),dim=-2)
        tag_dummy = True
    else:
        tag_dummy = False
    return x, tag_dummy

def half2half2(x):
    bdims = x.shape[:-2]
    M, D = x.shape[-2:]
    return x.view(bdims+(M//2,2,D)).transpose(-1,-2).contiguous().view(bdims+(M,D))

def half22half(x):
    bdims = x.shape[:-2]
    M, D = x.shape[-2:]
    return x.view(bdims+(M//2,D,2)).transpose(-1,-2).contiguous().view(bdims+(M,D))

def ranges2half2(ranges,N):
    # we have to convert true indices to half2 indices, and take into account the special
    # concatenate operation done in make_odd_cat, which doubles the number of ranges along j axis.
    ranges_i, slices_i, redranges_j = ranges
    ranges_i[:,0] = torch.floor(ranges_i[:,0]/2.).int()
    ranges_i[:,1] = torch.ceil(ranges_i[:,1]/2.).int()
    slices_i = torch.cat((slices_i,slices_i+redranges_j.shape[0]))
    redranges_j_block1 = torch.zeros(redranges_j.shape)
    redranges_j_block1[:,0] = torch.floor(redranges_j[:,0]/2.).int()
    redranges_j_block1[:,1] = torch.ceil(redranges_j[:,1]/2.).int()
    redranges_j_block2 = torch.zeros(redranges_j.shape)
    redranges_j_block2[:,0] = N//2 + torch.floor((redranges_j[:,0]+1)/2.).int()
    redranges_j_block2[:,1] = N//2 + torch.ceil((redranges_j[:,1]+1)/2.).int()
    if N%2==0:
        # special treatment in case the last range goes to the end of the array
        if redranges_j[-1,1]==N:
            redranges_j_block1[-1,1] += 1
            redranges_j_block2[-1,1] -= 1
    redranges_j = torch.cat((redranges_j,redranges_j_block2),dim=0)
    return ranges_i, slices_i, redranges_j

def preprocess_half2(args, aliases, axis, ranges, nx, ny):
    N = ny if axis==1 else nx
    if ranges is not None:
        if axis==1:
            ranges = ranges2half2(ranges[0:3],ny) + ranges[3:6]
        else:
            ranges = ranges[0:3] + ranges2half2(ranges[3:6],nx)
    newargs = len(aliases)*[None]
    for (var_ind, sig) in enumerate(aliases):
        _, cat, dim, pos = get_type(sig, position_in_list=var_ind)
        arg = args[pos].data # we don't want to record our cuisine in the Autograd mechanism !
        if cat==2:
            arg = arg[...,None,:]      # (...,D)   -> (...,1,D)
            arg, _ = make_even_size(arg)  # (...,1,D) -> (...,2,D)
        elif cat==axis:
            arg, Narg = make_odd_cat(arg)
            N = max(N,Narg)
        else:
            arg, tag_dummy = make_even_size(arg)
        arg = half2half2(arg)
        if cat==2:
            arg = arg.view(tuple(arg.shape[:-2])+(2*dim,))   # (...,2,D) -> (...,2*D) (we "hide" the factor 2 in the dimension...)
        newargs[pos] = arg
    return newargs, ranges, tag_dummy, N

def postprocess_half2(out, tag_dummy, reduction_op, N):
    out = half22half(out)
    if tag_dummy:
        out = out[...,:-1,:]
    if reduction_op in ('ArgMin', 'ArgMax', 'ArgKMin'):
        outind = out
    elif reduction_op in ('Min_ArgMin', 'MinArgMin', 'Max_ArgMax', 'MaxArgMax', 'KMinArgKMin', 'KMin_ArgKMin'):
        outind = out[...,out.shape[-1]//2:]
    else:
        return out
    if N%2==0:
        outind[outind==N] = N - 1
        outind[outind>N] -= N + 1
    else:
        outind[outind>=N] -= N
    return out


