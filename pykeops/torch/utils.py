import torch
from pykeops.torch import Genred
from pykeops.torch.generic.generic_red import Genred_lowlevel

def is_on_device(x):
    return x.is_cuda
    
class torchtools :
    copy = torch.clone
    exp = torch.exp
    log = torch.log
    norm = torch.norm
    Genred = Genred
    Genred_lowlevel = Genred_lowlevel
    def __init__(self):
        self.transpose = lambda x : x.t()
        self.solve = lambda A, b : torch.gesv(b,A)[0].contiguous()
        self.arraysum = lambda x, axis=None : x.sum() if axis is None else x.sum(dim=axis)
        self.numpy = lambda x : x.cpu().numpy()
        self.tile = lambda *args : torch.Tensor.repeat(*args)
        self.size = lambda x : x.numel()
        self.view = lambda x,s : x.view(s)
    def set_types(self,x):
        self.torchdtype = x.dtype
        self.torchdeviceId = x.device
        self.KeOpsdeviceId = self.torchdeviceId.index  # id of Gpu device (in case Gpu is  used)
        self.dtype = 'float32' if self.torchdtype==torch.float32 else 'float64'    
        self.rand = lambda self, m, n : torch.rand(m,n, dtype=self.torchdtype, device=self.torchdeviceId)
        self.randn = lambda m, n : torch.randn(m,n, dtype=self.torchdtype, device=self.torchdeviceId)
        self.zeros = lambda shape : torch.zeros(shape, dtype=self.torchdtype, device=self.torchdeviceId)
        self.eye = lambda n : torch.eye(n, dtype=self.torchdtype, device=self.torchdeviceId)
        self.array = lambda x : torch.tensor(x, dtype=self.torchdtype, device=self.torchdeviceId)
        self.randn = lambda m, n : torch.randn(m,n, dtype=self.torchdtype, device=self.torchdeviceId)
     

def WarmUpGpu():
    # dummy first calls for accurate timing in case of GPU use
    formula = 'Exp(-oos2*SqDist(x,y))*b'
    aliases = ['x = Vx(1)',  # First arg   : i-variable, of size 1
                 'y = Vy(1)',  # Second arg  : j-variable, of size 1
                 'b = Vy(1)',  # Third arg  : j-variable, of size 1
                 'oos2 = Pm(1)']  # Fourth arg  : scalar parameter
    my_routine = Genred(formula, aliases, reduction_op='Sum', axis=1, cuda_type='float32')
    dum = torch.rand(10,1)
    dum2 = torch.rand(10,1)
    my_routine(dum,dum,dum2,torch.tensor([1.0]))
    my_routine(dum,dum,dum2,torch.tensor([1.0]))

