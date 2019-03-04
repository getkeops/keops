import torch
from pykeops.torch import Genred as Genred

def is_on_device(x):
    return x.is_cuda
    
class torchtools :
    copy = torch.clone
    exp = torch.exp
    norm = torch.norm
    Genred = Genred
    def __init__(self):
        self.transpose = lambda x : x.t()
        self.solve = lambda A, b : torch.gesv(b,A)[0].contiguous()
        self.arraysum = lambda x, axis=None : x.sum() if axis is None else x.sum(dim=axis)
        self.numpy = lambda x : x.cpu().numpy()
        self.tile = lambda *args : torch.Tensor.repeat(*args)
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
     

