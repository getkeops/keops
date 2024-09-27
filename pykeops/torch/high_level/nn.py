import torch
from pykeops.torch import LazyTensor


# define activations - not efficient. Will prob pass this to hpc code
ACTS = {}
ACTS["relu"] = lambda x: x.relu()
ACTS["sigmoid"] = lambda x: 1 / ( 1 + (-x).exp() )
ACTS["tanh"] = lambda x: ( x.exp()-(-x).exp() ) / ( x.exp()+(-x).exp() )
ACTS["bisigmoid"] = lambda x: -1 + 2 * ACTS["sigmoid"](x)
ACTS["silu"] = lambda x: x * ACTS["sigmoid"](x)
ACTS["gelu_fast"] = lambda x: x * ACTS["sigmoid"](1.7206*x)
# parametrized
ACTS["elu"] = lambda x: x.relu() + a * ( x.clamp(float("-inf"), 0).exp()-1 )
ACTS["leaky_relu"] = lambda x,a=0.3: x.relu() + a * x.clamp(float("-inf"), 0)


def kernel_linear(x, linear=None, w=None, b=None, act=None):
    """ Applies a PyTorch nn.Linear layer to a keops LazyTensor. 
        Inputs: 
        * x: (..., D) LazyTensor to apply the MLP on
        * linear: torch.nn.Linear, or any class with "weight" and "bias" attrs
        * w: (M, N): a weights matrix
        * b: (N,): a bias vector
        * act: func. a non-linearity. already prepared for LazyTensors
        Output: (..., N)
    """
    assert linear is not None or w is not None or b is not None, 
        "A torch.nn.Linear or weights and biases must be passed"

    # get params from linear if not passed directly
    if linear is not None and w is None:
        w = linear.weight
        b = linear.bias 

    # reshape to last dim
    w_ = keops_torch.LazyTensor(w.view(1, 1, -1))
    b_ = keops_torch.LazyTensor(b.view(1, 1, -1)) if b is not None else 0
        
    # computation
    out = w_.matvecmult(x) + b_
    if act is not None: 
        out = act(out)

    return out


## ATTENTION CLASSES

class Multihead_Attention(torch.nn.Module): 
    def __init__(self, dim, heads=1, dim_head = 64, bias=True, **kwargs):
        """ Implements Mutihead attention. For self-attention,
            use same inputs and context.
            * dim: int. input dim.  
            * heads: int. head number. 
            * dim_head: int. head dimension for inner product
            * bias: bool. whether to use bias in MLPs
        """
        super().__init__()
        inner_dim = heads * dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_q = torch.nn.Linear(dim, inner_dim, bias = bias)
        self.to_k = torch.nn.Linear(dim, inner_dim, bias = bias)
        self.to_v = torch.nn.Linear(dim, inner_dim, bias = bias)
        self.to_out = torch.nn.Linear(inner_dim, dim)
        

    def forward(self, x, context, mask = None, **kwargs):
        """ Inputs: 
            * x: (..., N, dim). Generate queries from. 
            * context: (..., N, dim). Generate keys and values.
            * mask: (..., N). for masked attention. 

        """
        # rearrange( self.to_q(x), '... n (h d) -> ... h n () d', h=heads )
        q = self.to_q(x) * self.scale
        q.reshape_(*q.shape[:-1], self.heads, -1)
        q.transpose_(-2, -3)
        q.unsqueeze_(-2)
        # rearrange( self.to_k(context), '... n (h d) -> ... h () n d', h=heads )
        k = self.to_k(context)
        k.reshape_(*k.shape[:-1], self.heads, -1)
        k.transpose_(-2, -3)
        k.unsqueeze_(-3)
        # rearrange( self.to_v(context), '... n (h d) -> ... h () n d', h=heads )
        v = self.to_k(context)
        v.reshape_(*v.shape[:-1], self.heads, -1)
        v.transpose_(-2, -3)
        v.unsqueeze_(-3)
        
        # kernelized inner prod
        q,k,v = map( lambda t: keops_torch.LazyTensor(t), (q,k,v) )
        attn = (q*k).sum(dim=-1) # way faster than (q | k) # (B, H, N, N, 1)

        if mask is not None:
            mask = LazyTensor( mask.unsqueeze(-1) )
            attn += mask
        
        out = attn.sumsoftmaxweight(v, dim=len(x.shape)) # (B H N N) · (B H N D)
        out.transpose_(-3, -2) 
        out.reshape_(*out.shape[:-2], -1)

        return self.to_out( out )


