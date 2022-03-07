import torch
import pykeops
from pykeops.torch import LazyTensor


ttypes = (
    (torch.cuda.FloatTensor,) if torch.cuda.is_available() else (torch.FloatTensor,)
)

# Test when LazyTensors share underlying data
for ttype in ttypes:
    torch.set_default_tensor_type(ttype)

    # Input
    f = torch.randn([1000, 1])
    f.requires_grad = True
    f_i = LazyTensor(f[:, None, :])

    f_ref = f.detach().clone()
    f_ref.requires_grad = True

    # Values
    first = lambda x: x.sqrt()
    second = lambda x: x * x * x
    # Keops
    val = f_i.ifelse(first(f_i), second(f_i)).sum(dim=0)
    # Torch
    val_refs = torch.zeros_like(f_ref)
    val_refs[f_ref >= 0] = first(f_ref[f_ref >= 0])
    val_refs[f_ref < 0] = second(f_ref[f_ref < 0])
    val_ref = val_refs.sum(dim=0)

    print("Checking values")
    print(f"|val - val_ref|: {(val - val_ref).abs()}\n")
    assert torch.allclose(val, val_ref)

    # Gradients
    val.backward()
    val_ref.backward()
    print("Checking gradients:")
    print(f"max(|grad - grad_ref|): {(f.grad - f_ref.grad).abs().max()}\n")
    assert torch.allclose(f.grad, f_ref.grad)


# Test when other LazyTensors don't share underlying data (maybe unnecessary)
for ttype in ttypes:
    torch.set_default_tensor_type(ttype)

    # Input
    f = torch.randn([1000, 1])
    g = torch.randn_like(f).abs()
    h = torch.randn_like(f)
    f.requires_grad = True
    g.requires_grad = True
    h.requires_grad = True
    f_i = LazyTensor(f[:, None, :])
    g_i = LazyTensor(g[:, None, :])
    h_i = LazyTensor(h[:, None, :])

    f_ref = f.detach().clone()
    g_ref = g.detach().clone()
    h_ref = h.detach().clone()
    f_ref.requires_grad = True
    g_ref.requires_grad = True
    h_ref.requires_grad = True

    # Values
    first = lambda x: x.sqrt()
    second = lambda x: x * x * x
    val = f_i.ifelse(first(g_i), second(h_i)).sum(dim=0)
    val_refs = torch.zeros_like(f_ref)
    val_refs[f_ref >= 0] = first(g_ref[f_ref >= 0])
    val_refs[f_ref < 0] = second(h_ref[f_ref < 0])
    val_ref = val_refs.sum(dim=0)

    print("Checking values")
    print(f"|val - val_ref|: {(val - val_ref).abs()}\n")
    assert torch.allclose(val, val_ref)

    # Gradients
    val.backward()
    val_ref.backward()
    print("Checking gradients:")
    print(f"f: max(|grad|): {f.grad.abs().max()}\n")
    assert torch.allclose(f.grad, torch.zeros_like(f.grad))
    print(f"g: max(|grad - grad_ref|): {(g.grad - g_ref.grad).abs().max()}\n")
    print(f"h: max(|grad - grad_ref|): {(h.grad - h_ref.grad).abs().max()}\n")
    assert torch.allclose(f.grad, torch.zeros_like(f.grad))
