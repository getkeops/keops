import torch.cuda
from pykeops.torch import Vi, Vj, LazyTensor, Genred
import pykeops


d = 1
n, m = 1, 1
q_points = torch.rand((n, d)).requires_grad_(True).to("cuda")
s_points = torch.rand((m, d)).to("cuda")


# LazyTensor
def keops_knn(x, y):
    xi = LazyTensor(x[:, None, :])
    xj = LazyTensor(y[None, :, :])
    dij = ((xi - xj) ** 2).sum(-1)
    knn_func = dij.sum_reduction(dim=0)
    return knn_func


# Genred
formula = "Sum((a-b)**2)"
aliases = [
    f"a=Vi({d})",
    f"b=Vj({d})",
]
fn = Genred(formula, aliases, reduction_op="Sum", axis=1)


s = [0]
for i in range(100):
    # out_ = keops_knn(q_points, s_points)
    out = fn(q_points, s_points, backend="GPU_1D")

    s.append(torch.cuda.memory.memory_stats()["allocated_bytes.all.current"])
    print(
        "Reserved mem: ",
        torch.cuda.memory.memory_stats()["reserved_bytes.all.current"],
        ", Allocated mem: ",
        s[-1],
        "diff: ",
        s[-1] - s[-2],
    )
    torch.cuda.synchronize()
