import torch.cuda
from pykeops.torch import Vi, Vj, LazyTensor
import pykeops

def keops_knn(x, y):
    xi = LazyTensor(x[:, None, :])
    xj = LazyTensor(y[None, :, :])
    dij = ((xi - xj) ** 4).sum(-1)
    knn_func = dij.sum_reduction(dim=0)
    return knn_func

d = 1
n, m = 1, 1
q_points = torch.rand((n, d)).requires_grad_(True).to("cuda")
s_points = torch.rand((m, d)).to("cuda")

s = [0]
for i in range(100):
    result = keops_knn(q_points, s_points)
    s.append(torch.cuda.memory.memory_stats()["allocated_bytes.all.current"])
    print("Reserved mem: ", torch.cuda.memory.memory_stats()["reserved_bytes.all.current"], ", Allocated mem: ", s[-1], "diff: ", s[-1] - s[-2])
    torch.cuda.synchronize()



