from pykeops.torch import Vi, Vj
from torch import rand


def keops_knn(k, dimension=3):
    xi = Vi(0, dimension)
    xj = Vj(1, dimension)
    dij = (xi - xj).norm2()
    knn_func = dij.Kmin_argKmin(k, dim=1)
    return knn_func


k, d = 5, 3
q_points = rand(d, 100)
s_points = rand(d, 200)

knn_func = keops_knn(k, dimension=d)
knn_distances, knn_indices = knn_func(q_points, s_points)
