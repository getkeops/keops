from pykeops.torch import Vi, Vj

def GaussLapKernel(sigma,D):
    x, y = Vi(0, D), Vj(1, D)
    D2 = x.sqdist(y)
    K = (-D2 /(2*sigma**2)).exp()
    return (K *(D2-D*sigma**2)/sigma**4)

def GaussK(sigma):
    def K(z):
        return (-(z**2).sum(-1)/(2*sigma**2)).exp()
    return K

def LapKernel(K,D):
    x, y = Vi(0, D), Vj(1, D)
    K1 = K(x-y).grad(x,1)
    Klap = K1.elem(0).grad(x,1).elem(0)
    for i in range(1,D):
        Klap = Klap + K1.elem(i).grad(x,1).elem(i)
    return Klap

def LapKernel_alt(K,D):
    x, y = Vi(0, D), Vj(1, D)
    K1 = K(x-y).grad(x,1)
    GK1 = K1.grad_matrix(x)
    Klap = GK1.elem(0)
    for i in range(1,D**2,D):
        Klap = Klap +GK1.elem(i)
    return Klap

sigma, D = 1.5, 3

f1 = GaussLapKernel(sigma,D)
print("f1:")
print(f1)

f2 = LapKernel(GaussK(sigma),D)
print("f2:")
print(f2)

f3 = LapKernel_alt(GaussK(sigma),D)
print("f3:")
print(f3)