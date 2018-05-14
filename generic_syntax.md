# Writing custom formulas in KeOps.

KeOps uses a low-level syntax written in C++/Cuda to define virtually any reduction operation of the form

```math
\alpha_i = \text{Reduction}_j \big[ f(x^0_{\iota_0}, ... , x^{n-1}_{\iota_{n-1}})  \big]
```

where "Reduction" can be the Sum or the LogSumExp operation.
Each of the variables $`x^k_{\iota_k}`$ is specified by its positional index $`k`$, its category $`\iota_k\in\{i,j,\emptyset\}`$ (meaning that the variable is indexed by i, by j, or is a fixed parameter across indices) and its dimension $`d_k`$. These three characteristics are encoded as follows :

- category is given via the keywords "Vx", "Vy", "Pm" (meaning respectively: "variable indexed by i", "variable indexed by j", and "parameter")
- positional index $`k`$ and dimension $`d_k`$ are given as two integer parameters put into parenthesis after the category-specifier keyword.

For instance, `Vx(2,4)` specifies a variable indexed by "i", given as the third (k=2) input in the function call, and representing a vector of dimension 4.

Of course, using the same index $`k`$ for two different variables is not allowed and will be rejected by the compiler.

From these "variables" symbolic placeholders, one can build the function $`f`$ using standard mathematical operations, say

```cpp
Square(Pm(0,1)-Vy(1,1))*Exp(Vx(2,3)+Vy(3,3))
```

in which `+` and `-` denote the usual addition of vectors, `Exp` is the (element-wise) exponential function
and `*` denotes scalar-vector multiplication.

The operations available are listed below:

```cpp
a*b : scalar-vector multiplication (if a is scalar) or vector-vector element-wise multiplication
a+b : addition of two vectors
a-b : difference between two vectors or minus sign
a/b : element-wise division
(a,b) : scalar product between vectors
Exp(a) : element-wise exponential function
Log(a) : natural logarithm (element-wise)
Pow(a,N) : N-th power of a (element-wise), where N is a fixed-size integer
Pow(a,b) : power operation - alias for Exp(b*Log(a))
Square(a) : alias for Pow(a,2)
Grad(a,x,e) : gradient of a with respect to the variable x, with e as the "grad_output" to backpropagate
```

Variables can be given aliases, allowing us to write human-readable expressions for our formula. For example, one may define
`p=Pm(0,1)`, `a=Vy(1,1)`, `x=Vx(2,3)`, `y=Vy(3,3)`, and write the previous computation as

```cpp
Square(p-a)*Exp(x+y)
```

## Using the syntax in C++/Cuda code

The expressions and variables presented above all correspond to specific C++ types of variables defined by the KeOps library.
The C++ keyword "auto" allows us to define them without having to worry about explicit type naming:

```cpp
auto p = Pm(0,1);
auto a = Vy(1,1);
auto x = Vx(2,3);
auto y = Vy(3,3);
auto f = Square(p-y)*Exp(x+y);
```

Here, the `f` variable represents a symbolic computation; as a C++ object, it is completely useless.
However, we can retrieve its *type* -- which contains all the relevant information -- using the `decltype` keyword :

```cpp
using F = decltype(f);
```

Finally, the convolution operation is performed using one of these calls :

```cpp
CpuConv(Generic<F>::sEval(), Nx, Ny, pc, pp, pa, px, py);
GpuConv1D(Generic<F>::sEval(), Nx, Ny, pc, pp, pa, px, py);
GpuConv2D(Generic<F>::sEval(), Nx, Ny, pc, pp, pa, px, py);
```

where `pc`, `pp`, `pa`, `px`, and `py` are pointers to their respective arrays in (Cpu) memory, `pc` denoting the output. These three functions correspond to computations performed repectively on the Cpu, on the Gpu with a "1D" tiling algorithm, and with a "2D" tiling algorithm.

If data arrays are directly located in Gpu memory, one can call the more direct functions :

```cpp
GpuConv1D_FromDevice(Generic<F>::sEval(), Nx, Ny, pc, pp, pa, px, py);
GpuConv2D_FromDevice(Generic<F>::sEval(), Nx, Ny, pc, pp, pa, px, py);
```

## Using the syntax in the PyTorch bindings

The example described below is implemented in the example Python script [`generic_syntax_pytorch.py`](./pykeops/examples/generic_syntax_pytorch.py) located in [`./pykeops/examples/`](./pykeops/examples/). 
We also provide a PyTorch-free binding, showcased in [`generic_syntax_numpy.py`](./pykeops/examples/generic_syntax_numpy.py).

The Python helper functions `generic_sum` and `generic_logsumexp` take as input the formula and aliases `str` objects:

```python
from pykeops.torch.generic_sum       import generic_sum
from pykeops.torch.generic_logsumexp import generic_logsumexp

formula = "Square(p-a)*Exp(x+y)"
types   = ["output = Vx(3)", "p = Pm(1)","a = Vy(1)","x = Vx(3)","y = Vy(3)"]
routine = generic_sum(formula, *types)
```

Notice that the variables' indices are inferred from their positions in the list of types:
output first, then the arguments one after the other.
Having defined our symbolic reduction operation, we can then apply it
on arbitrary torch tensors, just like any other (fully-differentiable) PyTorch function:

```python
c = routine(p,a,x,y)
```

In this example, `p`, `a`, `x` and `y` should correspond to tensors with compatible sizes :

- `p` must be a vector of size 1 - with torch.Size([1]), not torch.Size([]) or torch.Size([1,1]).
- `a` must be a "list" of 1-vectors, i.e. a variable of size n*1, where n can be any positive integer,
- `x` must be a "list" of 3-vectors, i.e. a variable of size m*3, where m can be any positive integer,
- `y` must be a "list" of 3-vectors, i.e. a variable of size n*3.
- output `c` will be a "list" of 3-vectors, i.e. a variable of size m*3.

If the required formula has not been used on the current machine since the last KeOps update,
the previous call will first perform an on-the-fly compilation, producing (and storing) the appropriate .dll/.so file.
Otherwise, the CUDA kernel will simply be loaded into RAM memory.

**Autodiff engine.**
KeOps has an internal automatic differentiation engine for symbolic formulas
-- compatible with the PyTorch autograd package -- that allows
us to "bootstrap" all the derivatives required by the user (including gradients of gradients, etc.).
Feel free to use the output of `kernel_product`, `generic_sum` or `generic_logsumexp`
as any other torch tensor!

## Using the syntax in the Matlab bindings

The example described below is implemented in the example Matlab script script_generic_syntax.m located in keopslab/examples. 

The Matlab bindings provide a function Kernel which can be used to define the corresponding convolution operations. Following the previous example, one may write

```matlab
f = Kernel('p=Pm(0,1)','a=Vy(1,1)','x=Vx(2,3)','y=Vy(3,3)','Square(p-a)*Exp(x+y)');
```
which defines a Matlab function f which can be used to perform a sum reduction for this formula :

```matlab
c = f(p,a,x,y);
```

where p, a, x, y must be arrays with compatible dimensions as previously explained. A gradient function GradKernel is also provided. For example, to get the gradient with respect to y of the previously defined function f, one needs to write :

```matlab
Gfy = GradKernel(f,'y','e=Vx(4,3)');
```

which returns a new function that can be used as follows :

```matlab
Gfy(p,a,x,y,e)
```

where e is the input gradient array.