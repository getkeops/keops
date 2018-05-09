# Writing custom formulas in KeOps.

KeOps uses a low-level syntax written in C++/Cuda to define virtually any reduction operation of the form

```math
\alpha_i = \text{Reduction}_j \big[ f(x^0_{\iota_0}, ... , x^{n-1}_{\iota_{n-1}})  \big]
```

where "Reduction" can be the summation or LogSumExp operation. 
Each of the variables $`x^k_{\iota_k}`$ is specified by its positional index $`k`$, its category $`\iota_k\in\{i,j,\emptyset\}`$ (meaning that the variable is indexed by i, by j, or is a fixed parameter) and its dimension $`d_k`$. These three characteristics are entered as follows :

- category is entered via the keywords "Vx", "Vy", "Pm" (meaning respectively: "variable indexed by i", "variable indexed by j", and "parameter")
- positional index $`k`$ and dimension $`d_k`$ are entered as two integer parameters put into parenthesis after the previous keyword. 

Hence for example Vx(2,4) specifies a 4-dimensional variable which will be given as the third (k=2) input in the function call, and is indexed by i.

Of course, using the same index $`k`$ for two different variables is not allowed and will be rejected by the compiler.

From these variables expressions, one can build the function $f$ using usual mathematical operations, such as for example

```cpp
Square(Pm(0,1)-Vy(1,1))*Exp(Vx(2,3)+Vy(3,3))
```

in which the + and - operations denotes usual addition of vectors, Exp is the (element-wise) exponential function, and the
* sign denotes scalar-vector multiplication.

Here are the operations available:

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
Grad(a,x,e) : gradient of a with respect to x, and input e
```

Variables can be given aliases to get a more readable expression of the formula. For example, one may define 
p=Pm(0,1), a=Vy(1,1), x=Vx(2,3), y=Vy(3,3), and then write the previous expression as 

```cpp
Square(p-a)*Exp(x+y)
```

**Using the syntax in C++/Cuda code**

All these previous expressions and variables correspond to specific C++ types of variables defined in the KeOps library. The C++ keyword "auto" allows to define them without worrying about naming explicitely their type:

```cpp
auto p = Pm(0,1);
auto a = Vy(1,1);
auto x = Vx(2,3);
auto y = Vy(3,3);
auto f = Square(p-y)*Exp(x+y);
```
The f variable defined here only represents the symbolic expression of the formula and as a C++ object is in fact completely useless. The only important information is precisely its type, which we get with the "decltype" keyword :

```cpp
using F = decltype(f);
```
Finally, the convolution operation is performed using one of these calls :

```cpp
CpuConv(Generic<F>::sEval(), Nx, Ny, pc, pp, pa, px, py);
GpuConv1D(Generic<F>::sEval(), Nx, Ny, pc, pp, pa, px, py);
GpuConv2D(Generic<F>::sEval(), Nx, Ny, pc, pp, pa, px, py);
```
where pc, pp, pa, px, and py are pointers to their respective arrays in (Cpu) memory, pc denoting the output. These three functions correspond to computations performed repectively on the Cpu, on the Gpu with a "1D" tiling algorithm, and with a "2D" tiling algorithm. 

If data arrays are directly located in Gpu memory, one can call the more direct functions :

```cpp
GpuConv1D_FromDevice(Generic<F>::sEval(), Nx, Ny, pc, pp, pa, px, py);
GpuConv2D_FromDevice(Generic<F>::sEval(), Nx, Ny, pc, pp, pa, px, py);
```

**Using the syntax in the PyTorch bindings**

The example described below is implemented in the example Python script generic_syntax_pytorch.py located in pykeops/examples. Python bindings that do not use
PyTorch are also implemented ; see the script generic_syntax_numpy.py for the corresponding example.

The Python reduction operations GenericSum.apply and GenericLogSumExp.apply require to input the defined variables and formula as character strings :

```python
aliases = ["p=Pm(0,1)","a=Vy(1,1)","x=Vx(2,3)","y=Vy(3,3)"]
formula = "Square(p-a)*Exp(x+y)"
```
Furthermore, one needs to input the following "signature" of the formula giving dimension and category (0 for i, 1 for j, 2 for $\emptyset$) for the output and each variable in the correct order (note that these could be infered from the aliases but as for now it is required, and enforce the user to be carefull with the sizes of inputs and output) :

```python
signature   =   [ (3, 0), (1, 2), (1, 1), (3, 0), (3, 1) ]
```
give (dimension,category) for the output c and variables p, a, x, y
Next one needs to specify wether reduction will be performed on the $j$ index with output variables indexed by $i$, or conversely. 

```python
sum_index   = 0		# 0 means summation over j, 1 means over i 
```

Finally one can call the reduction operation on PyTorch variables (torch.autograd.Variable) :

```python
c = GenericSum.apply("auto",aliases,formula,signature,sum_index,p,a,x,y)
```

where p, a, x and y correspond to PyTorch Variables with compatible sizes : in this example,

- p must be a scalar variable
- a must be a variable of size n*1, where n can be any positive integer,
- x must be a variable of size m*3, where m can be any positive integer,
- y must be a variable of size n*3.
- output c will be a variable of size m*3

If this formula has never been used by the user, the previous call will first perform an on-the-fly compilation of the required dll file, prior to the computation.

KeOps implements its own automatic differentiation of formulas, which allows to make it compatible with PyTorch autograd package. Given the above definition of variable c for example, one gets its gradient with respect to y with the call 

```python
grad(c,y,e)
```

where e is a new variable giving the input of the gradient (with same size as c).
Note that internally, KeOps infers the formula of the gradient, which will require a new compilation (performed again on-the-fly) if the user has never used it before.

**Using the syntax in the Matlab bindings**

The Matlab bindings provide a function Kernel which can be used to define the corresponding convolution operations. Following the previous example, one may write

```matlab
f = Kernel('p=Pm(0,1)','a=Vy(1,3)','x=Vx(2,3)','y=Vy(3,3)','Square(p-a)*Exp(x+y)');
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
