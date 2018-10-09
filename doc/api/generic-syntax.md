# Supported formulas

## Variables category

KeOps uses a low-level syntax written in C++/Cuda to define virtually any reduction operation of the form

```math
\alpha_i = \text{Reduction}_j \big[ f(x^0_{\iota_0}, ... , x^{n-1}_{\iota_{n-1}})  \big]
```

where "Reduction" can be the Sum or the LogSumExp operation.


Each of the variables `$x^k_{\iota_k}$` is specified by its positional index `$k$`, its category `$\iota_k\in\{i,j,\emptyset\}$` (meaning that the variable is indexed by i, by j, or is a fixed parameter across indices) and its dimension `$d_k$`. These three characteristics are encoded as follows :

- category is given via the keywords "Vx", "Vy", "Pm" (meaning respectively: "variable indexed by i", "variable indexed by j", and "parameter")
- positional index `$k$` and dimension `$d_k$` are given as two integer parameters put into parenthesis after the category-specifier keyword.

For instance, `Vx(2,4)` specifies a variable indexed by "i", given as the third (k=2) input in the function call, and representing a vector of dimension 4.

Of course, using the same index `$k$` for two different variables is not allowed and will be rejected by the compiler.

## An example

Asume we want to compute the following sum

```math
f(x,y) = \left(\sum_j (p -y_j^u )^2 \exp(x_i^u + y_j^u) \right)_{u=1,2,3} \in \mathbb R^3
```

where `$p \in \mathbb R$` is a constant, `$ x \in \mathbb R^{N\times 3}$`, `$ y \in \mathbb R^{M\times 3}$`. From the "variables" symbolic placeholders, one can build the function `$f$` using the syntax 

```cpp
Square(Pm(0,1)-Vy(1,1))*Exp(Vx(2,3)+Vy(3,3))
```

in which `+` and `-` denote the usual addition of vectors, `Exp` is the (element-wise) exponential function and `*` denotes scalar-vector multiplication.

The operations available are listed [here](/math operations):

Variables can be given aliases, allowing us to write human-readable expressions for our formula. For example, one may define
`p=Pm(0,1)`, `a=Vy(1,1)`, `x=Vx(2,3)`, `y=Vy(3,3)`, and write the previous computation as

```cpp
Square(p-a)*Exp(x+y)
```
