# Supported math operations

Here is a list of the implemented operations that can be used in formulas:

```
a*b : scalar-vector multiplication (if a is scalar) or vector-vector element-wise multiplication
a+b : addition of two vectors
a-b : difference between two vectors or minus sign
a/b : element-wise division
(a|b) : scalar product between vectors
Exp(a) : element-wise exponential function
Log(a) : element-wise natural logarithm
Pow(a,N) : N-th power of a (element-wise), where N is a fixed-size integer
Pow(a,b) : power operation - alias for Exp(b*Log(a))
Square(a) : element-wise square
Grad(a,x,e) : gradient of a with respect to the variable x, with e as the "grad_output" to backpropagate
ConstInt : integer constant
Inv : element-wise inverse (1/b)
```