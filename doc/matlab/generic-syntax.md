# Matlab API

The example described below is implemented in the example Matlab script script_generic_syntax.m located in keopslab/examples. 

The Matlab bindings provide a function `Kernel` which can be used to define the corresponding convolution operations. Following the previous example, one may write

```matlab
f = Kernel('Square(p-a)*Exp(x+y)','p=Pm(1)','a=Vy(1)','x=Vx(3)','y=Vy(3)');
```
which defines a Matlab function handler `f` which can be used to perform a sum reduction for this formula:

```matlab
c = f(p,a,x,y);
```

where `p`, `a`, `x`, `y` must be arrays with compatible dimensions as previously explained. A gradient function `GradKernel` is also provided. For example, to get the gradient with respect to `y` of the previously defined function `f`, one needs to write :

```matlab
Gfy = GradKernel(f,'y','e=Vx(3)');
```

which returns a new function that can be used as follows :

```matlab
Gfy(p,a,x,y,e)
```

where `e` is the input gradient array.