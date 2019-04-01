![logo](./doc/_static/logo/keops_logo.png)


# What is KeOps?

KeOps is a library that computes on a GPU **generic reductions** of 2d arrays whose entries may be computed through a mathematical formula. We provide an autodiff engine to generate effortlessly the formula of the derivative. For instance, KeOps can compute **Kernel dot products** and **their derivatives**. 

A typical sample of (pseudo) code looks like

```python
from keops import Genred

# create the function computing the derivative of a Gaussian convolution
my_conv = Genred(reduction='Sum',
                 formula  ='Grad(Exp(-SqNorm2(x-y) / IntCst(2)), x, b)',
                 alias    =['x=Vi(3)', 'y=Vj(3)', 'b=Vi(3)'])

# ... apply it to the 2d array x, y, b with 3 columns and a (huge) number of lines
result = my_conv(x,y,b)
```

KeOps provides good performances and linear (instead of quadratic) memory footprint. It handles multi GPU. More details are provided here:

* [Installation](http://www.kernel-operations.io/keops/api/installation.html)
* [Documentation](http://www.kernel-operations.io/keops/)
* [Learning KeOps syntax with examples](http://www.kernel-operations.io/keops/_auto_examples/index.html)
* [Tutorials gallery](http://www.kernel-operations.io/keops/_auto_tutorials/index.html)


# Authors

- [Benjamin Charlier](http://imag.umontpellier.fr/~charlier/)
- [Jean Feydy](https://www.math.ens.fr/~feydy/)
- [Joan Alexis Glaunès](https://www.mi.parisdescartes.fr/~glaunes/)

