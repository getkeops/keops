The files where the complex math operators are defined.

Available complex operations are :
*      ComplexReal<F>                 : Real part of complex (vectorized)
*      ComplexImag<F>                 : Imaginary part of complex (vectorized)
*      Real2Complex<F>                : convert real vector to complex vector with zero imaginary part (F+0*i)
*      Imag2Complex<F>                : convert real vector to complex vector with zero real part (0+i*F)
*      Conj<F>                        : Complex conjugate (vectorized)
*      ComplexAbs<F>                  : Absolute value or modulus of complex (vectorized)
*      ComplexSquareAbs<F>            : Square of modulus of complex (vectorized)
*      ComplexAngle<F>                : Angle of complex (vectorized)
*      ComplexSum<F>                  : Sum of complex vector
*      ComplexSumT<F,DIM>             : Adjoint operation of ComplexSum - replicates F (complex scalar) DIM times
*      ComplexMult<F,G>               : Complex multiplication of F and G (vectorized)
*      ComplexScal<F,G>               : Multiplication of F (complex scalar) with G (complex vector)
*      ComplexRealScal<F,G>           : Multiplication of F (real scalar) with G (complex vector)
*      ComplexDivide<F,G>             : Complex division of F and G (vectorized)

Standard math functions involving complex numbers :   
*      ComplexExp<F>                  : Complex exponential (vectorized)
*      ComplexExp1j<F>                : Computes Exp(1j*F) where F is real (vectorized)
