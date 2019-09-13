The files where the elementary operators are defined.

The core operators of our engine are :
*      Var<N,DIM,CAT>           : the N-th variable, a vector of dimension DIM,
                                  with CAT = 0 (i-variable), 1 (j-variable) or 2 (parameter)
*      Grad<F,V,GRADIN>         : gradient (in fact transpose of diff op) of F with respect to variable V, applied to GRADIN
*      _P<N>, or Param<N>       : the N-th parameter variable
*      _X<N,DIM>                : the N-th variable, vector of dimension DIM, CAT = 0
*      _Y<N,DIM>                : the N-th variable, vector of dimension DIM, CAT = 1

