/*
 * The file where the elementary constants are defined.
 * Available constants are :
 * 
 *      Zero<DIM>					: zero-valued vector of dimension DIM
 *      IntConstant<N>				: constant integer function with value N
 *      Constant<PRM>				: constant function with value given by parameter PRM (ex : Constant<C> here)
 * 
 */
 
// A "zero" vector of size _DIM
// Declared using the   Zero<DIM>   syntax.
template < int _DIM >
struct Zero
{
    static const int DIM = _DIM;

    template < int CAT >      // Whatever CAT... 
    using VARS = univpack<>;  // there's no variable used in there.

    // Evaluation is easy : simply fill-up *out with zeros.
    template < class INDS, typename... ARGS >
    INLINE void Eval(__TYPE__* params, __TYPE__* out, ARGS... args)
    {
        for(int k=0; k<DIM; k++)
            out[k] = 0;
    }
    
    // There is no gradient to accumulate on V, whatever V.    
    template < class V, class GRADIN >
    using DiffT = Zero<V::DIM>;
};

// A constant integer value, defined using the     IntConstant<N>    syntax.
template < int N >
struct IntConstant
{
    static const int DIM = 1;

    template < int CAT >      // Whatever CAT... 
    using VARS = univpack<>;  // there's no variable used in there.
    
    // Evaluation is easy : simply fill *out = out[0] with N.
    template < class INDS, typename... ARGS >
    INLINE void Eval(__TYPE__* params, __TYPE__* out, ARGS... args)
    {   *out = N; }
        
    // There is no gradient to accumulate on V, whatever V.
    template < class V, class GRADIN >
    using DiffT = Zero<V::DIM>;
};


// A constant parameter value, a scalar (but we may use a pointer ?)
template < class PRM >
struct Constant
{
    static const int DIM = 1; // Scalar-valued parameters only.

    // A parameter is a variable of category "2" ( 0 = Xi, 1 = Yj )
    template < int CAT >
    using VARS = CondType<univpack<PRM>,univpack<>,CAT==2>;

    // "returns" the appropriate value in the params array.
    template < class INDS, typename... ARGS >
    INLINE void Eval(__TYPE__* params, __TYPE__* out, ARGS... args)
    {   *out = params[PRM::INDEX]; }
    
    // There's no gradient to accumulate in V, whatever V.    
    template < class V, class GRADIN >
    using DiffT = Zero<V::DIM>;
};
