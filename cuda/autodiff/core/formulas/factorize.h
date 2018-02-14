#ifndef FACTORIZE
#define FACTORIZE

//////////////////////////////////////////////////////////////
////      FACTORIZE OPERATOR  : Factorize< F,G >          ////
//////////////////////////////////////////////////////////////

// Factorize< F,G > is the same as F, but when evaluating we factorize
// the computation of G, meaning that if G appears several times inside the
// formula F, we will compute it once only

template < class F, class G > struct FactorizeAlias;
template < class F, class G > using Factorize = typename FactorizeAlias<F,G>::type;

template < class F, class G >
struct FactorizeImpl : BinaryOp<FactorizeImpl,F,G>
{

    static const int DIM = F::DIM;
    
    static void PrintId() {
    	using IndsTempVars = GetInds<typename F::template VARS<3>>;
    	static const int dummyPos = 1+IndsTempVars::MAX;
    	using dummyVar = Var<dummyPos,G::DIM,3>;
    	using Ffact = typename F::template Replace<G,dummyVar>;
        cout << "[";
        dummyVar::PrintId();
        cout << "=";
        G::PrintId();
        cout << ";";
        Ffact::PrintId();
        cout << "]";
    }

    using THIS = FactorizeImpl<F,G>;    

    using Factor = G;

    // we define a new formula from F (called factorized formula), replacing G inside by a new variable ; this is used in function Eval()
    template < class INDS >
    //using FactorizedFormula = typename F::template Replace<G,Var<INDS::MAX+1,G::DIM,3>>;	// means replace G by Var<INDS::SIZE,G::DIM,3> in formula F
    using FactorizedFormula = Replace<F,G,Var<INDS::MAX+1,G::DIM,3>>;	// means replace G by Var<INDS::SIZE,G::DIM,3> in formula F

    template < class INDS, typename ...ARGS >
    static HOST_DEVICE INLINE void Eval(__TYPE__* params, __TYPE__* out, ARGS... args) {
		// First we compute G
		__TYPE__ outG[G::DIM];
		G::template Eval<INDS>(params,outG,args...);
		// Ffact is the factorized formula
		using Ffact = typename THIS::template FactorizedFormula<INDS>;
		// new indices for the call to Eval : we add one more index to the list
		using NEWINDS = ConcatPacks<INDS,pack<INDS::MAX+1>>;
		// call to Eval on the factorized formula, we pass outG as last parameter
		Ffact::template Eval<NEWINDS>(params,out,args...,outG);
    }
    
    template < class V, class GRADIN >
    using DiffT = Factorize<typename F::template DiffT<V,GRADIN>,G>;
    
};

// specializing CountInSub
template<class F, class G, class H>
struct CountInSub<FactorizeImpl<F,G>,H> {
    static const int val = CountIn<F,H>::val - CountIn<G,H>::val * CountIn<F,G>::val + CountIn<G,H>::val;
};

// allow factorization only if subformula appears at least twice in the formula
template < class F, class G >
using CondFactorize = CondType<FactorizeImpl<F,G>,F,(CountIn<F,G>::val > 1)>;


template < class F, class G >
struct FactorizeAlias {
    using type = CondFactorize<F,G>;
};

// specialization in case G is of type Var : in this case there is no need for copying a Var into another Var,
// so we replace Factorize<F,Var> simply by F. This is usefull to avoid factorizing several times the same sub-formula
template < class F, int N, int DIM, int CAT >
struct FactorizeAlias<F,Var<N,DIM,CAT>> {
    using type = F;
};

// specialization in case G is of type IntConstant : again such a factorization is not interesting
template < class F, int N >
struct FactorizeAlias<F,IntConstant<N>> {
    using type = F;
};

// specialization in case G is of type Constant<Param<N>> : not interesting either
template < class F, int N >
struct FactorizeAlias<F,Constant<Param<N>>> {
    using type = F;
};

// specialization in case G = F : not interesting either
template < class F >
struct FactorizeAlias<F,F> {
    using type = F;
};



// factorize several times given a univpack of subformulas :

// default : no factorization (termination case)
template < class F, class PACK >
struct FactorizeList {
    using type = F;
};

// then specialization when there is at least one element in the pack
template < class F, class G, class... GS >
struct FactorizeList<F,univpack<G,GS...>> {
    using type = typename FactorizeList<Factorize<F,G>,univpack<GS...>>::type;
};

// Auto factorization : factorize F by each of its subformulas
template < class F >
using AutoFactorize = typename FactorizeList<F,typename F::AllTypes>::type;

#endif // FACTORIZE
