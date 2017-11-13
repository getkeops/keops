#include "CudaFunctions.h"

template < typename TYPE, int DIMPOINT, int DIMVECT, class RADIAL_FUN >
class ScalarRadialKernel
{
    
    RADIAL_FUN Rfun;
    
public:

    ScalarRadialKernel(RADIAL_FUN rfun)
    {
        Rfun = rfun;
    }
    
    struct sEval
    {
        typedef pack<DIMVECT,DIMPOINT> DIMSX;
        typedef pack<DIMPOINT,DIMVECT> DIMSY;
        __device__ __forceinline__ void operator()(ScalarRadialKernel& ker, TYPE* gammai, TYPE* xi, TYPE* yj, TYPE* betaj)
        {
            ker.Eval(gammai,xi,yj,betaj);
        }
    };
    
    __device__ __forceinline__ void Eval(TYPE* gammai, TYPE* xi, TYPE* yj, TYPE* betaj)
    {
        TYPE r2 = 0.0f;
        TYPE temp;
        for(int k=0; k<DIMPOINT; k++)
        {
            temp =  yj[k]-xi[k];
            r2 += temp*temp;
        }
        TYPE s = Rfun.Eval(r2);
        for(int k=0; k<DIMVECT; k++)
            gammai[k] += s * betaj[k];
    }
    
    struct sGrad1
    {
        typedef pack<DIMPOINT,DIMVECT,DIMPOINT> DIMSX;
        typedef pack<DIMPOINT,DIMVECT> DIMSY;
        __device__ __forceinline__ void operator()(ScalarRadialKernel& ker, TYPE* gammai, TYPE* alphai, TYPE* xi, TYPE* yj, TYPE* betaj)
        {
            ker.Grad1(gammai, alphai, xi, yj, betaj);
        }
    };
    
   __device__ __forceinline__ void Grad1(TYPE* gammai, TYPE* alphai, TYPE* xi, TYPE* yj, TYPE* betaj)
    {
        TYPE r2 = 0.0f, sga = 0.0f;
        TYPE xmy[DIMPOINT];
        for(int k=0; k<DIMPOINT; k++)
        {
            xmy[k] =  xi[k]-yj[k];
            r2 += xmy[k]*xmy[k];
        }
        for(int k=0; k<DIMVECT; k++)
            sga += betaj[k]*alphai[k];
        TYPE s = 2.0 * sga * Rfun.Diff(r2);
        for(int k=0; k<DIMPOINT; k++)
            gammai[k] += s * xmy[k];
    }
    
    struct sGrad
    {
        typedef pack<DIMPOINT,DIMVECT,DIMPOINT,DIMVECT> DIMSX;
        typedef pack<DIMVECT,DIMPOINT,DIMVECT> DIMSY;
        __device__ __forceinline__ void operator()(ScalarRadialKernel& ker, TYPE* gammai, TYPE* alphai, TYPE* xi, TYPE* betai, TYPE* alphaj, TYPE* xj, TYPE* betaj)
        {
            ker.Grad(gammai, alphai, xi, betai, alphaj, xj, betaj);
        }
    };
    
    __device__ __forceinline__ void Grad(TYPE* gammai, TYPE* alphai, TYPE* xi, TYPE* betai, TYPE* alphaj, TYPE* xj, TYPE* betaj)
    {
        TYPE r2 = 0.0f, sga = 0.0f;
        TYPE ximxj[DIMPOINT];
        for(int k=0; k<DIMPOINT; k++)
        {
            ximxj[k] =  xi[k]-xj[k];
            r2 += ximxj[k]*ximxj[k];
        }
        for(int k=0; k<DIMVECT; k++)
            sga += betaj[k]*alphai[k] + betai[k]*alphaj[k];
        TYPE s = 2.0 * sga * Rfun.Diff(r2);
        for(int k=0; k<DIMPOINT; k++)
            gammai[k] += s * ximxj[k];
    }
    
    struct sHess
    {
        typedef pack<DIMPOINT,DIMPOINT,DIMVECT,DIMPOINT> DIMSX;
        typedef pack<DIMPOINT,DIMVECT,DIMPOINT> DIMSY;
        __device__ __forceinline__ void operator()(ScalarRadialKernel& ker, TYPE* gammai, TYPE* xi, TYPE* betai, TYPE* etai, TYPE* xj, TYPE* betaj, TYPE* etaj)
        {
            ker.Hess(gammai, xi, betai, etai, xj, betaj, etaj);
        }
    };
    
   __device__ __forceinline__ void Hess(TYPE* gammai, TYPE* xi, TYPE* betai, TYPE* etai, TYPE* xj, TYPE* betaj, TYPE* etaj)
    {
        TYPE r2 = 0.0f, bidbj = 0.0f, dotex = 0.0f;
        TYPE ximxj[DIMPOINT], eimej[DIMPOINT];
        for(int k=0; k<DIMPOINT; k++)
        {
            ximxj[k] =  xi[k]-xj[k];
            r2 += ximxj[k]*ximxj[k];
            eimej[k] =  etai[k]-etaj[k];
            dotex += ximxj[k]*eimej[k];
        }
        for(int k=0; k<DIMVECT; k++)
            bidbj += betai[k]*betaj[k];
        TYPE d1, d2;
        Rfun.DiffDiff2(r2,&d1,&d2);
        d1 *= 4 * bidbj;
        d2 *= 8 * bidbj * dotex;
        for(int k=0; k<DIMPOINT; k++)
            gammai[k] += d1 * eimej[k] + d2 * ximxj[k];
    }
    
    struct sDiff
    {
        typedef pack<DIMVECT,DIMPOINT,DIMPOINT> DIMSX;
        typedef pack<DIMPOINT,DIMPOINT,DIMVECT> DIMSY;
        __device__ __forceinline__ void operator()(ScalarRadialKernel& ker, TYPE* gammai, TYPE* xi, TYPE* etai, TYPE* xj, TYPE* etaj, TYPE* betaj)
        {
            ker.Diff(gammai, xi, etai, xj, etaj, betaj);
        }
    };
    
    __device__ __forceinline__ void Diff(TYPE* gammai, TYPE* xi, TYPE* etai, TYPE* xj, TYPE* etaj, TYPE* betaj)
    {
        TYPE r2 = 0.0f, dotex = 0.0f;
        TYPE ximxj[DIMPOINT], eimej[DIMPOINT];
        for(int k=0; k<DIMPOINT; k++)
        {
            ximxj[k] =  xi[k]-xj[k];
            r2 += ximxj[k]*ximxj[k];
            eimej[k] =  etai[k]-etaj[k];
            dotex += ximxj[k]*eimej[k];
        }
        TYPE s = Rfun.Diff(r2) * 2.0 * dotex;
        for(int k=0; k<DIMVECT; k++)
            gammai[k] += s * betaj[k];
    }
};

#define SCALARRADIAL 1
#if KERNEL==SCALARRADIAL
typedef ScalarRadialKernel<__TYPE__,__DIMPOINT__,__DIMVECT__,RADIALFUN<__TYPE__ > > KER;
#endif



