
#include "CudaFunctions.h"

#define DIM 3

template < typename TYPE, class RADIAL_FUN >
class VarSurfKernel {

    RADIAL_FUN Rfun;

  public:

    VarSurfKernel(RADIAL_FUN rfun) {
        Rfun = rfun;
    }

    struct sScal {
        typedef pack<DIM,DIM,DIM> DIMSX;
        typedef pack<DIM,DIM,DIM> DIMSY;
        __host__ __device__ __forceinline__ void operator()
        (VarSurfKernel& ker, TYPE* gammai, TYPE* xi, TYPE* alphai, TYPE* yj, TYPE* betaj, TYPE* etaj) {
            ker.Scal(gammai, xi, alphai, yj, betaj, etaj);
        }
    };

    __host__ __device__ __forceinline__ void Scal( TYPE* gammai, TYPE* xi, TYPE* alphai, TYPE* yj, TYPE* betaj, TYPE* etaj) {
        TYPE r2 = 0.0f, sg = 0.0f;
        TYPE xmy;
        for(int k=0; k<DIM; k++) {
            xmy =  xi[k]-yj[k];
            r2 += xmy*xmy;
        }
        for(int k=0; k<DIM; k++)
            sg += betaj[k]*alphai[k];
        TYPE s = sg * Rfun.Eval(r2);
        for(int k=0; k<DIM; k++)
            gammai[k] += s * etaj[k];
    }

    struct sSqScal {
        typedef pack<1,DIM,DIM> DIMSX;
        typedef pack<DIM,DIM> DIMSY;
        __host__ __device__ __forceinline__ void operator()
        (VarSurfKernel& ker, TYPE* gammai, TYPE* xi, TYPE* alphai, TYPE* yj, TYPE* betaj) {
            ker.SqScal(gammai, xi, alphai, yj, betaj);
        }
    };

    __host__ __device__ __forceinline__ void SqScal(TYPE* gammai, TYPE* xi, TYPE* alphai, TYPE* yj, TYPE* betaj) {
        TYPE r2 = 0.0f, sg = 0.0f;
        TYPE ximyj;
        for(int k=0; k<DIM; k++) {
            ximyj =  xi[k]-yj[k];
            r2 += ximyj*ximyj;
            sg += alphai[k]*betaj[k];
        }
        *gammai += sg * sg * Rfun.Eval(r2);
    }

    struct sGradSqScal {
        typedef pack<DIM,DIM,DIM> DIMSX;
        typedef pack<DIM,DIM> DIMSY;
        __host__ __device__ __forceinline__ void operator()
        (VarSurfKernel& ker, TYPE* gammai, TYPE* xi, TYPE* alphai, TYPE* yj, TYPE* betaj) {
            ker.GradSqScal(gammai, xi, alphai, yj, betaj);
        }
    };

    __host__ __device__ __forceinline__ void GradSqScal(TYPE* gammai, TYPE* xi, TYPE* alphai, TYPE* yj, TYPE* betaj) {
        TYPE r2 = 0.0f, sg = 0.0f;
        TYPE ximyj[DIM];
        for(int k=0; k<DIM; k++) {
            ximyj[k] =  xi[k]-yj[k];
            r2 += ximyj[k]*ximyj[k];
        }
        for(int k=0; k<DIM; k++)
            sg += alphai[k]*betaj[k];
        TYPE s = 2 * sg * sg * Rfun.Diff(r2);
        for(int k=0; k<DIM; k++)
            gammai[k] += s * ximyj[k];
    }



};

#define VARSURF 3
#if KERNEL==VARSURF
typedef VarSurfKernel<__TYPE__,RADIALFUN<__TYPE__ >> KER;
#endif





