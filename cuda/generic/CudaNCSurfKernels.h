
#include "CudaFunctions.h"

#define DIM 3

template < typename TYPE, class RADIAL_FUN >
class NCSurfKernel
{
    
    RADIAL_FUN Rfun;
    
public:
    
    NCSurfKernel(RADIAL_FUN rfun)
    {
        Rfun = rfun;
    }
    
    struct sSimple
    {
        typedef pack<DIM,DIM> DIMSX;
        typedef pack<DIM,DIM> DIMSY;
        __host__ __device__ __forceinline__ void operator()(NCSurfKernel& ker, TYPE* gammai, TYPE* xi, TYPE* yj, TYPE* betaj)
        {
            ker.Simple(gammai,xi,yj,betaj);
        }
    };
    
    __host__ __device__ __forceinline__ void Simple(TYPE* gammai, TYPE* xi, TYPE* yj, TYPE* betaj)
    {
        TYPE r2 = 0.0f;
        TYPE temp;
        for(int k=0; k<DIM; k++)
        {
            temp =  yj[k]-xi[k];
            r2 += temp*temp;
        }
        TYPE s = Rfun.Eval(r2);
        for(int k=0; k<DIM; k++)
            gammai[k] += s * betaj[k];
    }
    
    struct sScal
    {
        typedef pack<DIM,DIM,DIM> DIMSX;
        typedef pack<DIM,DIM,DIM> DIMSY;
        __host__ __device__ __forceinline__ void operator()
        (NCSurfKernel& ker, TYPE* gammai, TYPE* xi, TYPE* alphai, TYPE* yj, TYPE* betaj, TYPE* etaj)
        {
            ker.Scal(gammai, xi, alphai, yj, betaj, etaj);
        }
    };
    
   __host__ __device__ __forceinline__ void Scal( TYPE* gammai, TYPE* xi, TYPE* alphai, TYPE* yj, TYPE* betaj, TYPE* etaj)
    {
        TYPE r2 = 0.0f, sg = 0.0f;
        TYPE xmy;
        for(int k=0; k<DIM; k++)
        {
            xmy =  xi[k]-yj[k];
            r2 += xmy*xmy;
        }
        for(int k=0; k<DIM; k++)
            sg += betaj[k]*alphai[k];
        TYPE s = sg * Rfun.Eval(r2);
        for(int k=0; k<DIM; k++)
            gammai[k] += s * etaj[k];
    }
    
    struct sGradScal2
    {
        typedef pack<DIM,DIM,DIM,DIM> DIMSX;
        typedef pack<DIM,DIM,DIM> DIMSY;
        __host__ __device__ __forceinline__ void operator()
        (NCSurfKernel& ker, TYPE* gammai, TYPE* xi, TYPE* alphai, TYPE* betai, TYPE* yj, TYPE* alphaj, TYPE* betaj)
        {
            ker.GradScal2(gammai, xi, alphai, betai, yj, alphaj, betaj);
        }
    };
    
    __host__ __device__ __forceinline__ void GradScal2(TYPE* gammai, TYPE* xi, TYPE* alphai, TYPE* betai, TYPE* yj, TYPE* alphaj, TYPE* betaj)
    {
        TYPE r2 = 0.0f, sga = 0.0f, sgb = 0.0f;
        TYPE ximyj[DIM];
        for(int k=0; k<DIM; k++)
        {
            ximyj[k] =  xi[k]-yj[k];
            r2 += ximyj[k]*ximyj[k];
        }
        for(int k=0; k<DIM; k++)
        {
            sga += alphai[k]*alphaj[k];
            sgb += betai[k]*betaj[k];
        }
        TYPE s = 2.0 * sga * sgb * Rfun.Diff(r2);
        for(int k=0; k<DIM; k++)
            gammai[k] += s * ximyj[k];
    }
    
    
    
    struct sGradScal
    {
        typedef pack<DIM,DIM,DIM> DIMSX;
        typedef pack<DIM,DIM> DIMSY;
        __host__ __device__ __forceinline__ void operator()
        (NCSurfKernel& ker, TYPE* gammai, TYPE* xi, TYPE* alphai, TYPE* yj, TYPE* alphaj)
        {
            ker.GradScal(gammai, xi, alphai, yj, alphaj);
        }
    };
    
    __host__ __device__ __forceinline__ void GradScal(TYPE* gammai, TYPE* xi, TYPE* alphai, TYPE* yj, TYPE* alphaj)
    {
        TYPE r2 = 0.0f, sga = 0.0f;
        TYPE ximyj[DIM];
        for(int k=0; k<DIM; k++)
        {
            ximyj[k] =  xi[k]-yj[k];
            r2 += ximyj[k]*ximyj[k];
        }
        for(int k=0; k<DIM; k++)
            sga += alphai[k]*alphaj[k];
        TYPE s = 2.0 * sga * Rfun.Diff(r2);
        for(int k=0; k<DIM; k++)
            gammai[k] += s * ximyj[k];
    }
    
    
    

    
    
    
    struct sGradCyl
    {
        typedef pack<2*DIM,DIM,DIM,DIM,DIM,DIM,DIM,DIM> DIMSX;
        typedef pack<DIM,DIM,DIM> DIMSY;
        __host__ __device__ __forceinline__ void operator()
        (NCSurfKernel& ker, TYPE* gammai, TYPE* xi1, TYPE* xi2, TYPE* xi3, TYPE* alpha1i, TYPE* alpha2i, TYPE* alpha3i,
                                   TYPE* etai, 
                                   TYPE* yj, TYPE* betaj, TYPE* nuj)
        {
            ker.GradCyl(gammai, xi1, xi2, xi3, alpha1i, alpha2i, alpha3i, etai, yj, betaj, nuj);
        }
    };
    
    __host__ __device__ __forceinline__ void cross(TYPE* gamma, TYPE* alpha, TYPE* beta)
    {
        gamma[0] = alpha[1]*beta[2]-alpha[2]*beta[1];
        gamma[1] = alpha[2]*beta[0]-alpha[0]*beta[2];
        gamma[2] = alpha[0]*beta[1]-alpha[1]*beta[0];
    }

    __host__ __device__ __forceinline__ void GradCyl
    (TYPE* gammai, TYPE* xi1, TYPE* xi2, TYPE* xi3, TYPE* alpha1i, TYPE* alpha2i, TYPE* alpha3i,
     TYPE* etai, 
     TYPE* yj, TYPE* betaj, TYPE* nuj)
    {
        TYPE pr[DIM], cr1[DIM], cr2[DIM];
        TYPE s = 0.0f;
        for(int k=0; k<DIM; k++)
            s += etai[k]*betaj[k];
        for(int k=0; k<DIM; k++)
            pr[k] = betaj[k] - s*etai[k];
        cross(cr1,alpha1i,pr);
        cross(cr2,alpha2i,pr);
        TYPE r2 = 0.0f, sg = 0.0f;
        TYPE xmy;
        for(int k=0; k<DIM; k++)
        {
            xmy =  xi1[k]-yj[k];
            r2 += xmy*xmy;
        }
        for(int k=0; k<DIM; k++)
            sg += alpha1i[k]*nuj[k];
        s = sg * Rfun.Eval(r2);
        r2 = 0.0f; sg = 0.0f;
        for(int k=0; k<DIM; k++)
        {
            xmy =  xi2[k]-yj[k];
            r2 += xmy*xmy;
        }
        for(int k=0; k<DIM; k++)
            sg += alpha2i[k]*nuj[k];
        s += sg * Rfun.Eval(r2);
        r2 = 0.0f; sg = 0.0f;
        for(int k=0; k<DIM; k++)
        {
            xmy =  xi3[k]-yj[k];
            r2 += xmy*xmy;
        }
        for(int k=0; k<DIM; k++)
            sg += alpha3i[k]*nuj[k];
        s += sg * Rfun.Eval(r2);
        for(int k=0; k<DIM; k++)
        {
            gammai[k] -= s*cr2[k];
            gammai[k+DIM] -= s*cr1[k];
        }
    }
    
    struct sGradSph
    {
        typedef pack<DIM,DIM,DIM,DIM> DIMSX;
        typedef pack<DIM,DIM> DIMSY;
        __host__ __device__ __forceinline__ void operator()
        (NCSurfKernel& ker, TYPE* gammai, TYPE* xi1, TYPE* xi2, TYPE* etai, TYPE* yj, TYPE* betaj)
        {
            ker.GradSph(gammai, xi1, xi2, etai, yj, betaj);
        }
    };
    
    __host__ __device__ __forceinline__ void GradSph
    (TYPE* gammai, TYPE* xi1, TYPE* xi2, TYPE* etai, TYPE* yj, TYPE* betaj)
    {
        TYPE pr[DIM];
        TYPE s = 0.0f;
        for(int k=0; k<DIM; k++)
            s += etai[k]*betaj[k];
        for(int k=0; k<DIM; k++)
            pr[k] = betaj[k] - s*etai[k];
        TYPE r2 = 0.0f;
        TYPE xmy;
        for(int k=0; k<DIM; k++)
        {
            xmy =  xi1[k]-yj[k];
            r2 += xmy*xmy;
        }
        s = Rfun.Eval(r2);
        r2 = 0.0f;
        for(int k=0; k<DIM; k++)
        {
            xmy =  xi2[k]-yj[k];
            r2 += xmy*xmy;
        }
        s -= Rfun.Eval(r2);
        for(int k=0; k<DIM; k++)
        	gammai[k] += s*pr[k];
    }
    
};

#define NCSURF 2
#if KERNEL==NCSURF
typedef NCSurfKernel<__TYPE__,RADIALFUN<__TYPE__ >> KER;
#endif




