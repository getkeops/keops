#include "CudaFunctions.h"


/* Implements a simple isotropic radial kernel on a vector space.
 * It is the class to use if you want to compute :
 *
 *   Gamma_i = sum_j f( |x_i - y_j|^2 ) b_j     and its derivatives,
 * 
 * where f is a RADIAL_FUN function, 
 * x_i and y_j are vectors of size DIMPOINT and 
 * b_j a vector of size DIMVECT.
 * 
 * Rfun is typically going to be a GaussFunction<float> or another class from CudaFunctions.h
 *
 * At the end of the file, if KERNEL==SCALARRADIAL, we define the KER operator whose methods
 * such as sEval, etc., can be plugged into a GpuConv2D or GpuConv1D 
*/
template < typename TYPE, int DIMPOINT, int DIMVECT, class RADIAL_FUN >
class ScalarRadialKernel {

    RADIAL_FUN Rfun;

  public:
    
    // It is not awfully elegant... But even though our desired kernel type is already specified
    // when we do templating (in the type RADIAL_FUN), we have to give explicitely the
    // radial function (which is just going to be RADIAL_FUN<__TYPE__>() ) when creating the operator.
    ScalarRadialKernel(RADIAL_FUN rfun) {
        Rfun = rfun;
    }

    // Order 0 operation : computing the sum  -------------------------------------------------------------------------
    //                           Gamma_i = \sum_j f( |x_i - y_j|^2 ) b_j
    //
    // ScalarRadialKernel::sEval is going to be used as a function; it defines the data packed types,
    // and calls Eval in the background.
    struct sEval {
        typedef pack<DIMVECT,DIMPOINT> DIMSX; // Output = x1i = VECT,  x_i = x2i = POINT 
        typedef pack<DIMPOINT,DIMVECT> DIMSY; //    y_j = y1j = POINT, b_j = y2j = VECT
        __host__ __device__ __forceinline__ void operator()(ScalarRadialKernel& ker, TYPE* gammai, TYPE* xi, TYPE* yj, TYPE* betaj) {
            ker.Eval(gammai,xi,yj,betaj);
        }
    };
    // Eval is the atomic evaluation operation, called in the loop, 
    // which increments *gammai by an amount Rfun.Eval( | x_i-y_j |_2^2 ) * betaj
    __host__ __device__ __forceinline__ void Eval(TYPE* gammai, TYPE* xi, TYPE* yj, TYPE* betaj) {
        TYPE r2 = 0.0f;                 // Don't forget to initialize at 0.0
        TYPE temp;
        for(int k=0; k<DIMPOINT; k++) { // Compute the L2 squared distance r2 = | x_i-y_j |_2^2
            temp =  yj[k]-xi[k];
            r2 += temp*temp;
        }
        TYPE s = Rfun.Eval(r2);         // Apply the (scalar) kernel function
        for(int k=0; k<DIMVECT; k++)    // Increment the output vector gammai - which is a VECTOR
            gammai[k] += s * betaj[k];
    }

    // Order 1 operation : computing the gradient wrt x -------------------------------------------------------------------------
    //       \partial_x < a, Eval(x, y, b) >  = \partial_x \sum_i <a_i, sum_j f( |x_i - y_j|^2 ) b_j >
    //                                        = \partial_x \sum_i \sum_j <a_i, b_j> * f( |x_i - y_j|^2 )
    //
    //       represented by a vector Gamma_i = sum_j 2*<a_i, b_j>*f'( |x_i - y_j|^2 ) * (x_i - y_j)
    //
    // ScalarRadialKernel::sGrad1 is going to be used as a function; it defines the data packed types,
    // and calls Grad1 in the background.
    struct sGrad1 {
        typedef pack<DIMPOINT,DIMVECT,DIMPOINT> DIMSX;
        typedef pack<DIMPOINT,DIMVECT> DIMSY;
        __host__ __device__ __forceinline__ void operator()(ScalarRadialKernel& ker, TYPE* gammai, TYPE* alphai, TYPE* xi, TYPE* yj, TYPE* betaj) {
            ker.Grad1(gammai, alphai, xi, yj, betaj);
        }
    };
    // Grad1 is the atomic evaluation operation, called in the loop, 
    // which increments *gammai by an amount 2*<a_i, b_j>*f'( |x_i - y_j|^2 ) * (x_i - y_j)
    __host__ __device__ __forceinline__ void Grad1(TYPE* gammai, TYPE* alphai, TYPE* xi, TYPE* yj, TYPE* betaj) {
        TYPE r2 = 0.0f, sga = 0.0f;          // Don't forget to initialize at 0.0
        TYPE xmy[DIMPOINT];
        for(int k=0; k<DIMPOINT; k++) {      // Compute the L2 squared distance r2 = | x_i-y_j |_2^2
            xmy[k] =  xi[k]-yj[k];
            r2 += xmy[k]*xmy[k];
        }
        for(int k=0; k<DIMVECT; k++)         // Compute the L2 dot product <a_i, b_j>
            sga += betaj[k]*alphai[k];
        TYPE s = 2.0 * sga * Rfun.Diff(r2);  // Don't forget the 2 !
        for(int k=0; k<DIMPOINT; k++)        // Increment the output vector gammai - which is a POINT
            gammai[k] += s * xmy[k];
    }

    struct sGrad {
        typedef pack<DIMPOINT,DIMVECT,DIMPOINT,DIMVECT> DIMSX;
        typedef pack<DIMVECT,DIMPOINT,DIMVECT> DIMSY;
        __host__ __device__ __forceinline__ void operator()(ScalarRadialKernel& ker, TYPE* gammai, TYPE* alphai, TYPE* xi, TYPE* betai, TYPE* alphaj, TYPE* xj, TYPE* betaj) {
            ker.Grad(gammai, alphai, xi, betai, alphaj, xj, betaj);
        }
    };

    __host__ __device__ __forceinline__ void Grad(TYPE* gammai, TYPE* alphai, TYPE* xi, TYPE* betai, TYPE* alphaj, TYPE* xj, TYPE* betaj) {
        TYPE r2 = 0.0f, sga = 0.0f;           // Don't forget to initialize at 0.0
        TYPE ximxj[DIMPOINT];
        for(int k=0; k<DIMPOINT; k++) {
            ximxj[k] =  xi[k]-xj[k];
            r2 += ximxj[k]*ximxj[k];
        }
        for(int k=0; k<DIMVECT; k++)
            sga += betaj[k]*alphai[k] + betai[k]*alphaj[k];
        TYPE s = 2.0 * sga * Rfun.Diff(r2);
        for(int k=0; k<DIMPOINT; k++)
            gammai[k] += s * ximxj[k];
    }

    struct sHess {
        typedef pack<DIMPOINT,DIMPOINT,DIMVECT,DIMPOINT> DIMSX;
        typedef pack<DIMPOINT,DIMVECT,DIMPOINT> DIMSY;
        __host__ __device__ __forceinline__ void operator()(ScalarRadialKernel& ker, TYPE* gammai, TYPE* xi, TYPE* betai, TYPE* etai, TYPE* xj, TYPE* betaj, TYPE* etaj) {
            ker.Hess(gammai, xi, betai, etai, xj, betaj, etaj);
        }
    };

    __host__ __device__ __forceinline__ void Hess(TYPE* gammai, TYPE* xi, TYPE* betai, TYPE* etai, TYPE* xj, TYPE* betaj, TYPE* etaj) {
        TYPE r2 = 0.0f, bidbj = 0.0f, dotex = 0.0f;
        TYPE ximxj[DIMPOINT], eimej[DIMPOINT];
        for(int k=0; k<DIMPOINT; k++) {
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

    struct sDiff {
        typedef pack<DIMVECT,DIMPOINT,DIMPOINT> DIMSX;
        typedef pack<DIMPOINT,DIMPOINT,DIMVECT> DIMSY;
        __host__ __device__ __forceinline__ void operator()(ScalarRadialKernel& ker, TYPE* gammai, TYPE* xi, TYPE* etai, TYPE* xj, TYPE* etaj, TYPE* betaj) {
            ker.Diff(gammai, xi, etai, xj, etaj, betaj);
        }
    };

    __host__ __device__ __forceinline__ void Diff(TYPE* gammai, TYPE* xi, TYPE* etai, TYPE* xj, TYPE* etaj, TYPE* betaj) {
        TYPE r2 = 0.0f, dotex = 0.0f;
        TYPE ximxj[DIMPOINT], eimej[DIMPOINT];
        for(int k=0; k<DIMPOINT; k++) {
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



