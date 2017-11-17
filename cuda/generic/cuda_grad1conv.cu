// nvcc -std=c++11 -Xcompiler -fPIC -shared -o cuda_grad1conv.so cuda_grad1conv.cu

#include "GpuConv2D.cu"
#include "CudaScalarRadialKernels.h"

#define CALL_SCALARRADIALKER(EVAL,ARGSCUDA,TYPE,DIMPOINT,DIMVECT) \
	if(dimPoint==DIMPOINT && dimVect==DIMVECT) \
	{ \
		typedef ScalarRadialKernel<TYPE,DIMPOINT,DIMVECT,RadialFun> Ker; \
		struct Ker::EVAL funeval; \
		return GpuConv2D(Ker(RadialFun(Sigma)),funeval, nx, ny, ARGSCUDA()); \
	}

// here we give the list of possible values for DIMVECT
#define CALL_SCALARRADIALKER_DIMVECT(EVAL,ARGSCUDA,TYPE,DIMPOINT) \
	CALL_SCALARRADIALKER(EVAL,ARGSCUDA,TYPE,DIMPOINT,1) \
	CALL_SCALARRADIALKER(EVAL,ARGSCUDA,TYPE,DIMPOINT,2) \
	CALL_SCALARRADIALKER(EVAL,ARGSCUDA,TYPE,DIMPOINT,3)
	
// here we give the list of possible values for DIMPOINT
#define CALL_SCALARRADIALKER_DIMPOINT_DIMVECT(EVAL,ARGSCUDA,TYPE) \
	CALL_SCALARRADIALKER_DIMVECT(EVAL,ARGSCUDA,TYPE,1) \
	CALL_SCALARRADIALKER_DIMVECT(EVAL,ARGSCUDA,TYPE,2) \
	CALL_SCALARRADIALKER_DIMVECT(EVAL,ARGSCUDA,TYPE,3)

#define DECLARE_EXTERNC_SCALARRADIAL(EVAL,ARGSC,ARGSCUDA,TYPE,FUNCNAME,RADFUNNAME) \
extern "C" int FUNCNAME(TYPE ooSigma2, ARGSC(TYPE), int dimPoint, int dimVect, int nx, int ny) \
{ \
	TYPE Sigma = sqrt(1/ooSigma2); \
	typedef RADFUNNAME<TYPE> RadialFun; \
	CALL_SCALARRADIALKER_DIMPOINT_DIMVECT(EVAL,ARGSCUDA,TYPE) \
	cout << "These dimensions are not implemented, but you just need to copy-paste one line and recompile." << endl; \
	return -1; \
}

#define GRAD1_ARGSCUDA()		gamma_h, alpha_h, x_h, y_h, beta_h
#define GRAD1_ARGSC(TYPE)		TYPE* alpha_h, TYPE* x_h, TYPE* y_h, TYPE* beta_h, TYPE* gamma_h

//  finally list of all convs
#define DECLARE_EXTERNC_SCALARRADIAL_RADFUN(TYPE) \
	DECLARE_EXTERNC_SCALARRADIAL(sGrad1,GRAD1_ARGSC,GRAD1_ARGSCUDA,TYPE,GaussGpuGrad1Conv,GaussFunction) \
	DECLARE_EXTERNC_SCALARRADIAL(sGrad1,GRAD1_ARGSC,GRAD1_ARGSCUDA,TYPE,CauchyGpuGrad1Conv,CauchyFunction) \
	DECLARE_EXTERNC_SCALARRADIAL(sGrad1,GRAD1_ARGSC,GRAD1_ARGSCUDA,TYPE,Sum4GaussGpuGrad1Conv,Sum4GaussFunction) \
	DECLARE_EXTERNC_SCALARRADIAL(sGrad1,GRAD1_ARGSC,GRAD1_ARGSCUDA,TYPE,Sum4CauchyGpuGrad1Conv,Sum4CauchyFunction) \
	DECLARE_EXTERNC_SCALARRADIAL(sGrad1,GRAD1_ARGSC,GRAD1_ARGSCUDA,TYPE,LaplaceGpuGrad1Conv,LaplaceFunction) \
	DECLARE_EXTERNC_SCALARRADIAL(sGrad1,GRAD1_ARGSC,GRAD1_ARGSCUDA,TYPE,EnergyGpuGrad1Conv,EnergyFunction)

#if (UseCudaOnDoubles) 
	DECLARE_EXTERNC_SCALARRADIAL_RADFUN(double)
#else
	DECLARE_EXTERNC_SCALARRADIAL_RADFUN(float)
#endif