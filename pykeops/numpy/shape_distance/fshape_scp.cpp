#include <vector>
#include <string>
// #include "formula.h" made in cmake

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#if USE_CUDA
extern "C" {
    int cudafshape(__TYPE__ ooSigmax2,__TYPE__ ooSigmaf2, __TYPE__ ooSigmaXi2,
                   __TYPE__* x_h, __TYPE__* y_h,
                   __TYPE__* f_h, __TYPE__* g_h,
                   __TYPE__* alpha_h, __TYPE__* beta_h,
                   __TYPE__* gamma_h,
                   int dimPoint, int dimSig, int dimVect, int nx, int ny);

};
#endif

namespace pykeops {

using namespace keops;
namespace py = pybind11;

// <__TYPE__, py::array::c_style>  ensures 2 things whatever is the arguments:
//  1) the precision used is __TYPE__ (float or double typically) on the device,
//  2) everything is convert as contiguous before being loaded in memory
// this is maybe not the best in term of performance... but at least it is safe.
using __NUMPYARRAY__ = py::array_t<__TYPE__, py::array::c_style>;

/////////////////////////////////////////////////////////////////////////////////
//                          Main functions
/////////////////////////////////////////////////////////////////////////////////

__NUMPYARRAY__ specific_fshape_scp(py::array_t<__TYPE__> x_py,
                                   py::array_t<__TYPE__> y_py,
                                   py::array_t<__TYPE__> f_py,
                                   py::array_t<__TYPE__> g_py,
                                   py::array_t<__TYPE__> alpha_py,
                                   py::array_t<__TYPE__> beta_py,
                                   __TYPE__ sigmax,
                                   __TYPE__ sigmaf,
                                   __TYPE__ sigmaXi){

    // Get address of args
    auto x = py::cast< __NUMPYARRAY__ >(x_py);
    __TYPE__ * x_data = (__TYPE__ *) x.data();
    auto y = py::cast< __NUMPYARRAY__ >(y_py);
    __TYPE__ * y_data = (__TYPE__ *) y.data();
    auto f = py::cast< __NUMPYARRAY__ >(f_py);
    __TYPE__ * f_data = (__TYPE__ *) f.data();
    auto g = py::cast< __NUMPYARRAY__ >(g_py);
    __TYPE__ * g_data = (__TYPE__ *) g.data();
    auto alpha = py::cast< __NUMPYARRAY__ >(alpha_py);
    __TYPE__ * alpha_data = (__TYPE__ *) alpha.data();
    auto beta = py::cast< __NUMPYARRAY__ >(beta_py);
    __TYPE__ * beta_data = (__TYPE__ *) beta.data();

    __TYPE__ casted_sigmax = (__TYPE__) sigmax;
    __TYPE__ ooSigmax2 = 1 / (casted_sigmax * casted_sigmax);
    __TYPE__ casted_sigmaf = (__TYPE__) sigmaf;
    __TYPE__ ooSigmaf2 = 1 / (casted_sigmaf * casted_sigmaf);
    __TYPE__ casted_sigmaXi = (__TYPE__) sigmaXi;
    __TYPE__ ooSigmaXi2 = 1 / (casted_sigmaXi * casted_sigmaXi);

    // Get and check dimensions
    int nx = x.shape(0);
    int ny = y.shape(0);
    int dimPoint = x.shape(1);
    int dimSig = f.shape(1);
    int dimVect = alpha.shape(1);

    if (dimPoint != y.shape(1))
        throw std::runtime_error("[Keops] Wrong number of columns for y: is " + std::to_string(y.shape(1))
                + " but should be " + std::to_string(dimPoint) ) ;

    if (dimSig != g.shape(1))
        throw std::runtime_error("[Keops] Wrong number of columns for g: is " + std::to_string(g.shape(1))
                + " but should be " + std::to_string(dimSig) ) ;

    if (dimVect != beta.shape(1))
        throw std::runtime_error("[Keops] Wrong number of columns for beta: is " + std::to_string(beta.shape(1))
                + " but should be " + std::to_string(dimVect) ) ;

    if (nx != f.shape(0))
        throw std::runtime_error("[Keops] Wrong number of rows for f: is " + std::to_string(f.shape(0))
                + " but should be " + std::to_string(nx) ) ;

    if (nx != alpha.shape(0))
        throw std::runtime_error("[Keops] Wrong number of rows for alpha: is " + std::to_string(alpha.shape(0))
                + " but should be " + std::to_string(nx) ) ;

    if (ny != g.shape(0))
        throw std::runtime_error("[Keops] Wrong number of rows for g: is " + std::to_string(g.shape(0))
                + " but should be " + std::to_string(ny) ) ;

    if (ny != beta.shape(0))
        throw std::runtime_error("[Keops] Wrong number of rows for beta: is " + std::to_string(beta.shape(0))
                + " but should be " + std::to_string(ny) ) ;


    // Declare Output
    auto result_array = __NUMPYARRAY__({nx,dimSig});
    __TYPE__ * result_data = (__TYPE__ *) result_array.data();

    cudafshape(ooSigmax2, ooSigmaf2, ooSigmaXi2, x_data, y_data, f_data, g_data, alpha_data, beta_data, result_data, dimPoint, dimSig, dimVect, nx, ny);


    return result_array;
}


/////////////////////////////////////////////////////////////////////////////////
//                    PyBind11 entry point
/////////////////////////////////////////////////////////////////////////////////

// the following macro force the compiler to change MODULE_NAME to its value
#define VALUE_OF(x) x

PYBIND11_MODULE(VALUE_OF(MODULE_NAME_FSHAPE_SCP), m) {
    m.doc() = "keops specific routine io through pybind11"; // optional module docstring

    m.def("specific_fshape_scp",
          &specific_fshape_scp,
          "Entry point to keops - numpy version.");

}

}