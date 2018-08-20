#include <vector>
#include <string>
// #include "formula.h" made in cmake

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#if USE_CUDA
extern "C" {
    int GaussGpuEval(__TYPE__ ooSigma2, __TYPE__* x_h, __TYPE__* y_h, __TYPE__* beta_h, __TYPE__* gamma_h, int dimPoint, int dimVect, int nx, int ny);
    int LaplaceGpuEval(__TYPE__ ooSigma2, __TYPE__* x_h, __TYPE__* y_h, __TYPE__* beta_h, __TYPE__* gamma_h, int dimPoint, int dimVect, int nx, int ny);
    int InverseMultiquadricGpuEval(__TYPE__ ooSigma2, __TYPE__* x_h, __TYPE__* y_h, __TYPE__* beta_h, __TYPE__* gamma_h, int dimPoint, int dimVect, int nx, int ny);
    int CauchyGpuEval(__TYPE__ ooSigma2, __TYPE__* x_h, __TYPE__* y_h, __TYPE__* beta_h, __TYPE__* gamma_h, int dimPoint, int dimVect, int nx, int ny);
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

py::array_t< __TYPE__, py::array::c_style > specific_conv(py::array_t<__TYPE__> x_py,
                                                        py::array_t<__TYPE__> y_py,
                                                        py::array_t<__TYPE__> b_py,
                                                        __TYPE__ sigma,
                                                        std::string kernel){

    // Get address of args
    auto x = py::cast<__NUMPYARRAY__>(x_py);
    __TYPE__ * x_data = (__TYPE__ *) x.data();
    auto y = py::cast<__NUMPYARRAY__>(y_py);
    __TYPE__ * y_data = (__TYPE__ *) y.data();
    auto b = py::cast<__NUMPYARRAY__>(b_py);
    __TYPE__ * b_data = (__TYPE__ *) b.data();

    __TYPE__ casted_sigma = (__TYPE__) sigma;
    __TYPE__ ooSigma2 = 1 / (casted_sigma * casted_sigma);

    // Get and check dimensions
    int nx = x.shape(0);
    int ny = y.shape(0);
    int dimPoint = x.shape(1);
    int dimVect = b.shape(1);

    if (dimPoint != y.shape(1))
        throw std::runtime_error("[Keops] Wrong number of columns for y: is " + std::to_string(y.shape(1))
                + " but should be " + std::to_string(dimPoint) ) ;

    if (ny != b.shape(0))
        throw std::runtime_error("[Keops] Wrong number of rows for b: is " + std::to_string(b.shape(0))
                + " but should be " + std::to_string(ny) ) ;


    // Declare output
    auto result_array = __NUMPYARRAY__({nx,dimVect});
    __TYPE__ * result_data = (__TYPE__ *) result_array.data();

    if (kernel.compare("gaussian") == 0) {
        GaussGpuEval(ooSigma2, x_data, y_data, b_data, result_data, dimPoint, dimVect, nx, ny);
    } else if(kernel.compare("cauchy") == 0) {
        CauchyGpuEval(ooSigma2, x_data, y_data, b_data, result_data, dimPoint, dimVect, nx, ny);
    } else if(kernel.compare("laplacian") == 0) {
        LaplaceGpuEval(ooSigma2, x_data, y_data, b_data, result_data, dimPoint, dimVect, nx, ny);
    } else if(kernel.compare("inverse_multiquadric") == 0) {
        InverseMultiquadricGpuEval(ooSigma2, x_data, y_data, b_data, result_data, dimPoint, dimVect, nx, ny);
    } else {
        throw std::runtime_error("[Keops] Wrong kernel: is " + kernel
                + " but should be one of : gaussian, cauchy, laplacian, inverse_multiquadric") ;
    }

    return result_array;
}


/////////////////////////////////////////////////////////////////////////////////
//                    PyBind11 entry point
/////////////////////////////////////////////////////////////////////////////////

PYBIND11_MODULE(radial_kernel_conv, m) {
    m.doc() = "keops specific routine io through pybind11"; // optional module docstring

    m.def("specific_conv",
          &specific_conv,
          "Entry point to keops - numpy version.");

}

}
