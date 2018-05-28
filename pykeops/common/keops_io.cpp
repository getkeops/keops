#include </home/bcharlier/src/pybind11/include/pybind11/pybind11.h>
#include </home/bcharlier/src/pybind11/include/pybind11/numpy.h>

#include <vector>
#include "formula.h"

#include "core/Pack.h"
#include "core/autodiff.h"

namespace py = pybind11;

extern "C" int CpuConv(int, int, __TYPE__*, __TYPE__**);
extern "C" int CpuTransConv(int, int, __TYPE__*, __TYPE__**);

#ifdef USE_CUDA
extern "C" int GpuConv1D(int, int, __TYPE__*, __TYPE__**);
extern "C" int GpuConv1D_FromDevice(int, int, __TYPE__*, __TYPE__**);
extern "C" int GpuConv2D(int, int, __TYPE__*, __TYPE__**);
extern "C" int GpuConv2D_FromDevice(int, int, __TYPE__*, __TYPE__**);
extern "C" int GpuTransConv1D(int, int, __TYPE__*, __TYPE__**);
extern "C" int GpuTransConv1D_FromDevice(int, int, __TYPE__*, __TYPE__**);
extern "C" int GpuTransConv2D(int, int, __TYPE__*, __TYPE__**);
extern "C" int GpuTransConv2D_FromDevice(int, int, __TYPE__*, __TYPE__**);
#endif



/*
py::array_t<__TYPE__>  generic_red(int nx, int ny, 
        py::array_t<__TYPE__> s, 
        py::array_t<__TYPE__> x, 
        py::array_t<__TYPE__> y, 
        py::array_t<__TYPE__> b) {

    using VARSI = typename F::template VARS<0>;	// list variables of type I used in formula F
    using VARSJ = typename F::template VARS<1>; // list variables of type J used in formula F
    using VARSP = typename F::template VARS<2>; // list variables of type parameter used in formula F

    const int NARGSI = VARSI::SIZE; // number of I variables used in formula F
    const int NARGSJ = VARSJ::SIZE; // number of J variables used in formula F
    const int NARGSP = VARSP::SIZE; // number of parameters variables used in formula F
    
    const int NARGS = NARGSI + NARGSJ + NARGSP;


    __TYPE__ **hhh = new __TYPE__* [NARGS];

    hhh[0] = (__TYPE__ *)s.data() ;
    hhh[1] = (__TYPE__ *)x.data() ;
    hhh[2] = (__TYPE__ *)y.data() ;
    hhh[3] = (__TYPE__ *)b.data() ;


    std::cout << endl;
    for (int i = 0; i < 3; i++){
        auto temps = hhh[1] ;
        std::cout << temps[i] << endl;
    }
    std::cout << endl;
    for (int i = 0; i < 3; i++){
        auto temps = hhh[2] ;
        std::cout << temps[i] << endl;
    }
    std::cout << endl;
    for (int i = 0; i < 3; i++){
        auto temps = hhh[3] ;
        std::cout << temps[i] << endl;
    }


    // dimension Output
    int dimout = F::DIM;
    int nout = nx;
    auto result = py::array_t<__TYPE__>(nx*dimout);
    
    CpuConv(nx, ny, (__TYPE__*)result.data(), hhh);
    
    return result;
}
*/


//template <typename __TYPE__> void save_array_from_data(int size, _pyObject * s,  __TYPE__ * cpp_ptr){

    //cpp_ptr = new __TYPE__[size];
    //__TYPE__ *temp_ptr = s->data();

    //auto s = py::cast<py::array_t<__TYPE__>  >(py_args);
    //for (auto j=0; j< size ; j++) {
        //cpp_ptr[j] = temp_ptr[j];
    //}
//}

py::array_t<__TYPE__>  generic_red(int nx, int ny, 
       int tagIJ,
       int tagCpuGpu,
       int tag1D2D,
       py::args py_args) {

    // ------ get the dimension from formula -----//
    using VARSI = typename F::template VARS<0>;	// list variables of type I 
    using VARSJ = typename F::template VARS<1>; // list variables of type J 
    using VARSP = typename F::template VARS<2>; // list variables of type parameter 

    using DIMSX = GetDims<VARSI>;
    using DIMSY = GetDims<VARSJ>;
    using DIMSP = GetDims<VARSP>;

    using INDSI = GetInds<VARSI>;
    using INDSJ = GetInds<VARSJ>;
    using INDSP = GetInds<VARSP>;

    using INDS = ConcatPacks<ConcatPacks<INDSI,INDSJ>,INDSP>;

    const int NARGSI = VARSI::SIZE; // number of I variables used in formula F
    const int NARGSJ = VARSJ::SIZE; // number of J variables used in formula F
    const int NARGSP = VARSP::SIZE; // number of parameters variables used in formula F
    
    const int NARGS = NARGSI + NARGSJ + NARGSP;
    
    // ------ check the dimensions ------------//
    int *typeargs = new int[NARGS];
    int *dimargs = new int[NARGS];

    for(int k=0; k<NARGS; k++) {
        typeargs[k] = -1;
        dimargs[k] = -1;
    }
    for(int k=0; k<NARGSI; k++) {
        //typeargs[INDSI::VAL(k)] = 0;
        typeargs[INDSI::VAL(k)] = nx;
        dimargs[INDSI::VAL(k)] = DIMSX::VAL(k);
    }
    for(int k=0; k<NARGSJ; k++) {
        //typeargs[INDSJ::VAL(k)] = 1;
        typeargs[INDSJ::VAL(k)] = ny;
        dimargs[INDSJ::VAL(k)] = DIMSY::VAL(k);
    }
    for(int k=0; k<NARGSP; k++) {
        //typeargs[INDSP::VAL(k)] = 2;
        typeargs[INDSP::VAL(k)] = 1;
        dimargs[INDSP::VAL(k)] = DIMSP::VAL(k);
    }

    assert(py_args.size() == NARGS);

    __TYPE__ **castedargs = new __TYPE__* [NARGS];
    // Seems to be necessary to store every pointer in an array to keep in memory
    // the right address 
    py::array_t<__TYPE__> *obj_ptr = new py::array_t<__TYPE__> [NARGS];
    for (size_t i = 0; i < NARGS; i++){
        obj_ptr[i] = py::cast<py::array_t<__TYPE__>> (py_args[i]);

        // check the dimension :
        assert(obj_ptr[i].shape(0) ==  typeargs[i]);
        assert(obj_ptr[i].shape(1) ==  dimargs[i]);


        castedargs[i] = (__TYPE__ *)obj_ptr[i].data() ;
    }

    // dimension Output
    int dimout = F::DIM;
    int nout = nx;
    auto result_array = py::array_t<__TYPE__>(nx*dimout);
    __TYPE__ * result = (__TYPE__ *) result_array.data();
    
    //////////////////////////////////////////////////////////////
    // Call Cuda codes
    //////////////////////////////////////////////////////////////

    // tagCpuGpu=0 means convolution on Cpu, tagCpuGpu=1 means convolution on Gpu, tagCpuGpu=2 means convolution on Gpu from device data
    // tagIJ=0 means sum over j, tagIJ=1 means sum over j
    // tag1D2D=0 means 1D Gpu scheme, tag1D2D=1 means 2D Gpu scheme

    if(tagCpuGpu==0) {
        if(tagIJ==0){
            CpuConv( nx, ny, result, castedargs);
        } else{
            CpuTransConv( nx, ny, result, castedargs);
        } 
    }
#ifdef USE_CUDA
    else if(tagCpuGpu==1) {
        if(tagIJ==0) {
            if(tag1D2D==0) {
                GpuConv1D( nx, ny, result, castedargs);
            } else {
                GpuConv2D( nx, ny, result, castedargs);
            }
        } else {
            if(tag1D2D==0){
                GpuTransConv1D( nx, ny, result, castedargs);
            } else {
                GpuTransConv2D( nx, ny, result, castedargs);
            }
        }
    } 
    else {
        if(tagIJ==0) {
            if(tag1D2D==0){
                GpuConv1D_FromDevice( nx, ny, result, castedargs);
            } else {
                GpuConv2D_FromDevice( nx, ny, result, castedargs);
            }
        } else {
            if(tag1D2D==0){
                GpuTransConv1D_FromDevice( nx, ny, result, castedargs);
            } else{
                GpuTransConv2D_FromDevice( nx, ny, result, castedargs);
            }
        }
    }
#endif
    
    delete[] castedargs;
    delete[] obj_ptr;

    return result_array;
}


PYBIND11_MODULE(pykeops_module, m) {
    m.doc() = "keops io through pybind11"; // optional module docstring

    m.def("gen_red", &generic_red, "A function...");
}





/*
//py::array_t<__TYPE__>  generic_red(int nx, int ny, py::args py_args) {
py::array_t<__TYPE__>  generic_red(int nx, int ny, py::array_t<__TYPE__> s, py::array_t<__TYPE__> x, py::array_t<__TYPE__>y, py::array_t<__TYPE__>b) {
//py::array_t<__TYPE__>  generic_red(int nx, int ny,__TYPE__ s,  __TYPE__* x, __TYPE__* y, __TYPE__* b) {
//py::array_t<__TYPE__>  generic_red(int nx, int ny) {

    using VARSI = typename F::template VARS<0>;	// list variables of type I used in formula F
    using VARSJ = typename F::template VARS<1>; // list variables of type J used in formula F
    using VARSP = typename F::template VARS<2>; // list variables of type parameter used in formula F

    using DIMSX = GetDims<VARSI>;
    using DIMSY = GetDims<VARSJ>;
    using DIMSP = GetDims<VARSP>;

    using INDSI = GetInds<VARSI>;
    using INDSJ = GetInds<VARSJ>;
    using INDSP = GetInds<VARSP>;

    using INDS = ConcatPacks<ConcatPacks<INDSI,INDSJ>,INDSP>;

    const int NARGSI = VARSI::SIZE; // number of I variables used in formula F
    const int NARGSJ = VARSJ::SIZE; // number of J variables used in formula F
    const int NARGSP = VARSP::SIZE; // number of parameters variables used in formula F
    
    //const int NARGS = NARGSI + NARGSJ + NARGSP;
    const int NARGS = INDS::SIZE;


    //py::tuple t(py_args.size());

    //__TYPE__ *hhh[py_args.size()]; 
    __TYPE__ **hhh = new __TYPE__* [NARGS];

    hhh[0] = (__TYPE__ *)s.data() ;
    hhh[1] = (__TYPE__ *)x.data() ;
    hhh[2] = (__TYPE__ *)y.data() ;
    hhh[3] = (__TYPE__ *)b.data() ;


    //for (size_t i = 0; i < py_args.size(); i++){
        ////t[i] = (PyTuple_GET_ITEM(py_args.ptr(), static_cast<ssize_t>(i)));
        //hhh[i] = (__TYPE__*) (PyTuple_GET_ITEM(py_args.ptr(), static_cast<ssize_t>(i)));
        ////t[i] = (int) Py_REFCNT(PyTuple_GET_ITEM(py_args.ptr(), static_cast<ssize_t>(i)));
        ////hhh[i] = (__TYPE__*) static_cast<py::array_t<__TYPE__>>(t[i]).data();

        ////t[i] = (PyTuple_GET_ITEM(py_args.ptr(), static_cast<ssize_t>(i)));
    //}

    //py::print(static_cast<py::array_t<__TYPE__>>(t[0]).data());
    //py::print(py_args[0]);
    //py::print(py_args[1]);
    //py::print(py_args[2]);
    //py::print(py_args[3]);
    //py::print(t[0]);
    //py::print(t[1]);
    //py::print(t[2]);
    //py::print(t[3]);

    //std::cout << endl;
    //py::print(hhh[0]);
    //py::print(hhh[1]);
    //py::print(hhh[2]);
    //py::print(hhh[3]);
    
    std::cout << endl;
    //for (int i = 0; i < 1; i++){
        //auto temps = s ;
        ////auto temps = hhh[0] ;
        //std::cout << temps << endl;
    //}
    std::cout << endl;
    for (int i = 0; i < 3; i++){
        auto temps = hhh[1] ;
        std::cout << temps[i] << endl;
    }
    std::cout << endl;
    for (int i = 0; i < 3; i++){
        auto temps = hhh[2] ;
        std::cout << temps[i] << endl;
    }
    std::cout << endl;
    for (int i = 0; i < 3; i++){
        auto temps = hhh[3] ;
        std::cout << temps[i] << endl;
    }
     //__TYPE__ mo = *(hhh[2]);
     //py::print(mo);


    // dimension Output
    int dimout = F::DIM;
    int nout = nx;

    //__TYPE__ **args = new __TYPE__*[NARGS];
    //__TYPE__ **castedargs = new __TYPE__*[NARGS];
    //for(int k=0; k<py_args.size(); k++) {
        //auto a = py_args;

        //args[k] = p.data();
        //castedargs[k] = castedFun(args[k],prhs[argu+k]);
    //}

    auto result = py::array_t<__TYPE__>(nx*dimout);
    //__TYPE__ ** hh;
    
    CpuConv(nx, ny, (__TYPE__*)result.data(), hhh);
    
    
    return result;
}
*/






















//py::array_t<double> add_arrays(py::array_t<double> input1, py::array_t<double> input2) {
    //auto buf1 = input1.request(), buf2 = input2.request();

    //if (buf1.ndim != 1 || buf2.ndim != 1)
        //throw std::runtime_error("Number of dimensions must be one");

    //if (buf1.size != buf2.size)
        //throw std::runtime_error("Input shapes must match");

    //auto result = py::array_t<double>(buf1.size);

    //auto buf3 = result.request();

    //double *ptr1 = (double *) buf1.ptr,
           //*ptr2 = (double *) buf2.ptr,
           //*ptr3 = (double *) buf3.ptr;

    //for (size_t idx = 0; idx < buf1.shape[0]; idx++)
        //ptr3[idx] = ptr1[idx] + ptr2[idx];

    //return result;
//}

//PYBIND11_MODULE(pykeops_module, m) {
    //m.def("add_arrays", &add_arrays, "Add two NumPy arrays");
//}
