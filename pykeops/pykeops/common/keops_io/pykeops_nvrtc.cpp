// g++ -std=c++11  -shared -fPIC -O3 -fpermissive -lcuda -lnvrtc  -L/usr/lib -L/opt/cuda/lib64 -L/opt/cuda/targets/x86_64-linux/lib/ -I/opt/cuda/targets/x86_64-linux/include/ -I /home/bcharlier/projets/keops/keops/keopscore/ -I/usr/include/python3.10/ -DnvrtcGetTARGET=nvrtcGetCUBIN -DnvrtcGetTARGETSize=nvrtcGetCUBINSize -DARCHTAG=\"sm\"  keops_io.cpp -o keops_io.cpython-310-x86_64-linux-gnu.so


#include <binders/nvrtc/keops_nvrtc.cpp>
#include <pybind11/pybind11.h>

namespace py = pybind11;

template< typename TYPE >
class KeOps_module_python : public KeOps_module< TYPE > {
public:

    using KeOps_module< TYPE >::KeOps_module;


    int operator()(int tagHostDevice, int dimY, int nx, int ny,
                   int tagI, int tagZero, int use_half,
                   int tag1D2D, int dimred,
                   int cuda_block_size, int use_chunk_mode,
                   py::tuple py_indsi, py::tuple py_indsj, py::tuple py_indsp,
                   int dimout,
                   py::tuple py_dimsx, py::tuple py_dimsy, py::tuple py_dimsp,
                   py::tuple py_ranges,
                   py::tuple py_shapeout,
                   long out_void,
                   py::tuple py_arg,
                   py::tuple py_argshape
    ) {

        /*------------------------------------*/
        /*         Cast input args            */
        /*------------------------------------*/

        std::vector< int > indsi_v(py_indsi.size());
        for (auto i = 0; i < py_indsi.size(); i++)
            indsi_v[i] = py::cast< int >(py_indsi[i]);


        std::vector< int > indsj_v(py_indsj.size());
        for (auto i = 0; i < py_indsj.size(); i++)
            indsj_v[i] = py::cast< int >(py_indsj[i]);


        std::vector< int > indsp_v(py_indsp.size());
        for (auto i = 0; i < py_indsp.size(); i++)
            indsp_v[i] = py::cast< int >(py_indsp[i]);


        std::vector< int > dimsx_v(py_dimsx.size());
        for (auto i = 0; i < py_dimsx.size(); i++)
            dimsx_v[i] = py::cast< int >(py_dimsx[i]);


        std::vector< int > dimsy_v(py_dimsy.size());
        for (auto i = 0; i < py_dimsy.size(); i++)
            dimsy_v[i] = py::cast< int >(py_dimsy[i]);


        std::vector< int > dimsp_v(py_dimsp.size());
        for (auto i = 0; i < py_dimsp.size(); i++)
            dimsp_v[i] = py::cast< int >(py_dimsp[i]);


        // Cast the ranges arrays
        std::vector< int * > ranges_v(py_ranges.size());
        for (int i = 0; i < py_ranges.size(); i++)
            ranges_v[i] = (int *) py::cast< long >(py_ranges[i]);
        int **ranges = (int **) ranges_v.data();

        // for (auto i: ranges_v)
        //    std::cout << " " <<  (long) i << " ";
        // std::cout << std::endl;

        //for (auto i=0; i<7; i++)
        //   std::cout << " " <<  (long) ranges[i] << " ";
        //std::cout << std::endl;

        std::vector< int > shapeout_v(py_shapeout.size());
        for (auto i = 0; i < py_shapeout.size(); i++)
            shapeout_v[i] = py::cast< int >(py_shapeout[i]);

        TYPE *out = (TYPE *) out_void;
        // std::cout << "out_ptr : " << (long) out << std::endl;

        std::vector < TYPE * > arg_v(py_arg.size());
        for (int i = 0; i < py_arg.size(); i++)
            arg_v[i] = (TYPE *) py::cast< long >(py_arg[i]);
        TYPE **arg = (TYPE **) arg_v.data();

        std::vector <std::vector< int >> argshape_v(py_argshape.size());
        for (auto i = 0; i < py_argshape.size(); i++) {
            py::tuple tmp = py_argshape[i];
            std::vector< int > tmp_v(tmp.size());
            for (auto j = 0; j < tmp.size(); j++)
                tmp_v[j] = py::cast< int >(tmp[j]);
            argshape_v[i] = tmp_v;
        }

//        for (auto i : argshape_v)
//            for (auto j : i)
//                std::cout << j << " " ;

        return KeOps_module< TYPE >::launch_kernel(tagHostDevice,
                                                   dimY,
                                                   nx,
                                                   ny,
                                                   tagI,
                                                   tagZero,
                                                   use_half,
                                                   tag1D2D,
                                                   dimred,
                                                   cuda_block_size,
                                                   use_chunk_mode,
                                                   indsi_v,
                                                   indsj_v,
                                                   indsp_v,
                                                   dimout,
                                                   dimsx_v,
                                                   dimsy_v,
                                                   dimsp_v,
                                                   ranges,
                                                   shapeout_v,
                                                   out,
                                                   arg,
                                                   argshape_v);
    }

};
/////////////////////////////////////////////////////////////////////////////////
//                    PyBind11 entry point                                     //
/////////////////////////////////////////////////////////////////////////////////


PYBIND11_MODULE(pykeops_nvrtc, m) {
m.doc() = "pyKeOps: KeOps for pytorch through pybind11 (pytorch flavour).";

py::class_< KeOps_module_python< float > >(m, "KeOps_module_float")
.def(py::init<int, int, const char *>())
.def("__call__", &KeOps_module_python< float >::operator());

py::class_< KeOps_module_python< double > >(m, "KeOps_module_double")
.def(py::init<int, int, const char *>())
.def("__call__", &KeOps_module_python< double >::operator());

py::class_< KeOps_module_python< half2 > >(m, "KeOps_module_half2")
.def(py::init<int, int, const char *>())
.def("__call__", &KeOps_module_python< half2 >::operator());
}
