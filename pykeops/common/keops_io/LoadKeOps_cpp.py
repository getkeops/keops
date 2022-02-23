import os
import sysconfig

import keops.config.config
from keops.config.config import get_build_folder
import pykeops
from keops.utils.Cache import Cache_partial
from pykeops.common.keops_io.LoadKeOps import LoadKeOps
from pykeops.common.utils import pyKeOps_Message
from pykeops.config import pykeops_cpp_name


def get_pybind11_code(tag, include):
    return f"""
        #include "{include}"

        #include <pybind11/pybind11.h>
        namespace py = pybind11;

        template < typename TYPE >
        int launch_pykeops_{tag}_cpu(int dimY, int nx, int ny,
                 int tagI, int tagZero, int use_half,
                 int dimred,
                 int use_chunk_mode,
                 py::tuple py_indsi, py::tuple py_indsj, py::tuple py_indsp,
                 int dimout,
                 py::tuple py_dimsx, py::tuple py_dimsy, py::tuple py_dimsp,
                 py::tuple py_ranges,
                 py::tuple py_shapeout,
                 long out_void,
                 py::tuple py_arg,
                 py::tuple py_argshape){{

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
          std::vector< int* > ranges_v(py_ranges.size());
          for (int i = 0; i < py_ranges.size(); i++)
            ranges_v[i] = (int*) py::cast< long >(py_ranges[i]);
          int **ranges = (int**) ranges_v.data();

         std::vector< int > shapeout_v(py_shapeout.size());
         for (auto i = 0; i < py_shapeout.size(); i++)
            shapeout_v[i] = py::cast< int >(py_shapeout[i]);

        TYPE *out = (TYPE*) out_void;
        // std::cout << "out_ptr : " << (long) out << std::endl;

        std::vector< TYPE* > arg_v(py_arg.size());
          for (int i = 0; i < py_arg.size(); i++)
            arg_v[i] = (TYPE*) py::cast< long >(py_arg[i]);
        TYPE **arg = (TYPE**) arg_v.data();

        std::vector< std::vector< int > > argshape_v(py_argshape.size());
        for (auto i = 0; i < py_argshape.size(); i++){{
    py::tuple tmp = py_argshape[i];
            std::vector< int > tmp_v(tmp.size());
            for (auto j =0; j < tmp.size(); j++)
                tmp_v[j] = py::cast< int >(tmp[j]);
            argshape_v[i] = tmp_v;
        }}


                return launch_keops_cpu_{tag}< TYPE >(dimY,
                    nx,
                    ny,
                    tagI,
                    tagZero,
                    use_half,
                    dimred,
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

        }}

        PYBIND11_MODULE(pykeops_cpp_{tag}, m) {{
            m.doc() = "pyKeOps: KeOps for pytorch through pybind11 (pytorch flavour).";
            m.def("launch_pykeops_cpu_float", &launch_pykeops_{tag}_cpu < float >, "Entry point to keops - float .");
            m.def("launch_pykeops_cpu_double", &launch_pykeops_{tag}_cpu < double >, "Entry point to keops - float .");
        }}                     
            """


class LoadKeOps_cpp_class(LoadKeOps):
    def __init__(self, *args, fast_init=False):
        super().__init__(*args, fast_init=fast_init)

    def init_phase1(self):
        srcname = pykeops_cpp_name(tag=self.params.tag, extension=".cpp")

        dllname = pykeops_cpp_name(
            tag=self.params.tag, extension=sysconfig.get_config_var("EXT_SUFFIX")
        )

        if not os.path.exists(dllname):
            f = open(srcname, "w")
            f.write(get_pybind11_code(self.params.tag, self.params.source_name))
            f.close()
            compile_command = f"{keops.config.config.cxx_compiler} {keops.config.config.cpp_flags} {pykeops.config.python_includes} {srcname} -o {dllname}"
            pyKeOps_Message(
                "Compiling pykeops cpp " + self.params.tag + " module ... ",
                flush=True,
                end="",
            )
            os.system(compile_command)
            pyKeOps_Message("OK", use_tag=False, flush=True)

    def init_phase2(self):
        import importlib

        mylib = importlib.import_module(
            os.path.basename(pykeops_cpp_name(tag=self.params.tag))
        )
        if self.params.c_dtype == "float":
            self.launch_keops_cpu = mylib.launch_pykeops_cpu_float
        if self.params.c_dtype == "double":
            self.launch_keops_cpu = mylib.launch_pykeops_cpu_double

    def call_keops(self, nx, ny):
        self.launch_keops_cpu(
            self.params.dimy,
            nx,
            ny,
            self.params.tagI,
            self.params.tagZero,
            self.params.use_half,
            self.params.dimred,
            self.params.use_chunk_mode,
            self.params.indsi,
            self.params.indsj,
            self.params.indsp,
            self.params.dim,
            self.params.dimsx,
            self.params.dimsy,
            self.params.dimsp,
            self.ranges_ptr_new,
            self.outshape,
            self.out_ptr,
            self.args_ptr_new,
            self.argshapes_new,
        )


LoadKeOps_cpp = Cache_partial(
    LoadKeOps_cpp_class, use_cache_file=True, save_folder=get_build_folder()
)
