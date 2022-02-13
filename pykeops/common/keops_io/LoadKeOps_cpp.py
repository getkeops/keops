import os

import keops.config.config
import pykeops
from keops.utils.Cache import Cache_partial
from pykeops.common.keops_io.LoadKeOps import LoadKeOps


class LoadKeOps_cpp_class(LoadKeOps):

    def __init__(self, *args, fast_init=False):
        super().__init__(*args, fast_init=fast_init)

    def init_phase2(self):

        if self.params.tagCPUGPU == 0:
            tag = os.path.basename(self.params.dllname).split(".")[0]
            launch_keops_fun_name = ("launch_keops_cpu_" + tag)
            srcname = os.path.join(keops.config.config.build_path, "keops_io_cpp_" + tag + ".cpp")
            import sysconfig
            dllname = os.path.join(keops.config.config.build_path,
                                   "keops_io_cpp_" + tag + sysconfig.get_config_var('EXT_SUFFIX'))

            f = open(srcname, "w")
            f.write(
                f"""
        #include "{tag}.cpp"

        #include <pybind11/pybind11.h>
        namespace py = pybind11;

        template < typename TYPE >
        int launch_keops_{tag}_cpu(int nx, int ny, int tagI, int use_half,
                             py::tuple py_ranges,
                             long py_out, int nargs,
                             py::tuple py_arg,
                             py::tuple py_argshape){{

                  // Cast the ranges arrays
                  std::vector< void* > ranges_v(py_ranges.size());
                  for (int i = 0; i < py_ranges.size(); i++)
                    ranges_v[i] = (void*) py::cast< long >(py_ranges[i]);
                  //int **ranges = (int**) ranges_v.data();

                void *out_void = (void*) py_out;
                // std::cout << "out_ptr : " << (long) out << std::endl;

                std::vector< void* > arg_v(py_arg.size());
                  for (int i = 0; i < py_arg.size(); i++)
                    arg_v[i] = (void*) py::cast< long >(py_arg[i]);
                //TYPE **arg = (TYPE**) arg_v.data();

                std::vector< std::vector< int > > argshape_v(py_argshape.size());
                std::vector< int* > argshape_ptr_v(py_argshape.size());
                for (auto i = 0; i < py_argshape.size(); i++){{
                    py::tuple tmp = py_argshape[i];
                    std::vector< int > tmp_v(tmp.size());
                    for (auto j =0; j < tmp.size(); j++)
                        tmp_v[j] = py::cast< int >(tmp[j]);
                    argshape_v[i] = tmp_v;
                     argshape_ptr_v[i] = argshape_v[i].data();
                }}

                int **argshape = argshape_ptr_v.data();

                return {launch_keops_fun_name}(nx, ny, tagI, use_half,
                             ranges_v,
                             out_void, nargs,
                             arg_v,
                             argshape_ptr_v);


        }}

        PYBIND11_MODULE(keops_io_cpp_{tag}, m) {{
            m.doc() = "pyKeOps: KeOps for pytorch through pybind11 (pytorch flavour).";
            m.def("launch_keops_cpu_float", &launch_keops_{tag}_cpu < float >, "Entry point to keops - float .");
            m.def("launch_keops_cpu_double", &launch_keops_{tag}_cpu < double >, "Entry point to keops - float .");
        }}                     
                    """)
            f.close()

            compile_command = f"{keops.config.config.cxx_compiler} {keops.config.config.cpp_flags} {pykeops.config.python_includes} {srcname} -o {dllname}"
            print("cooooompile")
            os.system(compile_command)

            import importlib
            mylib = importlib.import_module("keops_io_cpp_" + tag)
            if self.params.c_dtype == "float":
                self.launch_keops_cpu = mylib.launch_keops_cpu_float
            if self.params.c_dtype == "double":
                self.launch_keops_cpu = mylib.launch_keops_cpu_double

    def call_keops(self, nx, ny):
        self.launch_keops_cpu(
            nx,
            ny,
            self.params.tagI,
            self.params.use_half,
            self.ranges_ptr_new,
            self.out_ptr,
            self.params.nargs,
            self.args_ptr_new,
            self.argshapes_new,
        )


LoadKeOps_cpp = Cache_partial(
    LoadKeOps_cpp_class, use_cache_file=True, save_folder=keops.config.config.build_path
)
