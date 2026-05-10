#include "binders/cpp/keops_cpu_runtime.h"

#include <pybind11/pybind11.h>

#include <stdexcept>
#include <string>
#include <vector>

#if defined(_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace py = pybind11;

namespace pykeops_cpp_backend {

class DynamicLibrary {
public:
  explicit DynamicLibrary(const char *path) : path_(path), handle_(nullptr) {
#if defined(_WIN32)
    handle_ = LoadLibraryA(path_.c_str());
    if (handle_ == nullptr) {
      throw std::runtime_error("[pyKeOps] Error: could not load CPU formula library " +
                               path_ + ": " + last_error());
    }
#else
    dlerror();
    handle_ = dlopen(path_.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (handle_ == nullptr) {
      const char *error = dlerror();
      throw std::runtime_error("[pyKeOps] Error: could not load CPU formula library " +
                               path_ + ": " + (error ? error : "unknown error"));
    }
#endif
  }

  ~DynamicLibrary() {
    if (handle_ != nullptr) {
#if defined(_WIN32)
      FreeLibrary(static_cast<HMODULE>(handle_));
#else
      dlclose(handle_);
#endif
    }
  }

  DynamicLibrary(const DynamicLibrary &) = delete;
  DynamicLibrary &operator=(const DynamicLibrary &) = delete;

  void *symbol(const char *name) {
#if defined(_WIN32)
    void *sym = reinterpret_cast<void *>(
        GetProcAddress(static_cast<HMODULE>(handle_), name));
    if (sym == nullptr) {
      throw std::runtime_error("[pyKeOps] Error: could not find symbol " +
                               std::string(name) + " in " + path_ + ": " +
                               last_error());
    }
    return sym;
#else
    dlerror();
    void *sym = dlsym(handle_, name);
    const char *error = dlerror();
    if (error != nullptr) {
      throw std::runtime_error("[pyKeOps] Error: could not find symbol " +
                               std::string(name) + " in " + path_ + ": " +
                               error);
    }
    return sym;
#endif
  }

private:
#if defined(_WIN32)
  static std::string last_error() {
    DWORD error = GetLastError();
    if (error == 0) {
      return "unknown error";
    }

    LPSTR buffer = nullptr;
    size_t size = FormatMessageA(
        FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM |
            FORMAT_MESSAGE_IGNORE_INSERTS,
        nullptr, error, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
        reinterpret_cast<LPSTR>(&buffer), 0, nullptr);

    std::string message(buffer, size);
    LocalFree(buffer);
    return message;
  }
#endif

  std::string path_;
  void *handle_;
};

template <typename TYPE> class KeOps_module_python {
public:
  explicit KeOps_module_python(const char *formula_library_name)
      : library_(formula_library_name), launch_(nullptr) {
    launch_ = reinterpret_cast<KeOpsCpuLaunchFn>(
        library_.symbol("keops_cpu_launch"));
  }

  int operator()(signed long int dimY, signed long int nx, signed long int ny,
                 int tagI, int tagZero, int use_half,
                 signed long int dimred, int use_chunk_mode,
                 py::tuple py_indsi, py::tuple py_indsj, py::tuple py_indsp,
                 signed long int dimout, py::tuple py_dimsx,
                 py::tuple py_dimsy, py::tuple py_dimsp, py::tuple py_ranges,
                 py::tuple py_shapeout, long out_void, py::tuple py_arg,
                 py::tuple py_argshape) {

    std::vector<int> indsi_v(py_indsi.size());
    for (py::ssize_t i = 0; i < py_indsi.size(); i++)
      indsi_v[i] = py::cast<int>(py_indsi[i]);

    std::vector<int> indsj_v(py_indsj.size());
    for (py::ssize_t i = 0; i < py_indsj.size(); i++)
      indsj_v[i] = py::cast<int>(py_indsj[i]);

    std::vector<int> indsp_v(py_indsp.size());
    for (py::ssize_t i = 0; i < py_indsp.size(); i++)
      indsp_v[i] = py::cast<int>(py_indsp[i]);

    std::vector<signed long int> dimsx_v(py_dimsx.size());
    for (py::ssize_t i = 0; i < py_dimsx.size(); i++)
      dimsx_v[i] = py::cast<signed long int>(py_dimsx[i]);

    std::vector<signed long int> dimsy_v(py_dimsy.size());
    for (py::ssize_t i = 0; i < py_dimsy.size(); i++)
      dimsy_v[i] = py::cast<signed long int>(py_dimsy[i]);

    std::vector<signed long int> dimsp_v(py_dimsp.size());
    for (py::ssize_t i = 0; i < py_dimsp.size(); i++)
      dimsp_v[i] = py::cast<signed long int>(py_dimsp[i]);

    std::vector<signed long int *> ranges_v(py_ranges.size());
    for (py::ssize_t i = 0; i < py_ranges.size(); i++)
      ranges_v[i] =
          reinterpret_cast<signed long int *>(py::cast<signed long int>(py_ranges[i]));

    std::vector<signed long int> shapeout_v(py_shapeout.size());
    for (py::ssize_t i = 0; i < py_shapeout.size(); i++)
      shapeout_v[i] = py::cast<signed long int>(py_shapeout[i]);

    TYPE *out = reinterpret_cast<TYPE *>(out_void);

    std::vector<TYPE *> typed_arg_v(py_arg.size());
    std::vector<void *> arg_v(py_arg.size());
    for (py::ssize_t i = 0; i < py_arg.size(); i++) {
      typed_arg_v[i] = reinterpret_cast<TYPE *>(py::cast<long>(py_arg[i]));
      arg_v[i] = typed_arg_v[i];
    }

    std::vector<std::vector<signed long int>> argshape_v(py_argshape.size());
    for (py::ssize_t i = 0; i < py_argshape.size(); i++) {
      py::tuple tmp = py_argshape[i];
      std::vector<signed long int> tmp_v(tmp.size());
      for (py::ssize_t j = 0; j < tmp.size(); j++)
        tmp_v[j] = py::cast<signed long int>(tmp[j]);
      argshape_v[i] = tmp_v;
    }

    for (int k = 0; k < (int)indsi_v.size(); k++) {
      int idx = indsi_v[k];
      const auto &shape = argshape_v[idx];
      if ((int)shape.size() < 2)
        throw std::invalid_argument(
            "[pyKeOps] Error: Vi argument #" + std::to_string(idx) +
            " requires at least 2 dimensions, got " +
            std::to_string(shape.size()) + ".");
      if (shape.back() != dimsx_v[k])
        throw std::invalid_argument(
            "[pyKeOps] Error: Vi argument #" + std::to_string(idx) +
            " has trailing dim " + std::to_string(shape.back()) +
            ", expected " + std::to_string(dimsx_v[k]) + ".");
    }
    for (int k = 0; k < (int)indsj_v.size(); k++) {
      int idx = indsj_v[k];
      const auto &shape = argshape_v[idx];
      if ((int)shape.size() < 2)
        throw std::invalid_argument(
            "[pyKeOps] Error: Vj argument #" + std::to_string(idx) +
            " requires at least 2 dimensions, got " +
            std::to_string(shape.size()) + ".");
      if (shape.back() != dimsy_v[k])
        throw std::invalid_argument(
            "[pyKeOps] Error: Vj argument #" + std::to_string(idx) +
            " has trailing dim " + std::to_string(shape.back()) +
            ", expected " + std::to_string(dimsy_v[k]) + ".");
    }
    for (int k = 0; k < (int)indsp_v.size(); k++) {
      int idx = indsp_v[k];
      const auto &shape = argshape_v[idx];
      if ((int)shape.size() < 1)
        throw std::invalid_argument(
            "[pyKeOps] Error: Pm argument #" + std::to_string(idx) +
            " requires at least 1 dimension (got a scalar).");
      if (shape.back() != dimsp_v[k])
        throw std::invalid_argument(
            "[pyKeOps] Error: Pm argument #" + std::to_string(idx) +
            " has trailing dim " + std::to_string(shape.back()) +
            ", expected " + std::to_string(dimsp_v[k]) + ".");
    }

    std::vector<const signed long int *> argshape_ptr_v(argshape_v.size());
    std::vector<int> argshape_size_v(argshape_v.size());
    for (std::size_t i = 0; i < argshape_v.size(); ++i) {
      argshape_ptr_v[i] = argshape_v[i].data();
      argshape_size_v[i] = static_cast<int>(argshape_v[i].size());
    }

    KeOpsCpuArgs runtime_args;
    runtime_args.dimY = dimY;
    runtime_args.nx = nx;
    runtime_args.ny = ny;
    runtime_args.tagI = tagI;
    runtime_args.tagZero = tagZero;
    runtime_args.use_half = use_half;
    runtime_args.dimred = dimred;
    runtime_args.use_chunk_mode = use_chunk_mode;
    runtime_args.n_indsi = static_cast<int>(indsi_v.size());
    runtime_args.indsi = indsi_v.data();
    runtime_args.n_indsj = static_cast<int>(indsj_v.size());
    runtime_args.indsj = indsj_v.data();
    runtime_args.n_indsp = static_cast<int>(indsp_v.size());
    runtime_args.indsp = indsp_v.data();
    runtime_args.dimout = dimout;
    runtime_args.n_dimsx = static_cast<int>(dimsx_v.size());
    runtime_args.dimsx = dimsx_v.data();
    runtime_args.n_dimsy = static_cast<int>(dimsy_v.size());
    runtime_args.dimsy = dimsy_v.data();
    runtime_args.n_dimsp = static_cast<int>(dimsp_v.size());
    runtime_args.dimsp = dimsp_v.data();
    runtime_args.ranges = ranges_v.data();
    runtime_args.n_shapeout = static_cast<int>(shapeout_v.size());
    runtime_args.shapeout = shapeout_v.data();
    runtime_args.out = out;
    runtime_args.nargs = static_cast<int>(arg_v.size());
    runtime_args.arg = arg_v.data();
    runtime_args.argshape = argshape_ptr_v.data();
    runtime_args.argshape_sizes = argshape_size_v.data();

    py::gil_scoped_release release;
    return launch_(&runtime_args);
  }

private:
  DynamicLibrary library_;
  KeOpsCpuLaunchFn launch_;
};

} // namespace pykeops_cpp_backend

PYBIND11_MODULE(pykeops_cpp, m) {
  m.doc() = "pyKeOps: reusable CPU frontend through pybind11.";

  py::class_<pykeops_cpp_backend::KeOps_module_python<float>>(
      m, "KeOps_module_float", py::module_local())
      .def(py::init<const char *>())
      .def("__call__",
           &pykeops_cpp_backend::KeOps_module_python<float>::operator());

  py::class_<pykeops_cpp_backend::KeOps_module_python<double>>(
      m, "KeOps_module_double", py::module_local())
      .def(py::init<const char *>())
      .def("__call__",
           &pykeops_cpp_backend::KeOps_module_python<double>::operator());
}
