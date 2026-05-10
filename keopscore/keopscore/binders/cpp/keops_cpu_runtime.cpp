#include "binders/cpp/keops_cpu_runtime.h"

namespace {

template <typename T> std::vector<T> keops_cpu_make_vector(const T *data, int size) {
  if (data == nullptr || size <= 0) {
    return std::vector<T>();
  }
  return std::vector<T>(data, data + size);
}

} // namespace

template <typename TYPE>
int keops_cpu_launch_impl(const KeOpsCpuArgs *runtime_args,
                          KeOpsCpuFormulaLaunchFn<TYPE> formula_launcher) {
  if (runtime_args == nullptr) {
    return -1;
  }

  std::vector<int> indsi_v =
      keops_cpu_make_vector(runtime_args->indsi, runtime_args->n_indsi);
  std::vector<int> indsj_v =
      keops_cpu_make_vector(runtime_args->indsj, runtime_args->n_indsj);
  std::vector<int> indsp_v =
      keops_cpu_make_vector(runtime_args->indsp, runtime_args->n_indsp);

  std::vector<signed long int> dimsx_v =
      keops_cpu_make_vector(runtime_args->dimsx, runtime_args->n_dimsx);
  std::vector<signed long int> dimsy_v =
      keops_cpu_make_vector(runtime_args->dimsy, runtime_args->n_dimsy);
  std::vector<signed long int> dimsp_v =
      keops_cpu_make_vector(runtime_args->dimsp, runtime_args->n_dimsp);
  std::vector<signed long int> shapeout_v =
      keops_cpu_make_vector(runtime_args->shapeout, runtime_args->n_shapeout);

  std::vector<TYPE *> arg_v(runtime_args->nargs);
  for (int i = 0; i < runtime_args->nargs; ++i) {
    arg_v[i] = reinterpret_cast<TYPE *>(runtime_args->arg[i]);
  }

  std::vector<std::vector<signed long int>> argshape_v(runtime_args->nargs);
  for (int i = 0; i < runtime_args->nargs; ++i) {
    const signed long int *shape = runtime_args->argshape[i];
    int shape_size = runtime_args->argshape_sizes[i];
    if (shape != nullptr && shape_size > 0) {
      argshape_v[i].assign(shape, shape + shape_size);
    }
  }

  return formula_launcher(
      runtime_args->dimY, runtime_args->nx, runtime_args->ny, runtime_args->tagI,
      runtime_args->tagZero, runtime_args->use_half, runtime_args->dimred,
      runtime_args->use_chunk_mode, indsi_v, indsj_v, indsp_v,
      runtime_args->dimout, dimsx_v, dimsy_v, dimsp_v, runtime_args->ranges,
      shapeout_v, reinterpret_cast<TYPE *>(runtime_args->out), arg_v.data(),
      argshape_v);
}

template int keops_cpu_launch_impl<float>(
    const KeOpsCpuArgs *runtime_args,
    KeOpsCpuFormulaLaunchFn<float> formula_launcher);

template int keops_cpu_launch_impl<double>(
    const KeOpsCpuArgs *runtime_args,
    KeOpsCpuFormulaLaunchFn<double> formula_launcher);
