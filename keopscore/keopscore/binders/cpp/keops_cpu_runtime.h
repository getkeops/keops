#pragma once

#if defined(_WIN32)
#define KEOPS_CPU_EXPORT __declspec(dllexport)
#define KEOPS_CPU_INTERNAL
#else
#define KEOPS_CPU_EXPORT __attribute__((visibility("default")))
#define KEOPS_CPU_INTERNAL __attribute__((visibility("hidden")))
#endif

#include <vector>

struct KeOpsCpuArgs {
  signed long int dimY;
  signed long int nx;
  signed long int ny;
  int tagI;
  int tagZero;
  int use_half;
  signed long int dimred;
  int use_chunk_mode;

  int n_indsi;
  const int *indsi;
  int n_indsj;
  const int *indsj;
  int n_indsp;
  const int *indsp;

  signed long int dimout;
  int n_dimsx;
  const signed long int *dimsx;
  int n_dimsy;
  const signed long int *dimsy;
  int n_dimsp;
  const signed long int *dimsp;

  signed long int **ranges;

  int n_shapeout;
  const signed long int *shapeout;
  void *out;

  int nargs;
  void **arg;
  const signed long int **argshape;
  const int *argshape_sizes;
};

typedef int (*KeOpsCpuLaunchFn)(const KeOpsCpuArgs *);

template <typename TYPE>
using KeOpsCpuFormulaLaunchFn = int (*)(
    signed long int dimY, signed long int nx, signed long int ny, int tagI,
    int tagZero, int use_half, signed long int dimred, int use_chunk_mode,
    std::vector<int> indsi, std::vector<int> indsj, std::vector<int> indsp,
    signed long int dimout, std::vector<signed long int> dimsx,
    std::vector<signed long int> dimsy, std::vector<signed long int> dimsp,
    signed long int **ranges, std::vector<signed long int> shapeout, TYPE *out,
    TYPE **arg, std::vector<std::vector<signed long int>> argshape);

template <typename TYPE>
KEOPS_CPU_INTERNAL int
keops_cpu_launch_impl(const KeOpsCpuArgs *runtime_args,
                      KeOpsCpuFormulaLaunchFn<TYPE> formula_launcher);
