#include <Rcpp.h>
#include <RcppEigen.h>
#include <vector>

#include "rkeops_data_type.h"
#include "rkeops_matrix.h"

#include "generic_red.h"

// Eigen matrix type (row or col major, Keops side)
#if C_CONTIGUOUS
using eigen_matrix_t = Eigen::Matrix<rkeops::type_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
#else
using eigen_matrix_t = Eigen::Matrix<rkeops::type_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;
#endif

// Eigen R-style matrix (double and col-major)
using eigen_r_matrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;

// Eigen matrix for io (copy or Map between R-style matrix and Keops side)
#if !USE_DOUBLE || C_CONTIGUOUS
// if float or row-major: cast with copy
using io_matrix = eigen_r_matrix;
using input_matrix = eigen_matrix_t;
#else
// if double and col-major: no copy
using io_matrix = Eigen::Map<eigen_r_matrix>;
using input_matrix = Eigen::Map<eigen_matrix_t>;
#endif

// list of input data
using input_list_t = std::vector<input_matrix>;

// rkeops matrix type (proxy type)
using rkeops_base_matrix_t = rkeops::base_matrix<rkeops::type_t>;
using rkeops_matrix_t = rkeops::matrix<rkeops::type_t>;

// list of raw input data (list of proxy matrix)
using rkeops_list_t = std::vector<rkeops_base_matrix_t>;

// Interface
// [[Rcpp::depends(RcppEigen)]]
// [[Rcpp::export]]
SEXP r_genred(
        const Rcpp::List & input,
        const Rcpp::List & param) {
    
    // tagCpuGpu=0 means Reduction on Cpu,
    // tagCpuGpu=1 means Reduction on Gpu,
    // tagCpuGpu=2 means Reduction on Gpu from device data
    int tagCpuGpu = Rcpp::as<int>(param["tagCpuGpu"]);
    // tag1D2D=0 means 1D Gpu scheme,
    // tag1D2D=1 means 2D Gpu scheme
    int tag1D2D = Rcpp::as<int>(param["tag1D2D"]);
    // tagHostDevice=1 means _fromDevice suffix. tagHostDevice=0 means _fromHost suffix
    int tagHostDevice = Rcpp::as<int>(param["tagHostDevice"]);
    // id of GPU device
    int Device_Id = Rcpp::as<int>(param["device_id"]);
    // nx, ny
    int nx = Rcpp::as<int>(param["nx"]);
    int ny = Rcpp::as<int>(param["ny"]);
    
    // ---------------------------------------------------------------------- //
    // Input Data
    // matrix in R = double and col-major
    
    // Prepare input data
    input_list_t input_list;
    for(int i=0; i < input.length(); i++) {
        // Rcpp input matrix to Eigen::Map (double, col-major): no copy
        io_matrix tmp(Rcpp::as<io_matrix>(input[i]));
        // if float or row-major: copy (cast to float and/or row-major)
        // if double and col-major: no copy
        input_matrix casted_tmp(tmp.cast<rkeops::type_t>());
        input_list.push_back(casted_tmp);
    }
    
    // Data flatten (no copy)
    rkeops_list_t raw_input_list;
    for(int i=0; i < input.length(); i++) {
        rkeops_base_matrix_t raw_data(
            (rkeops::type_t*) input_list[i].data(), 
            input_list[i].rows(), input_list[i].cols());
        raw_input_list.push_back(raw_data);
    }
    
    // ---------------------------------------------------------------------- //
    // Computation
    rkeops_matrix_t raw_output = genred(
            tagCpuGpu, tag1D2D, tagHostDevice, 
            Device_Id, nx, ny,
            raw_input_list);
    
    // ---------------------------------------------------------------------- //
    // Result
    // back to Eigen matrix (no copy)
    Eigen::Map<eigen_matrix_t> tmp_output(
        raw_output.get_data(), raw_output.get_nrow(), raw_output.get_ncol());
    // copy only if necessary (from float or row-major to double col-major)
    io_matrix output(tmp_output.cast<double>());
    
    return(Rcpp::NumericMatrix(Rcpp::wrap(output)));
}
