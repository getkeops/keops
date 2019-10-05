#include <Rcpp.h>
#include <RcppEigen.h>
#include <vector>

#include "rkeops_data_type.h"
#include "rkeops_matrix.h"

#include "generic_red.h"

// Eigen matrix type
#if C_CONTIGUOUS
using eigen_matrix_t = Eigen::Matrix<rkeops::type_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
#else
using eigen_matrix_t = Eigen::Matrix<rkeops::type_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;
#endif

// rkeops matrix type
using rkeops_matrix_t = rkeops::matrix<rkeops::type_t>;
// list of input data
using rkeops_list_t = std::vector<rkeops_matrix_t>;

// raw data type
using rkeops_array_t = std::vector<rkeops::type_t>;

// Interface
// [[Rcpp::depends(RcppEigen)]]
// [[Rcpp::export]]
SEXP r_genred(
        Rcpp::List & input,
        Rcpp::List & param) {
    
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
    
    // Data
    rkeops_list_t raw_input;
    for(int i=0; i < input.length(); i++) {
        eigen_matrix_t tmp = Rcpp::as<eigen_matrix_t>(input[i]);
        rkeops_matrix_t raw_data = rkeops_matrix_t(tmp.data(), tmp.rows(), tmp.cols());
        raw_input.push_back(raw_data);
    }
    
    // Computation
    rkeops_matrix_t raw_output = genred(
            tagCpuGpu, tag1D2D, tagHostDevice, 
            Device_Id, nx, ny,
            raw_input);
    
    // Result
    eigen_matrix_t output = Eigen::Map<eigen_matrix_t>(raw_output.get_data(), 
                                                       raw_output.get_nrow(), 
                                                       raw_output.get_ncol());
    
    return(Rcpp::NumericMatrix(Rcpp::wrap(output)));
}
