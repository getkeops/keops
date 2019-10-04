#include <Rcpp.h>
#include <RcppEigen.h>

// [[Rcpp::export]]
int test_binder() {
    return(1);
}

// // [[Rcpp::depends(RcppEigen)]]
// // [[Rcpp::export]]
// float test_double(Eigen::MatrixXf test) {
//     return(test.sum());
// }
// 
// // [[Rcpp::depends(RcppEigen)]]
// // [[Rcpp::export]]
// Eigen::MatrixXf test_list(Rcpp::List input) {
//     return(Rcpp::as<Eigen::MatrixXf>(input[0]));
// }