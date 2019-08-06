#include <Rcpp.h>
#include <RcppEigen.h>
#include "rkeops.h"

#ifdef _USE_RCPP
bool use_rcpp = true;

//' Dummy function to check if `rkeops` is installed on the system
//' @keywords internal
//' @description
//' This function is used in automatic tests and check fucntions where the 
//' expected behavior is different depending if `rkeops` is installed or not 
//' (e.g. in calls to `devtools::check()` vs `devtools::test()` respectively).
//' @details
//' When running automatic tests without building and installing the package, 
//' i.e. with `devtools::test()`, the package file structure is different 
//' from the structure of a built and installed package (obtained with 
//' `devtools::check()`). For instance, the directory `inst/include` is 
//' replaced by `include`, or `src` by `lib`. Check functions inside `rkeops` 
//' have different behavior between these two cases.
//' 
//' The function `is_compiled` is used, if available, by the function 
//' [rkeops::is_installed()] to check the availability of `rkeops` package 
//' installation files. If not, i.e. if `rkeops` is not installed, the 
//' function [rkeops::is_installed()] can detect it.
//' @author Ghislain Durif
//' @seealso [rkeops::is_installed()]
//' @export
// [[Rcpp::export]]
int is_compiled() {
    return(1);
}
#endif