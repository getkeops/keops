#ifndef RKEOPS_UTILS_H
#define RKEOPS_UTILS_H

#include <Rcpp.h>
#include <string>

void rkeops_error(std::basic_string< char > msg) {
    Rcpp::stop(msg);
}

#endif // RKEOPS_UTILS_H