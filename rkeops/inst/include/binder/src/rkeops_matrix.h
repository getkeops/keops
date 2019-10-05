#ifndef RKEOPS_MATRIX_H
#define RKEOPS_MATRIX_H

#include <vector>
#include <iostream>
#include <iterator>

namespace rkeops {

#if C_CONTIGUOUS
const bool RKEOPS_C_CONTIGUOUS = 1;
#else
const bool RKEOPS_C_CONTIGUOUS = 0;
#endif

// rkeops matrix type based on std::vector
template <typename T>
class matrix {
private:
    std::vector<T> _data; // flat vector of matrix data
    size_t _nrow; // number of row
    size_t _ncol; // number of column
    bool _is_contiguous; // true = row-major or c_contiguous, false = column-major or f_contiguous

public:
    matrix() : _data() {
        _nrow = 0;
        _ncol = 0;
        _is_contiguous = RKEOPS_C_CONTIGUOUS;
    };
    
    matrix(size_t nrow, size_t ncol) {
        _data = std::vector<T>(nrow * ncol);
        _nrow = nrow;
        _ncol = ncol;
        _is_contiguous = RKEOPS_C_CONTIGUOUS;
    };
    
    matrix(std::vector<T> & data, size_t nrow, size_t ncol) {
        _data = std::vector<T>(data.begin(), data.end());
        _nrow = nrow;
        _ncol = ncol;
        _is_contiguous = RKEOPS_C_CONTIGUOUS;
    };
    
    ~matrix() {};
    
    int get_ndim() {
        return(2);
    };
    
    int get_nrow() {
        return(_nrow);
    };
    
    int get_ncol() {
        return(_ncol);
    };
    
    std::vector<size_t> get_shape() {
        std::vector<size_t> out = {_nrow, _ncol};
        return(out);
    };
    
    int get_size(int l) {
        return(this->get_shape()[l]);
    };
    
    int get_length() {
        return(_nrow * _ncol);
    };
    
    T* get_data() {
        return( (T*) _data.data());
    };
    
    std::vector<T> get_raw_data() {
        return(_data);
    };
    
    bool is_contiguous() {
        return(_is_contiguous);
    };
    
    void print() {
        for (auto j = _data.begin(); j != _data.end(); ++j) {
            std::cout << *j << ' ';
            if( ((int) std::distance(_data.begin(), j) + 1) % _ncol == 0)
                std::cout << std::endl;
        }
        std::cout << std::endl;
    };
};

}

#endif // RKEOPS_MATRIX_H