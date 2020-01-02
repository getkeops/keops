#ifndef RKEOPS_MATRIX_H
#define RKEOPS_MATRIX_H

#include <iostream>
#include <iterator>
#include <vector>

namespace rkeops {

// rkeops base matrix type (storing a pointer and dimension)
template <typename T>
class base_matrix {
protected:
    T* _data;     // pointer to data
    size_t _nrow; // number of row
    size_t _ncol; // number of column
    bool _is_contiguous; // true (handled in Rcpp entry point)
    
public:
    base_matrix() {
        _data = nullptr;
        _nrow = 0;
        _ncol = 0;
        _is_contiguous = true;
    };
    
    base_matrix(size_t nrow, size_t ncol) {
        _data = nullptr;
        _nrow = nrow;
        _ncol = ncol;
        _is_contiguous = true;
    };
    
    base_matrix(T* data, size_t nrow, size_t ncol) : base_matrix(nrow, ncol) {
        _data = data;
    };
    
    base_matrix(std::vector<T> & raw_data, size_t nrow, size_t ncol) : 
        base_matrix(raw_data.data(), nrow, ncol) {};
    
    ~base_matrix() {};
    
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
        return(_data);
    };
    
    std::vector<T> get_raw_data() {
        return(std::vector<T>(_data, _data + (this->get_length())));
    };
    
    bool is_contiguous() {
        return(_is_contiguous);
    };
    
    void print() {
        for (auto j = 0; j < this->get_length(); ++j) {
            std::cout << _data[j] << ' ';
            if( (j + 1) % _ncol == 0)
                std::cout << std::endl;
        }
        std::cout << std::endl;
    };
    
};

// rkeops matrix type based on std::vector
template <typename T>
class matrix : public base_matrix<T> {
protected:
    // flat vector of matrix data (replace _data)
    std::vector<T> _raw_data;
    
public:
    matrix() : base_matrix<T>(), _raw_data() {
    };
    
    matrix(size_t nrow, size_t ncol) : 
        base_matrix<T>(nrow, ncol), _raw_data(nrow * ncol) {};
    
    matrix(T* data, size_t nrow, size_t ncol) : 
        base_matrix<T>(data, nrow, ncol) {
            _raw_data = std::vector<T>(data, data + (nrow * ncol));
    };
    
    matrix(std::vector<T> & data, size_t nrow, size_t ncol) : 
        base_matrix<T>(nrow, ncol), _raw_data(data) {};
    
    ~matrix() {};
    
    // !! to be used wisely !!
    void update_data() {
        this->_data = _raw_data.data();
    };
    
    std::vector<T> get_raw_data() {
        return(_raw_data);
    };
    
    void print() {
        for (auto j = _raw_data.begin(); j != _raw_data.end(); ++j) {
            std::cout << *j << ' ';
            if( ((int) std::distance(_raw_data.begin(), j) + 1) % this->_ncol == 0)
                std::cout << std::endl;
        }
        std::cout << std::endl;
    };
};

}

#endif // RKEOPS_MATRIX_H