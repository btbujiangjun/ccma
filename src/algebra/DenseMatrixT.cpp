/**********************************************
* Author: Jun Jiang - jiangjun4@sina.com
* Created: 2016-11-22 11:30
* Last modified:	2016-12-15 10:54
* Filename:		DenseMatrixT.cpp
* Description: Dense Data Structure of training/predict data
**********************************************/

#include "BaseMatrix.h"
#include <stdio.h>
#include <string.h>

namespace ccma{
namespace algebra{

template<class T>
DenseMatrixT<T>::DenseMatrixT(){
    _data = nullptr;
    _cache_matrix_det = static_cast<T>(INT_MAX);
}


template<class T>
DenseMatrixT<T>::DenseMatrixT(const T* data,
                              const uint rows,
                              const uint cols):BaseMatrixT<T>(rows, cols){
    _data = new T[rows * cols];
    memcpy(_data, data, sizeof(T) * rows * cols);
    this->_rows = rows;
    this->_cols = cols;

    _cache_matrix_det = static_cast<T>(INT_MAX);
}

template<class T>
DenseMatrixT<T>::~DenseMatrixT(){
    if(_data != nullptr){
        delete[] _data;
        _data = nullptr;
    }
}

template<class T>
DenseMatrixT<T>* DenseMatrixT<T>::copy_data(){
    return new DenseMatrixT<T>(_data, this->_rows, this->_cols);
}

template<class T>
T* DenseMatrixT<T>::get_data() const{
    return _data;
}
template<class T>
void DenseMatrixT<T>::set_data(const T* data,
                               const uint rows,
                               const uint cols){
    if(_data != nullptr){
        delete[] _data;
    }
    _data = new T[rows * cols];
    memcpy(_data, data, sizeof(T)* rows * cols);
    this->_rows = rows;
    this->_cols = cols;

    clear_cache();
}

template<class T>
void DenseMatrixT<T>::set_shallow_data(T* data,
                                       const uint rows,
                                       const uint cols){
    if(_data != nullptr){
        delete[] _data;
    }
    _data = data;
    this->_rows = rows;
    this->_cols = cols;
    clear_cache();
}


template<class T>
T DenseMatrixT<T>::get_data(const uint idx) const{
    //todo check_range(idx)
    return _data[idx];
}
template<class T>
bool DenseMatrixT<T>::set_data(const T& value, const uint idx){
    if(check_range(idx)){
        _data[idx] = value;
        clear_cache();
        return true;
    }
    return false;
}

template<class T>
T DenseMatrixT<T>::get_data(const uint row, const uint col) const{
    //todo check_range(row, col)
    return _data[row * this->_cols + col];
}

template<class T>
bool DenseMatrixT<T>::set_data(const T& value,
                               const uint row,
                               const uint col){
    if(check_range(row, col)){
        _data[row * this->_cols + col] = value;
        clear_cache();
        return true;
    }
    return false;
}

template<class T>
DenseMatrixT<T>* DenseMatrixT<T>::get_row_data(const uint row) const{
    // check row range
    DenseMatrixT<T>* dm = new DenseMatrixT<T>();
    T* data = new T[this->_cols];
    memcpy(data, &_data[row * this->_cols], sizeof(T) * this->_cols);
    dm->_data = data;
    dm->_rows = 1;
    dm->_cols = this->_cols;
    return dm;
}
template<class T>
bool DenseMatrixT<T>::set_row_data(BaseMatrixT<T>* mat, const int row){
    if(this->_cols != mat->get_rows()){
        return false;
    }
    int start_row = row;
    if(start_row == -1 || start_row >= this->_rows){
        start_row = this->_rows;
    }

    T* data = new T[(this->_rows + mat->get_rows()) * this->_cols];
    if(start_row > 0){
        memcpy(data, _data, sizeof(T) * this->_cols * start_row);
    }
    memcpy(&data[(start_row) * this->_cols], mat->get_data(), sizeof(T) * mat->get_rows() * mat->get_cols());
    if(start_row < this->_rows){
        memcpy(&data[(start_row + mat->get_rows()) * this->_cols], &_data[this->_cols * start_row], sizeof(T) * (this->_rows - start_row) * this->_cols);
    }

    if(_data != nullptr){
        delete[] _data;
    }
    _data = data;
    this->_rows += mat->get_rows();
    clear_cache();

    return true;
}

template<class T>
bool DenseMatrixT<T>::extend(const BaseMatrixT<T>* mat){
    if(this->_rows != mat->get_rows()){
        return false;
    }

    T* data = new T[this->_rows * (this->_cols + mat->get_cols())];
    for(int i = 0; i < this->_rows; i++){
        memcpy(&data[i * (this->_cols + mat->get_cols())], &_data[i * this->_cols], sizeof(T) * this->_cols);
        memcpy(&data[(i+1) * (this->_cols + mat->get_cols()) - this->_cols], &mat->get_data()[i*mat->get_cols()], sizeof(T) * mat->get_cols());
    }

    set_shallow_data(data, this->_rows, (this->_cols + mat->get_cols()));
}

template<class T>
bool DenseMatrixT<T>::add(const BaseMatrixT<T>* mat){
    if(this->_rows != mat->get_rows() || this->_cols != mat->get_cols()){
        return false;
    }

    for(int i = 0; i < this->_rows * this->_cols; i++){
        _data[i] += mat->get_data(i);
    }
    clear_cache();

    return true;
}

template<class T>
bool DenseMatrixT<T>::add(const BaseMatrixT<T>* mat, BaseMatrixT<T>* result){
    if(this->_rows != mat->get_rows() || this->_cols != mat->get_cols()){
        return false;
    }

    T* data = new T[this->_rows * this->_cols];
    for(int i = 0; i < this->_rows * this->_cols; i++){
        data[i] = _data[i] + mat->get_data(i);
    }

    if(result == nullptr){
        result = new DenseMatrixT<T>();
    }
    result->set_shallow_data(data, this->_rows, this->_cols);

    return true;
}

template<class T>
bool DenseMatrixT<T>::subtract(const BaseMatrixT<T>* mat){
    if( this->_rows != mat->get_rows() || this->_cols != mat->get_cols()){
        return false;
    }

    for(int i = 0; i < this->_rows * this->_cols; i++){
        _data[i] -= mat->get_data(i);
    }
    clear_cache();

    return true;
}


template<class T>
bool DenseMatrixT<T>::subtract(const BaseMatrixT<T>* mat, BaseMatrixT<T>* result){
    if( this->_rows != mat->get_rows() || this->_cols != mat->get_cols()){
        return false;
    }

    T* data = new T[this->_rows * this->_cols];
    for(int i = 0; i < this->_rows * this->_cols; i++){
        data[i] = _data[i] -mat->get_data(i);
    }

    if(result == nullptr){
        result = new DenseMatrixT<T>();
    }
    result->set_shallow_data(data, this->_rows, this->_cols);

    return true;
}

template<class T>
bool DenseMatrixT<T>::product(const T value){
    for(int i = 0; i < this->_rows * this->_cols; i++){
        _data[i] *= value;
    }
    clear_cache();

    return true;
}
template<class T>
bool DenseMatrixT<T>::product(const T value, BaseMatrixT<T>* result){
    T* data = new T[this->_rows * this->_cols];
    for(int i = 0; i < this->_rows * this->_cols; i++){
        data[i] = _data[i] * value;
    }

    if(result == nullptr){
        result = new DenseMatrixT<T>();
    }
    result->set_shallow_data(data, this->_rows, this->_cols);

    return true;
}

/* A(m,p) * B(p,n) = C(m,n)
 *
 *
 */
template<class T>
bool DenseMatrixT<T>::product(const BaseMatrixT<T>* mat, BaseMatrixT<T>* result){
    if(this->_cols != mat->get_rows()){
        return false;
    }

    T* data = new T[this->_rows * mat->get_cols()];
    uint row_cursor = 0;
    for(int i = 0; i < this->_rows; i++){
        for(int j = 0; j < mat->get_cols(); j++){
            T value = static_cast<T>(0);
            for(int k = 0; k < this->_cols; k++){
                value += this->get_data(i, k) * mat->get_data(k, j);
            }
            data[row_cursor++] = value;
        }
    }

    if(result == nullptr){
        result = new DenseMatrixT<T>();
    }
    result->set_shallow_data(data, this->_rows, mat->get_cols());

    return true;
}
template<class T>
bool DenseMatrixT<T>::product(const BaseMatrixT<int>* mat, BaseMatrixT<real>* result){
    if(this->_cols != mat->get_rows()){
        return false;
    }

    real* data = new real[this->_rows * mat->get_cols()];
    uint row_cursor = 0;
    for(int i = 0; i < this->_rows; i++){
        for(int j = 0; j < mat->get_cols(); j++){
            real value = static_cast<real>(0);
            for(int k = 0; k < this->_cols; k++){
                value += (real)this->get_data(i, k) * (real)mat->get_data(k, j);
            }
            data[row_cursor++] = value;
        }
    }

    if(result == nullptr){
        result = new DenseMatrixT<real>();
    }
    result->set_shallow_data(data, this->_rows, mat->get_cols());

    return true;
}

template<class T>
bool DenseMatrixT<T>::swap(const uint a_row,
                           const uint a_col,
                           const uint b_row,
                           const uint b_col){
    if(!check_range(a_row, a_col) || !check_range(b_row, b_col)){
        return false;
    }

    T value;
    value = get_data(a_row, a_col);
    set_data(get_data(b_row, b_col),a_row, a_col);
    set_data(value, b_row, b_col);

    clear_cache();

    return true;
}
template<class T>
bool DenseMatrixT<T>::swap_row(const uint a, const uint b){

    if(a == b){
        return true;
    }

    if(a >= this->_rows || b >= this->_rows){
        return false;
    }

    T* data = new T[this->_cols];
    memcpy(data, &_data[a * this->_cols], sizeof(T) * this->_cols);
    memcpy(&_data[a * this->_cols], &_data[b * this->_cols], sizeof(T) * this->_cols);
    memcpy(&_data[b * this->_cols], data, sizeof(T) * this->_cols);
    delete[] data;

    clear_cache();

    return true;
}
template<class T>
bool DenseMatrixT<T>::swap_col(const uint a, const uint b){
    if(a >= this->_cols || b >= this->_cols){
        return false;
    }

    if(a == b){
        return true;
    }

    for(int i = 0; i < this->_rows; i++){
        swap(i, a, i, b);
    }

    return true;
}

template<class T>
void DenseMatrixT<T>::transpose(){
    T* data = new T[this->_rows * this->_cols];
    uint new_data_idx = 0;

    for(int i = 0; i < this->_cols; i++){
        for(int j = 0; j < this->_rows; j++){
            data[new_data_idx++] = get_data(j, i);
        }
    }

    set_shallow_data(data, this->_cols, this->_rows);
}

template<class T>
void DenseMatrixT<T>::transpose(BaseMatrixT<T>* result){

    uint new_data_idx = 0;
    T* data = new T[this->_rows * this->_cols];

    for(int i = 0; i < this->_cols; i++){
        for(int j = 0; j < this->_rows; j++){
            data[new_data_idx++] = get_data(j, i);
        }
    }

    if(result == nullptr){
        result = new DenseMatrixT<T>();
    }
    result->set_shallow_data(data, this->_cols, this->_rows);
}

template<class T>
bool DenseMatrixT<T>::det(T* result){
    if(this->_rows != this->_cols){
        return false;
    }

    if(_cache_matrix_det != static_cast<T>(INT_MAX)){
        *result = _cache_matrix_det;
        return true;
    }

    T sum1 = static_cast<T>(0), sum2 = static_cast<T>(0);

    for(int i = 0; i < this->_cols; i++){
        T s = static_cast<T>(1);
        for(int j = 0; j < this->_rows; j++){
            s *= get_data(j, (i+ j) % this->_rows);
        }
        sum1 += s;
    }

    for(int i = 0; i < this->_cols; i++){
        T s = static_cast<T>(1);
        for(int j = 0; j < this->_rows; j++){
            int row = i - j;
            if(row < 0){
                row += this->_rows;
            }
            s *= get_data(j, row);
        }
        sum2 += s;
    }

    _cache_matrix_det = sum1 - sum2;
    *result = _cache_matrix_det;

    return true;
}

template<class T>
bool DenseMatrixT<T>::inverse(BaseMatrixT<real>* result){
    if(this->_rows != this->_cols){
        return false;
    }

    DenseMatrixT<real>* extend_mat = new DenseMatrixT<real>();
    //copy src matrix
    real* data = new real[this->_rows * this->_cols];
    for(int i = 0; i < this->_rows * this->_cols ; i++){
        data[i] = static_cast<real>(_data[i]);
    }
    extend_mat->set_shallow_data(data, this->_rows, this->_cols);

    //extend eye matrix
    DenseEyeMatrixT<real>* eye_mat = new DenseEyeMatrixT<real>(this->_rows);
    extend_mat->extend(eye_mat);
    delete eye_mat;

    //adjust row
    for(int i = 0; i < extend_mat->get_rows(); i++){
        if(extend_mat->get_data(i, i) == static_cast<T>(0)){
            int j;
            for(j = 0; j < extend_mat->get_rows(); j++){
                if(extend_mat->get_data(j, i) != static_cast<T>(0)){
                    extend_mat->swap_row(i, j);
                    break;
                }
            }
            if(j >= extend_mat->get_rows()){//every element is zero in col i, cannot be inverse.
                return false;
            }
        }
    }

    //calc extend matrix
    for(int i = 0; i < extend_mat->get_rows(); i++){

        //all element div the first element,to make diagonal elemnt is 1
        real diagonal_element = extend_mat->get_data(i, i);
        for(int j = 0; j < extend_mat->get_cols(); j++){
            extend_mat->set_data(extend_mat->get_data(i, j)/diagonal_element, i, j);
        }

        //to make the element of other rows in col i is 0
        for(int m = 0; m < extend_mat->get_rows(); m++){
            if(m != i){//skip itself
                real element = extend_mat->get_data(m, i);
                for(int n = 0; n < extend_mat->get_cols(); n++){
                    extend_mat->set_data(extend_mat->get_data(m, n) - extend_mat->get_data(i, n) * element, m, n);
                }
            }
        }
    }

    //calc inverse matrix
    data = new real[this->_rows * this->_cols];
    uint new_data_idx = 0;
    for(int i = 0; i < extend_mat->get_rows(); i++){
        for(int j = this->_cols; j < extend_mat->get_cols(); j++){
            data[new_data_idx++] = extend_mat->get_data(i, j);
        }
    }

    if(result == nullptr){
        result = new DenseMatrixT<real>();
    }
    result->set_shallow_data(data, this->_rows, this->_cols);
    delete extend_mat;

    return true;
}

template<class T>
void DenseMatrixT<T>::display(){
    printf("[%d*%d][\n", this->_rows, this->_cols);
    for(int i = 0; i < this->_rows; i++){
        for(int j = 0; j < this->_cols; j++){
            if(typeid(T) == typeid(real)){
                printf("%f\t", get_data(i, j));
            }else{
                printf("%d\t", get_data(i, j));
            }
        }
        printf("\n");
    }
    printf("]\n");
}


template<class T>
LabeledDenseMatrixT<T>::LabeledDenseMatrixT(){
    _labels = nullptr;
}

template<class T>
LabeledDenseMatrixT<T>::LabeledDenseMatrixT(const T* data,
                                            const T* labels,
                                            const uint rows,
                                            const uint cols) : DenseMatrixT<T>(data, rows, cols){
    _labels = new T[rows];
    memcpy(_labels, labels, sizeof(T) * rows);
}


template<class T>
LabeledDenseMatrixT<T>::~LabeledDenseMatrixT(){
    if(_labels != nullptr){
        delete[] _labels;
        _labels = nullptr;
    }
}

template<class T>
DenseColMatrixT<T>* LabeledDenseMatrixT<T>::get_labels(){
    return new DenseColMatrixT<T>(_labels, this->_rows);
}

template<class T>
void LabeledDenseMatrixT<T>::set_data(const T* data,
                                      const T* labels,
                                      const uint rows,
                                      const uint cols){
    DenseMatrixT<T>::set_data(data, rows, cols);

    if(_labels != nullptr){
        delete[] _labels;
    }

    _labels = new T[rows];
    memcpy(_labels, labels, sizeof(T) * rows);
}

template<class T>
void LabeledDenseMatrixT<T>::set_shallow_data(T* data,
                                              T* labels,
                                              const uint rows,
                                              const uint cols){
    DenseMatrixT<T>::set_shallow_data(data, rows, cols);

    if(_labels != nullptr){
        delete[] _labels;
    }

    _labels = labels;
}

template<class T>
void LabeledDenseMatrixT<T>::display(){
    for(uint i = 0; i < this->_cols; i++){
        printf("col:%d\t", i);
    }
    printf("label\n");
    for(uint i = 0; i < this->_rows; i++){
        for(uint j = 0; j < this->_cols; j++){
            if(typeid(T) == typeid(real)){
                printf("%f\t", this->get_data(i, j));
            }else{
                printf("%d\t", this->get_data(i, j));
            }
        }
        if(typeid(T) == typeid(real)){
            printf("%f\n", get_label(i));
        }else{
            printf("%d\n", get_label(i));
        }
    }
}


template class DenseMatrixT<int>;
template class DenseMatrixT<real>;

template class DenseColMatrixT<int>;
template class DenseColMatrixT<real>;

template class LabeledDenseMatrixT<int>;
template class LabeledDenseMatrixT<real>;

}//namespace algebra
}//namespace ccma
