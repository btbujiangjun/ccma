/**********************************************
* Author: Jun Jiang - jiangjun4@sina.com
* Created: 2016-11-22 11:30
* Last modified: 2016-11-22 11:30
* Filename: BaseMatrix.h
* Description: Base Data Structure of training/predict data
**********************************************/

#ifndef _CCMA_ALGEBRA_BASEMATRIX_H_
#define _CCMA_ALGEBRA_BASEMATRIX_H_

#include <cmath>
#include <typeinfo>
#include <limits.h>
#include <unordered_map>
#include "utils/TypeDef.h"

namespace ccma{
namespace algebra{

template<class T>
class CCMap : public std::unordered_map<T, uint>{
public:
    CCMap():std::unordered_map<T, uint>(), _value(0){}

    uint get_max_value(){
        if(_value == 0){
            calc();
        }
        return _value;
    }
    T get_max_key(){
        if(_value == 0){
            calc();
        }
        return _key;
    }
private:
    uint _value;
    T _key;

    void calc(){
        typename std::unordered_map<T, uint>::const_iterator it;
        for(it = this->begin(); it != this->end(); it++){
            if(it->second > _value){
                _value = it->second;
                _key = it->first;
            }
        }
    }
};//class CCMap

template<class T>
class BaseMatrixT{
public:
    BaseMatrixT();

    BaseMatrixT(uint rows,
                uint cols,
                const T* data);

    ~BaseMatrixT();

    CCMap<T>* get_feature_map(uint feature_id) const;

    uint get_rows() const{
        return _rows;
    }
    uint get_cols() const{
        return _cols;
    }

    void set_data(T* data, uint rows, uint cols){
        if(_data != nullptr){
            delete[] _data;
            _data = nullptr;
            clear_cache();
        }
        _data = new T[rows*cols];
        for(int i = 0; i < rows * cols; i++){
            _data[i] = data[i];
        }
        _rows = rows;
        _cols = cols;
    }
    T get_data(uint idx) const{
        return _data[idx];
    }

    int set_data(T value, uint idx){
        if( idx >= _rows * _cols){
            return -1;
        }
        _data[idx] = value;
        return 1;
    }

    T get_data(uint row, uint col){
        //todo range check
        return _data[row * _cols + col];
    }

    int set_data(T value, uint row, uint col){
        if(row >= _rows || col >= _cols){
            return -1;
        }
        _data[row * _cols + col] = value;
        return 1;
    }

    int get_row_data(uint row, BaseMatrixT<T>* result){
        if(row >= _rows){
            return -1;
        }

        T* data = new T[_cols];
        for(int i = 0; i < _cols; i++){
            data[i] = get_data(row, i);
        }
        result->set_data(data, 1, _cols);
        delete[] data;

        return 1;
    }

    int set_row_data(const BaseMatrixT<T>* mat, uint row){
        if(row >= _rows || mat->get_rows() != 1 || mat->get_cols() != _cols){
            return -1;
        }
        for(int i = 0; i < _cols; i++){
            set_data(row * _cols + i);
        }
        return 1;
    }

    int extend(BaseMatrixT<T>* mat){
        if(_rows != mat->get_rows()){
            return false;
        }

        int new_data_idx = 0;
        T value;
        uint new_cols = _cols + mat->get_cols();
        T* data = new T[new_cols * _rows];
        for(int i = 0; i < _rows; i++){
            for(int j = 0; j < new_cols; j++){
                if(j < _cols){
                    value = get_data(i, j);
                }else{
                    value = mat->get_data(i, (j - _cols));
                }
                data[new_data_idx++] = value;
            }
        }
        set_data(data, _rows, new_cols);
        delete[] data;

        return true;
    }

    int add(BaseMatrixT<T>* mat);
    int add(BaseMatrixT<T>* mat, BaseMatrixT<T>* result);

    int sum(BaseMatrixT<T>* data);

    int subtract(BaseMatrixT<T>* mat);
    int subtract(BaseMatrixT<T>* mat, BaseMatrixT<T>* result);

    int product(BaseMatrixT<T>* mat, BaseMatrixT<T>* result);
    int product(int value);
    int product(int value, BaseMatrixT<T>* result);
    int product(uint value);
    int product(uint value, BaseMatrixT<T>* result);
    int product(real value);
    int product(real value, BaseMatrixT<real>* result);

    int inner_product(BaseMatrixT<T>* mat, BaseMatrixT<T>* result){
        if(mat->get_rows() != _cols || mat->get_cols() != _rows){
            return -1;
        }

        T* data = new T[_rows * _rows];
        for(int i = 0; i < _rows; i++){
            for(int j = 0; j < mat->get_rows(); j++){
                T value = (T)0;
                for(int k = 0; k < _cols; k++){
                    value += get_data(i, k) * mat->get_data(k, j);
                }
                data[i * _rows + j] = value;
            }
        }
        result->set_data(data, _cols, _cols);
        delete[] data;

        return 1;
    }

    int dot_product(BaseMatrixT<T>* mat, BaseMatrixT<T>* result){
        if(mat->get_rows() != _cols || mat->get_cols() != 1){
            return -1;
        }

        T* data = new T[_rows];
        for(int i = 0; i < _rows; i++){
            T value = (T)0;
            for(int j = 0; j < _cols; j++){
                value += get_data(i, j) * mat->get_data(0, j);
            }
            data[i] = value;
        }
        result->set_data(data, _rows, 1);
        delete[] data;

        return 1;
    }

    bool swap(uint a_row, uint a_col, uint b_row, uint b_col);
    bool swap_row(uint a, uint b);
    bool swap_col(uint a, uint b);

    void transpose(BaseMatrixT<T>* result);

    bool det(T* result);

    bool inverse(BaseMatrixT<real>* result);
    //bool inverse2(BaseMatrixT<real>* result);

    void copy_data(BaseMatrixT<T>* result);

    void display(){
        printf("[%d*%d][\n", _rows, _cols);
        for(int i = 0; i < _rows; i++){
            for(int j = 0; j < _cols; j++){
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

protected:
    uint _rows;
    uint _cols;
    T* _data;

private:
    std::unordered_map<uint, CCMap<T>* >* _cache_feature_map;
    T _cache_matrix_det = (T)INT_MAX;

    void clear_cache(){
        CCMap<T>* item;
        typename std::unordered_map<uint, CCMap<T>*>::iterator it;
        it = _cache_feature_map->begin();

        while(it != _cache_feature_map->end()){
            item = it->second;
            it = _cache_feature_map->erase(it);
            delete item;
        }
        _cache_feature_map->clear();

        _cache_matrix_det = (T)INT_MAX;
    }

};//class BaseMatrixT


template<class T>
class MatrixMN : public BaseMatrixT<T>{
public:
    MatrixMN(uint rows, uint cols, T value){
        T* data = new T[rows * cols];
        for(int i = 0; i < rows * cols; i++){
            data[i] = value;
        }
        this->set_data(data, rows, cols);
        delete[] data;
    }
};

template<class T>
class ColMatrix : public MatrixMN<T>{
public:
    ColMatrix(uint rows, T value):MatrixMN<T>(rows, 1, value){}
};

template<class T>
class RowMatrix : public MatrixMN<T>{
public:
    RowMatrix(uint cols, T value):MatrixMN<T>(1, cols, value){}
};

template<class T>
class OneMatrix : public MatrixMN<T>{
public:
    OneMatrix(uint rows):MatrixMN<T>(rows, 1, 1){}
};

template<class T>
class ZeroMatrix : public MatrixMN<T>{
public:
    ZeroMatrix(uint rows):MatrixMN<T>(rows, 1, 0){}
};

template<class T>
class EyeMatrix : public BaseMatrixT<T>{
public:
    explicit EyeMatrix(uint size){
        T* data = new T[size * size];
        uint data_idx = 0;

        for(int i = 0; i < size; i++){
            for(int j = 0; j < size; j++){
                if(i == j){
                    data[data_idx++] = (T)1;
                }else{
                    data[data_idx++] = (T)0;
                }
            }
        }

        this->set_data(data, size, size);
        delete[] data;
    }
};

template<class T, class LT, class FT>
class LabeledMatrixT : public BaseMatrixT<T>{
public:
    LabeledMatrixT();

    LabeledMatrixT(uint rows,
                  uint cols,
                  T* data,
                  const LT* label_data,
                  const FT* feature_label);

    ~LabeledMatrixT();

    CCMap<LT>* get_label_map() const;

    real get_shannon_entropy();

    void set_data(T* data,
                  LT* label_data,
                  FT* feature_label,
                  uint rows,
                  uint cols){

        BaseMatrixT<T>::set_data(data, rows, cols);

        _shannon_entropy =(real)0;
        _cache_label_map->clear();

        if(_label_data != nullptr){
            delete[] _label_data;
            _label_data = nullptr;
        }
        _label_data = new LT[rows];
        for(int i = 0; i < rows; i++){
            _label_data[i] = label_data[i];
        }

        if(_feature_label != nullptr){
            delete[] _feature_label;
            _feature_label = nullptr;
        }
        _feature_label = new FT[cols];
        for(int i = 0; i < cols; i++){
            _feature_label[i] = feature_label[i];
        }
    }

    LT get_label(uint i) const{
        return _label_data[i];
    }

    int get_labels(BaseMatrixT<LT>* result){
        LT* data = new LT[this->_rows];
        for(int i = 0; i < this->_rows; i++){
            data[i] = _label_data[i];
        }

        result->set_data(data, this->_rows, 1);
        delete[] data;

        return 1;
    }

    int set_label(LT value, uint i){
        if(i >= this->_rows){
            return -1;
        }
        _label_data[i] = value;
        return 1;
    }

    FT get_feature_label(uint i) const{
        return _feature_label[i];
    }

    uint split(uint feature_id,
               T split_value,
               LabeledMatrixT<T, LT, FT>* sub_matrix);

    int subtract_label(BaseMatrixT<LT>* mat);
    int subtract_label(LT value, uint row);

protected:
    LT* _label_data;
    FT* _feature_label;

private:
    real _shannon_entropy;
    CCMap<LT>* _cache_label_map;
};//class LabeledMatrixT



typedef BaseMatrixT<real> Matrix;
typedef BaseMatrixT<int> IMatrix;


template<class T>
BaseMatrixT<T>::BaseMatrixT(){
    _rows = 0;
    _cols = 0;
    _data = nullptr;
    _cache_feature_map = new std::unordered_map<uint, CCMap<T>* >();
}

template<class T>
BaseMatrixT<T>::BaseMatrixT(uint rows,
                            uint cols,
                            const T* data):
    _rows(rows), _cols(cols){

    _data = new T[rows * cols];
    for(int i = 0; i < rows * cols; i++){
        _data[i] = data[i];
    }

    _cache_feature_map = new std::unordered_map<uint, CCMap<T>* >();
}

template<class T>
BaseMatrixT<T>::~BaseMatrixT(){
    _rows = 0;
    _cols = 0;

    if(_data != nullptr){
        delete[] _data;
        _data = nullptr;
    }

    clear_cache();

    delete _cache_feature_map;
    _cache_feature_map = nullptr;
}

template<class T>
CCMap<T>* BaseMatrixT<T>::get_feature_map(uint feature_id) const{

    typename std::unordered_map<uint, CCMap<T>* >::const_iterator it;
    it = _cache_feature_map->find(feature_id);

    if(it != _cache_feature_map->end()){//retrieve from cache
        return it->second;
    }

    CCMap<T>* uv = new CCMap<T>();
    if(feature_id >= _cols){//range check
        return uv;
    }

    typename CCMap<T>::iterator itv;
    for(int i = 0; i < _rows; i++){
        T d = _data[_cols * i + feature_id];
        itv = uv->find(d);
        if(itv== uv->end()){
            uv->insert(std::make_pair(d, 1));
        }else{
            itv->second += 1;
        }
    }
    _cache_feature_map->insert(std::make_pair(feature_id, uv));

    return uv;
}

template<class T>
int BaseMatrixT<T>::add(BaseMatrixT<T>* mat){
    if(_rows != mat->get_cols() || _cols != mat->get_rows()){
        return -1;
    }

    for(int i = 0; i < _rows * _cols; i++){
        _data[i] += mat->get_data(i);
    }

    return 1;
}

template<class T>
int BaseMatrixT<T>::add(BaseMatrixT<T>* mat, BaseMatrixT<T>* result){
    if(_rows != mat->get_cols() || _cols != mat->get_rows()){
        return -1;
    }

    T* data = new T[_rows * _cols];
    for(int i = 0; i < _rows * _cols; i++){
        data[i] = _data[i] + mat->get_data(i);
    }

    result->set_data(data, _rows, _cols);
    delete[] data;

    return 1;
}

template<class T>
int BaseMatrixT<T>::sum(BaseMatrixT<T>* data){

    if(_rows != data->get_rows() || _cols != data->get_cols()){
        return -1;
    }
    for(int i = 0; i < _rows * _cols; i++){
        _data[i] += data->get_data(i);
    }

    return 1;
}

/*

template<class T>
int BaseMatrixT<T>::col_sum(BaseMatrixT<T>* result){
    T* data = new T[_rows];

    for(int i = 0; i < _rows; i++){
        T value = (T)0;
        for(int j = 0; j < _cols; j++){
            value += get_data(i, j);
        }
        data[i] = value;
    }

    result->set_data(data, _rows, 1);
    delete[] data;

    return 1;
}

*/


template<class T>
int BaseMatrixT<T>::subtract(BaseMatrixT<T>* mat){
    if(_rows != mat->get_rows() || _cols != mat->get_cols()){
        return -1;
    }

    for(int i = 0; i < _rows * _cols; i++){
        _data[i] -= mat->get_data(i);
    }

    return 1;
}

template<class T>
int BaseMatrixT<T>::subtract(BaseMatrixT<T>* mat, BaseMatrixT<T>* result){
    if(_rows != mat->get_rows() || _cols != mat->get_cols()){
        return -1;
    }

    T* data = new T[_rows * _cols];
    for(int i = 0; i < _rows * _cols; i++){
        data[i] = _data[i] - mat->get_data(i);
    }

    result->set_data(data, _rows, _cols);
    delete[] data;

    return 1;
}

template<class T>
int BaseMatrixT<T>::product(BaseMatrixT<T>* mat, BaseMatrixT<T>* result){
    if(_rows != mat->get_cols() || _cols != mat->get_rows()){
        return -1;
    }

    uint new_data_idx = 0;
    T* data = new T[this->_rows * mat->get_cols()];
    for(int i = 0; i < this->_rows; i++){
        for(int j = 0; j < this->_rows; j++){
            T value = (T)0;
            for(int k = 0; k < this->_cols; k++){
                value += this->get_data(i, k) * mat->get_data(k, j);
            }
            data[new_data_idx++] = value;
        }
    }

    result->set_data(data, _rows, mat->get_cols());
    delete[] data;

    return 1;
}

template<class T>
int BaseMatrixT<T>::product(int value){

    for(int i = 0; i < _rows * _cols; i++){
        _data[i] *= (T)value;
    }

    return 1;
}

template<class T>
int BaseMatrixT<T>::product(int value, BaseMatrixT<T>* result){

    T* data = new T[_rows * _cols];
    for(int i = 0; i < _rows * _cols; i++){
        data[i] = _data[i] * (T)value;
    }
    result->set_data(data, _rows, _cols);
    delete[] data;

    return 1;
}

template<class T>
int BaseMatrixT<T>::product(uint value){
    return product((int)value);
}

template<class T>
int BaseMatrixT<T>::product(uint value, BaseMatrixT<T>* result){
    return product((int)value, result);
}

template<class T>
int BaseMatrixT<T>::product(real value, BaseMatrixT<real>* result){

    real* data = new real[_rows * _cols];
    for(int i = 0; i < _rows * _cols; i++){
        data[i] = (real)_data[i] * value;
    }
    result->set_data(data, _rows, _cols);
    delete[] data;

    return 1;
}

template<class T>
int BaseMatrixT<T>::product(real value){

    for(int i = 0; i < _rows * _cols; i++){
        _data[i] *= (T)value;
    }

    return 1;
}

template<class T>
bool BaseMatrixT<T>::swap(uint a_row, uint a_col, uint b_row, uint b_col){
    if(a_row >= _rows || b_row >= _rows || a_col >= _cols || b_col >= _cols){
        return false;
    }

    T value;
    value = get_data(a_row, a_col);
    set_data(get_data(b_row, b_col),a_row, a_col);
    set_data(value, b_row, b_col);

    return true;
}

template<class T>
bool BaseMatrixT<T>::swap_row(uint a, uint b){
    if(a == b){
        return true;
    }

    if(a >= _rows || b >= _rows){
        return false;
    }

    for(int i = 0; i < _cols; i++){
        swap(a, i, b, i);
    }

    return true;
}

template<class T>
bool BaseMatrixT<T>::swap_col(uint a, uint b){
    if(a >= _cols || b >= _cols){
        return false;
    }

    if(a == b){
        return true;
    }

    for(int i = 0; i < _rows; i++){
        swap(i, a, i, b);
    }

    return true;
}



template<class T>
void BaseMatrixT<T>::transpose(BaseMatrixT<T>* result){
    T* data = new T[_rows * _cols];
    uint new_data_idx = 0;

    for(int i = 0; i < _cols; i++){
        for(int j = 0; j < _rows; j++){
            data[new_data_idx++] = get_data(j, i);
        }
    }

    result->set_data(data, _cols, _rows);
    delete[] data;
}

template<class T>
bool BaseMatrixT<T>::det(T* result){
    if(_rows != _cols){
        return false;
    }

    if(_cache_matrix_det != (T)INT_MAX){
        *result = _cache_matrix_det;
        return true;
    }

    T sum1 = (T)0, sum2 = (T)0;

    for(int i = 0; i < _cols; i++){
        T s = (T)1;
        for(int j = 0; j < _rows; j++){
            s *= get_data(j, (i+ j) % _rows);
        }
        sum1 += s;
    }

    for(int i = 0; i < _cols; i++){
        T s = (T)1;
        for(int j = 0; j < _rows; j++){
            int row = i - j;
            if(row < 0){
                row += _rows;
            }
            s *= get_data(j, row);
        }
        sum2 += s;
    }

    _cache_matrix_det = sum1 - sum2;
    *result = _cache_matrix_det;

    return true;
}

/*
template<class T>
bool BaseMatrixT<T>::inverse2(BaseMatrixT<real>* result){
    if(_rows != _cols){
        return false;
    }

    //copy src matrix
    real* data = new real[_rows * _cols];
    if(typeid(T) == typeid(real)){
        memcpy(data, _data, sizeof(T)*(_rows * _cols));
    }else{
        for(int i = 0; i < _rows * _cols ; i++){
            data[i] = (real)_data[i];
        }
    }
    result->set_data(data, _rows, _cols);

    uint* swaped_rows = new uint[_rows];
    uint* swaped_cols = new uint[_cols];

    //iterator every row
    for(int i = 0; i < _rows; i++){
        //step 1: find max of lower right corner,全选主元
        T max_value = (T)0, value;
        for(int j = i; j< _rows; j++){
            for(int k = i; k < _cols; k++){
                value = abs(result->get_data(j, k));
                if(value > max_value){
                    max_value = value;
                    swaped_rows[i] = j;
                    swaped_cols[i] = k;
                }
            }
        }

        //step 2: move max_value to i*i
        result->swap_row(i, swaped_rows[i]);
        result->swap_col(i, swaped_cols[i]);

        //step 3: calc inverse matrix m(i, i) = 1 / m(i, i)
        real mii = (real)1/result->get_data(i, i);
        result->set_data(mii, i, i);
        //step 4: m(i, j) = m(i, j)* m(i, i)，j = 0, 1, ..., n-1；j != i
        for(int j = 0; j < _cols; j++){// row i multiply mii
            if(i != j){
                result->set_data(result->get_data(i, j) * mii, i, j);
            }
        }

        //step 5: m(j, k) -= m(j, i) * m(i, k)，j, k = 0, 1, ..., n-1；k, k != i
        for(int j = 0; j < _rows; j++){
            if(i == j ){
                continue;
            }
            for(int k = 0; k < _cols; k++){
                if(i != k){
                    result->set_data(result->get_data(j, k) - result->get_data(j, i) * result->get_data(i, k), j, k);
                }
            }
        }

        //step 6: m(j, i) = -m(j, i) * m(i, i)，j = 0, 1, ..., n-1；i != j
        for(int j = 0; j < _rows; j++){
            if(i != j){
                result->set_data(-result->get_data(j, i) * result->get_data(i, i) , j, i);
            }
        }
    }

    //step 7: recovery swap rows
    for(int i = _rows -1 ; i >= 0; i--){
        result->swap_col(i, swaped_rows[i]);
        result->swap_row(i, swaped_cols[i]);
    }
    delete[] swaped_rows;
    delete[] swaped_cols;

    return true;
}
*/


template<class T>
bool BaseMatrixT<T>::inverse(BaseMatrixT<real>* result){
    if(_rows != _cols){
        return false;
    }


    BaseMatrixT<T>* extend_mat = new BaseMatrixT<T>();
    //copy src matrix
    real* data = new real[_rows * _cols];
    if(typeid(T) == typeid(real)){
        memcpy(data, _data, sizeof(T)*(_rows * _cols));
    }else{
        for(int i = 0; i < _rows * _cols ; i++){
            data[i] = (real)_data[i];
        }
    }
    extend_mat->set_data(data, _rows, _cols);
    delete[] data;

    //extend eye matrix
    EyeMatrix<real>* eye_mat = new EyeMatrix<real>(_rows);
    extend_mat->extend(eye_mat);
    delete eye_mat;

    //adjust row
    for(int i = 0; i < extend_mat->get_rows(); i++){
        if(extend_mat->get_data(i, i) == (T)0){
            int j;
            for(int j = 0; j < extend_mat->get_rows(); j++){
                if(extend_mat->get_data(j, i) != (T)0){
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
    data = new real[_rows * _cols];
    uint new_data_idx = 0;
    for(int i = 0; i < extend_mat->get_rows(); i++){
        for(int j = _cols; j < extend_mat->get_cols(); j++){
            data[new_data_idx++] = extend_mat->get_data(i, j);
        }
    }
    result->set_data(data, _rows, _cols);
    delete[] data;

    delete extend_mat;

    return true;
}


template<class T>
void BaseMatrixT<T>::copy_data(BaseMatrixT<T>* result){
    T* data = new T[_rows * _cols];

    for(int i = 0; i < _rows * _cols; i++){
        data[i] = _data[i];
    }

    result->set_data(data, _rows, _cols);

    delete[] data;
}

template<class T, class LT, class FT>
LabeledMatrixT<T, LT, FT>::LabeledMatrixT(){
    _label_data = nullptr;
    _feature_label = nullptr;
    _shannon_entropy = (real)0;
    _cache_label_map = new CCMap<LT>();
}

template<class T, class LT, class FT>
LabeledMatrixT<T, LT, FT>::LabeledMatrixT(uint rows,
                            uint cols,
                            T* data,
                            const LT* label_data,
                            const FT* feature_label):
    BaseMatrixT<T>(rows, cols, data){
    _label_data = new LT[rows];
    _feature_label = new FT[cols];
    _shannon_entropy = (real)0;
    for(int i = 0; i < rows; i++){
        _label_data[i] = label_data[i];
    }
    for(int i = 0; i < cols; i++){
        _feature_label[i] = feature_label[i];
    }
    _cache_label_map = new CCMap<LT>();
}

template<class T, class LT, class FT>
LabeledMatrixT<T, LT, FT>::~LabeledMatrixT(){
    if(_label_data != nullptr){
        delete[] _label_data;
        _label_data = nullptr;
    }
    if(_feature_label != nullptr){
        delete[] _feature_label;
        _feature_label = nullptr;
    }
    _cache_label_map->clear();
    delete(_cache_label_map);
    _cache_label_map = nullptr;
}

template<class T, class LT, class FT>
CCMap<LT>* LabeledMatrixT<T, LT, FT>::get_label_map() const{
    if( _cache_label_map->size() > 0){
        return _cache_label_map;
    }
    typename CCMap<LT>::iterator it;
    for(int i = 0; i < this->get_rows(); i++){
        it = _cache_label_map->find(_label_data[i]);
        if(it == _cache_label_map->end()){
            _cache_label_map->insert(std::make_pair(_label_data[i], 1));
        }else{
            it->second += 1;
        }
    }
    return _cache_label_map;
}

template<class T, class LT, class FT>
real LabeledMatrixT<T, LT, FT>::get_shannon_entropy(){
    if(_shannon_entropy != (real)0){
        return _shannon_entropy;
    }
    real ent = 0;
    CCMap<LT>* vm = get_label_map();
    typename CCMap<LT>::const_iterator it;
    for(it = vm->begin(); it != vm->end(); it++){
        real prob = (real)it->second/this->get_rows();
        ent -= prob * (log(prob)/log(2));
    }
    _shannon_entropy = ent;
    return ent;
}

template<class T, class LT, class FT>
uint LabeledMatrixT<T, LT, FT>::split(uint feature_id,
                           T split_value,
                           LabeledMatrixT<T, LT, FT>* sub_matrix){
    CCMap<T>* uvm = this->get_feature_map(feature_id);
    typename CCMap<T>::const_iterator it;
    it = uvm->find(split_value);
    if(it == uvm->end()){
        return -1;
    }

    uint new_rows = it->second;
    uint new_cols = this->_cols - 1;

    T* new_data = new T[new_rows * new_cols];
    LT* new_label_data = new LT[new_rows];
    FT* new_feature_label = new FT[new_cols];

    uint new_data_idx = 0;
    uint new_label_idx = 0;
    for(int i = 0; i < this->_rows; i++){
        if (this->get_data(i, feature_id) != split_value){
            continue;
        }
        for(int j = 0; j < this->_cols; j++){
            if(j != feature_id){//skip self
                new_data[new_data_idx++] = this->get_data(i, j);
            }
        }
        new_label_data[new_label_idx++] = _label_data[i];
    }

    uint new_feature_label_idx = 0;
    for(int i = 0; i < this->_cols; i++){
        if(i != feature_id){//skip self
            new_feature_label[new_feature_label_idx++] = this->_feature_label[i];
        }
    }

    sub_matrix->set_data(new_data, new_label_data, new_feature_label, new_rows, new_cols);

    delete[] new_data;
    delete[] new_label_data;
    delete[] new_feature_label;

    return 1;
}

template<class T, class LT, class FT>
int LabeledMatrixT<T, LT, FT>::subtract_label(BaseMatrixT<LT>* mat){
    if(mat->get_cols() != 1 || mat->get_rows() != this->_rows){
        return -1;
    }

    for(int i = 0; i < this->_rows; i++){
        _label_data[i] -= mat->get_data(i, 0);
    }
}

template<class T, class LT, class FT>
int LabeledMatrixT<T, LT, FT>::subtract_label(LT value, uint row){
    if(row >= this->_rows){
        return -1;
    }
    return set_label(get_label(row) - value, row);
}

}//namespace algebra
}//namespace ccma

#endif //_CCMA_ALGEBRA_BASEMATRIX_H_
