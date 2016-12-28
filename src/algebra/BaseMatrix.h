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
#include <string.h>
#include <limits.h>
#include "utils/TypeDef.h"

namespace ccma{
namespace algebra{

template<class T>
class BaseMatrixT{
public:
    BaseMatrixT(){}
    BaseMatrixT(const uint rows, const uint cols){
        _rows = rows;
        _cols = cols;
    }

    virtual ~BaseMatrixT() = default;

    inline uint get_rows() const { return _rows;}
    inline uint get_cols() const { return _cols;}

    virtual BaseMatrixT<T>* copy_data() = 0;

    virtual T* get_data() const = 0;
    virtual void set_data(const T* data,
                          const uint rows,
                          const uint cols) = 0;
    virtual void set_shallow_data(T* data,
                                  const uint rows,
                                  const uint cols) = 0;

    virtual T get_data(const uint idx) const = 0;
    virtual bool set_data(const T& value, const uint idx) = 0;

    virtual T get_data(const uint row, const uint col) const = 0;
    virtual bool set_data(const T& data,
                          const uint row,
                          const uint col) = 0;

    virtual BaseMatrixT<T>* get_row_data(const uint row) const = 0;
    virtual bool set_row_data(BaseMatrixT<T>* mat, const int row) = 0;

    virtual bool extend(const BaseMatrixT<T>* mat) = 0;

    virtual bool add(const BaseMatrixT<T>* mat) = 0;
    virtual bool add(const BaseMatrixT<T>* mat, BaseMatrixT<T>* result) = 0;

    virtual bool subtract(const BaseMatrixT<T>* mat) = 0;
    virtual bool subtract(const BaseMatrixT<T>* mat, BaseMatrixT<T>* result) = 0;


    virtual bool product(const T value) = 0;
    virtual bool product(const T value, BaseMatrixT<T>* result) = 0;
    virtual bool product(const BaseMatrixT<T>* mat, BaseMatrixT<T>* result) = 0;
    virtual bool product(const BaseMatrixT<int>* mat, BaseMatrixT<real>* result) = 0;

    virtual bool swap(const uint a_row,
                      const uint a_col,
                      const uint b_row,
                      const uint b_col) = 0;
    virtual bool swap_row(const uint a, const uint b) = 0;
    virtual bool swap_col(const uint a, const uint b) = 0;

    virtual void transpose() = 0;
    virtual void transpose(BaseMatrixT<T>* result) = 0;

    virtual bool det(T* result) = 0;

    virtual bool inverse(BaseMatrixT<real>* result) = 0;

    virtual void display() = 0;
protected:
    uint _rows;
    uint _cols;

};//class BaseMatrixT


template<class T>
class DenseMatrixT : public BaseMatrixT<T>{
public:
    DenseMatrixT();
    DenseMatrixT(const T* data,
                 const uint rows,
                 const uint cols);
    ~DenseMatrixT();

    DenseMatrixT<T>* copy_data();

    T* get_data() const;
    void set_data(const T* data,
                  const uint rows,
                  const uint cols);

    T get_data(const uint idx) const;
    bool set_data(const T& value, const uint idx);

    T get_data(const uint row, const uint col) const;
    bool set_data(const T& data,
                  const uint row,
                  const uint col);
    void set_shallow_data(T* data,
                          const uint rows,
                          const uint cols);


    DenseMatrixT<T>* get_row_data(const uint row) const;
    bool set_row_data(BaseMatrixT<T>* mat, const int row);

    bool extend(const BaseMatrixT<T>* mat);

    bool add(const BaseMatrixT<T>* mat);
    bool add(const BaseMatrixT<T>* mat, BaseMatrixT<T>* result);

    bool subtract(const BaseMatrixT<T>* mat);
    bool subtract(const BaseMatrixT<T>* mat, BaseMatrixT<T>* result);


    bool product(const T value);
    bool product(const T value, BaseMatrixT<T>* result);
    bool product(const BaseMatrixT<T>* mat, BaseMatrixT<T>* result);
    bool product(const BaseMatrixT<int>* mat, BaseMatrixT<real>* result);

    bool swap(const uint a_row,
              const uint a_col,
              const uint b_row,
              const uint b_col);
    bool swap_row(const uint a, const uint b);
    bool swap_col(const uint a, const uint b);

    void transpose();
    void transpose(BaseMatrixT<T>* result);

    bool det(T* result);

    bool inverse(BaseMatrixT<real>* result);

    void display();
protected:
    T* _data;

    inline bool check_range(const uint idx){
        return idx < this->_rows * this->_cols;
    }
    inline bool check_range(const uint row, const uint col){
        return row < this->_rows && col < this->_cols;
    }

private:
    T _cache_matrix_det;

    inline void clear_cache(){
        T _cache_matrix_det = static_cast<T>(INT_MAX);
    }
};


template<class T>
class DenseMatrixMNT : public DenseMatrixT<T>{
public:
    DenseMatrixMNT(const T* data, uint rows, uint cols):DenseMatrixT<T>(data, rows, cols){}

    DenseMatrixMNT(uint rows, uint cols, T value){
        this->_data = new T[rows * cols];

        //memset only support 0 or -1.
        if(value == static_cast<T>(0) || value == static_cast<T>(-1)){
            memset(this->_data, value, sizeof(T) * rows * cols);
        }else{
            for(uint i = 0; i < rows * cols; i++){
                this->_data[i] = value;
            }
        }
        this->_rows = rows;
        this->_cols = cols;
    }
};

template<class T>
class DenseColMatrixT : public DenseMatrixMNT<T>{
public:
    DenseColMatrixT(uint rows, T value):DenseMatrixMNT<T>(rows, 1, value){}
    DenseColMatrixT(const T* data, uint rows):DenseMatrixMNT<T>(data, rows, 1){}
};

template<class T>
class DenseRowMatrixT : public DenseMatrixMNT<T>{
public:
    DenseRowMatrixT(uint cols, T value):DenseMatrixMNT<T>(1, cols, value){}
    DenseRowMatrixT(const T* data, uint cols):DenseMatrixMNT<T>(data, 1, cols){}
};

template<class T>
class DenseOneMatrixT : public DenseMatrixMNT<T>{
public:
    explicit DenseOneMatrixT(uint rows):DenseMatrixMNT<T>(rows, 1, 1){}
};

template<class T>
class DenseZeroMatrixT : public DenseMatrixMNT<T>{
public:
    explicit DenseZeroMatrixT(uint rows):DenseMatrixMNT<T>(rows, 1, 0){}
};

template<class T>
class DenseEyeMatrixT : public DenseMatrixT<T>{
public:
    explicit DenseEyeMatrixT(uint size){

        T* data = new T[size * size];
        memset(data, 0, sizeof(T) * size * size);

        for(uint i = 0; i < size; i++){
            data[i * size + i] = static_cast<T>(1);
        }

        this->set_shallow_data(data, size, size);
    }
};


template<class T>
class LabeledDenseMatrixT : public DenseMatrixT<T>{
public:
    LabeledDenseMatrixT();
    LabeledDenseMatrixT(const T* data,
                        const T* labels,
                        const uint rows,
                        const uint cols);

    ~LabeledDenseMatrixT();

    inline T get_label(uint idx) const{
        return _labels[idx];
    }

    DenseColMatrixT<T>* get_labels();

    void set_data(const T* data,
                  const T* labels,
                  const uint rows,
                  const uint cols);

    void set_shallow_data(T* data,
                          T* labels,
                          const uint rows,
                          const uint cols);

    void display();
protected:
    T* _labels = nullptr;
};

/*
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
    DenseEyeMatrixT<real>* eye_mat = new DenseEyeMatrixT<real>(_rows);
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

*/

}//namespace algebra
}//namespace ccma

#endif //_CCMA_ALGEBRA_BASEMATRIX_H_
