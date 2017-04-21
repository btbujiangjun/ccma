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
#include <unordered_map>
#include "utils/TypeDef.h"

namespace ccma{
namespace algebra{

template<class T>
class BaseMatrixT{
public:
    BaseMatrixT(){
        _rows = 0;
        _cols = 0;
    }
    BaseMatrixT(const uint rows, const uint cols){
        _rows = rows;
        _cols = cols;
    }

    virtual ~BaseMatrixT() = default;

    inline uint get_rows() const { return this->_rows;}
    inline uint get_cols() const { return this->_cols;}
    inline uint get_sizes() const { return this->_rows * this->_cols;}

    virtual BaseMatrixT<T>* clone() = 0;

    virtual T* get_data() const = 0;
    virtual void set_data(const T* data,
                          const uint rows,
                          const uint cols) = 0;

    void set_data(const BaseMatrixT<T>* mat){set_data(mat->get_data(), mat->get_rows(), mat->get_cols());}

    virtual void set_shallow_data(T* data,
                                  const uint rows,
                                  const uint cols) = 0;

    virtual T get_data(int idx) const = 0;
    virtual bool set_data(const T& value, int idx) = 0;

    virtual T get_data(int row, int col) const = 0;
    virtual bool set_data(const T& data,
                          int row,
                          int col) = 0;

    virtual BaseMatrixT<T>* get_row_data(const int row) = 0;
    virtual bool set_row_data(BaseMatrixT<T>* mat, const int row) = 0;

    virtual bool extend(const BaseMatrixT<T>* mat) = 0;

    bool add(const BaseMatrixT<T>* mat);
    bool subtract(const BaseMatrixT<T>* mat);

    virtual bool product(const BaseMatrixT<T>* mat) = 0;
    virtual bool pow(T exponent) = 0;

    bool add(const T value);
    bool subtract(const T value);
    bool multiply(const T value);
    bool multiply(const BaseMatrixT<T>* mat);
    bool division(const T value);

    bool sigmoid();

    virtual T sum() const = 0;

    virtual bool swap(const uint a_row,
                      const uint a_col,
                      const uint b_row,
                      const uint b_col) = 0;
    virtual bool swap_row(const uint a, const uint b) = 0;
    virtual bool swap_col(const uint a, const uint b) = 0;

    virtual void transpose() = 0;

    virtual void add_x0() = 0;
    virtual void add_x0(BaseMatrixT<T>* result) = 0;

    virtual bool det(T* result) = 0;

    virtual real mean() = 0;
    virtual real mean(uint col) = 0;

    virtual real var() = 0;
    virtual real var(uint col) = 0;

    virtual bool inverse(BaseMatrixT<real>* result) = 0;

    virtual void display(const std::string& split="\t") = 0;
    virtual std::string* to_string() = 0;

    virtual bool operator==(BaseMatrixT<T>* mat) const = 0;

protected:
    uint _rows;
    uint _cols;

};//class BaseMatrixT


template<class T>
class DenseMatrixT : public BaseMatrixT<T>{
public:
    DenseMatrixT();
    DenseMatrixT(const uint rows,
                 const uint cols);
    DenseMatrixT(const T* data,
                 const uint rows,
                 const uint cols);
    ~DenseMatrixT();

    DenseMatrixT<T>* clone();
    void clear_matrix();

    T* get_data() const;
    void set_data(const T* data,
                  const uint rows,
                  const uint cols);

    T get_data(int idx) const;
    bool set_data(const T& value, int idx);

    T get_data(int row, int col) const;
    bool set_data(const T& data,
                  int row,
                  int col);
    void set_shallow_data(T* data,
                          const uint rows,
                          const uint cols);


    DenseMatrixT<T>* get_row_data(const int row);
    bool set_row_data(BaseMatrixT<T>* mat, int row);

    bool extend(const BaseMatrixT<T>* mat);

    bool pow(T exponent);

    bool product(const BaseMatrixT<T>* mat);

    T sum() const;

    bool swap(const uint a_row,
              const uint a_col,
              const uint b_row,
              const uint b_col);
    bool swap_row(const uint a, const uint b);
    bool swap_col(const uint a, const uint b);

    void transpose();

    void add_x0();
    void add_x0(BaseMatrixT<T>* result);

    bool det(T* result);

    real mean();
    real mean(uint col);

    real var();
    real var(uint col);

    bool inverse(BaseMatrixT<real>* result);

    bool operator==(BaseMatrixT<T>* mat) const;

    void display(const std::string& split="\t");

    std::string* to_string();

protected:
    T* _data;

    inline bool check_range(int idx){
        if(idx < 0){
            idx += this->_rows * this->_cols;
        }
        return idx < this->_rows * this->_cols;
    }
    inline bool check_range(int row, int col){
        if(row < 0){
            row += this->_rows;
        }
        if(col < 0){
            col += this->_cols;
        }

        return row < this->_rows && col < this->_cols;
    }

    inline void clear_cache(){
        this->_cache_matrix_det = ccma::utils::get_max_value<T>();

        if(_cache_to_string != nullptr){
            delete _cache_to_string;
            _cache_to_string = nullptr;
        }
    }

private:
    T _cache_matrix_det;
    std::string* _cache_to_string = nullptr;
};//class DenseMatrixT


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
};//class DenseMatrixMNT

template<class T>
class DenseColMatrixT : public DenseMatrixMNT<T>{
public:
    DenseColMatrixT(uint rows, T value):DenseMatrixMNT<T>(rows, 1, value){}
    DenseColMatrixT(const T* data, uint rows):DenseMatrixMNT<T>(data, rows, 1){}
};//class DenseColMatrixT

template<class T>
class DenseRowMatrixT : public DenseMatrixMNT<T>{
public:
    DenseRowMatrixT(uint cols, T value):DenseMatrixMNT<T>(1, cols, value){}
    DenseRowMatrixT(const T* data, uint cols):DenseMatrixMNT<T>(data, 1, cols){}
};//class DenseRowMatrixT

template<class T>
class DenseOneMatrixT : public DenseMatrixMNT<T>{
public:
    explicit DenseOneMatrixT(uint rows):DenseMatrixMNT<T>(rows, 1, 1){}
};//class DenseOneMatrixT

template<class T>
class DenseZeroMatrixT : public DenseMatrixMNT<T>{
public:
    explicit DenseZeroMatrixT(uint rows):DenseMatrixMNT<T>(rows, 1, 0){}
};//class DenseZeroMatrixT

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
class CCMap : public std::unordered_map<T, uint>{
public:
    CCMap() : std::unordered_map<T, uint>(), _value(0){}

    uint get_max_value(){
        if(_value == 0){
            find_max_iterator();
        }
        return _value;
    }

    T get_max_key(){
        if(_value == 0){
            find_max_iterator();
        }
        return _key;
    }

private:
    uint _value;
    T _key;

    void find_max_iterator(){
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
class LabeledDenseMatrixT : public DenseMatrixT<T>{
public:
    LabeledDenseMatrixT();
    LabeledDenseMatrixT(const T* data,
                        const T* labels,
                        const uint rows,
                        const uint cols);

    LabeledDenseMatrixT(const T* data,
                        const T* labels,
                        const uint* feature_names,
                        const uint rows,
                        const uint cols);
    ~LabeledDenseMatrixT();

    inline T get_label(uint idx) const{
        return this->_labels[idx];
    }

    inline uint get_feature_name(uint idx) const{
        return this->_feature_names[idx];
    }

    DenseMatrixT<T>* get_data_matrix();

    DenseColMatrixT<T>* get_labels();

    void set_data(const T* data,
                  const T* labels,
                  const uint rows,
                  const uint cols);

    void set_data(const T* data,
                  const T* labels,
                  const uint* feature_names,
                  const uint rows,
                  const uint cols);

    LabeledDenseMatrixT<T>* get_row_data(const int row_id);
    void set_row_data(LabeledDenseMatrixT<T>* mat, const int row_id);

    void set_shallow_data(T* data,
                          T* labels,
                          const uint rows,
                          const uint cols);

    void set_shallow_data(T* data,
                          T* labels,
                          uint* feature_names,
                          const uint rows,
                          const uint cols);

    void clear_matrix();

    real get_shannon_entropy();

    //only equal
    bool split(uint feature_idx,
               T split_value,
               LabeledDenseMatrixT<T>* sub_mat);
    //binary_split
    //lt_mat --> less than and equal
    //gt_mat --> greater than
    bool binary_split(uint feature_idx,
                      T split_value,
                      LabeledDenseMatrixT<T>* lt_mat,
                      LabeledDenseMatrixT<T>* gt_mat);

    CCMap<T>* get_label_cnt_map();

    CCMap<T>* get_feature_cnt_map(uint feature_idx);

    real label_mean();
    real label_var();

    bool is_unique_label();

    bool operator==(LabeledDenseMatrixT<T>* mat) const;

    void display(const std::string& split="\t");

protected:
    T* _labels = nullptr;
    uint* _feature_names = nullptr;

    void clear_cache();

private:
    real _cache_shannon_entropy = 0.0;
    CCMap<T>* _cache_label_cnt_map;
    std::unordered_map<uint, CCMap<T>* >* _cache_feature_cnt_map;
};//class LabeledDenseMatrixT


}//namespace algebra
}//namespace ccma

#endif //_CCMA_ALGEBRA_BASEMATRIX_H_
