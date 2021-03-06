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
#include <thread>
#include <iostream>
#include <random>
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

    inline uint get_rows() const { return _rows;}
    inline uint get_cols() const { return _cols;}
    inline uint get_size() const { return _rows * _cols;}

    virtual void clone(BaseMatrixT<T>* out_mat) = 0;

    virtual T* get_data() = 0;
    virtual void set_data(const T* data,
                          const uint rows,
                          const uint cols) = 0;

    void set_data(BaseMatrixT<T>* mat){set_data(mat->get_data(), mat->get_rows(), mat->get_cols());}

    virtual void set_shallow_data(T* data,
                                  const uint rows,
                                  const uint cols) = 0;

    virtual T get_data(const int idx) = 0;
    virtual bool set_data(const T& value, const int idx) = 0;

    virtual T get_data(const int row, const int col) = 0;
    virtual bool set_data(const T& data,
                          const int row,
                          const int col) = 0;

    virtual bool get_row_data(const int row_id, BaseMatrixT<T>* out_mat) = 0;
    virtual bool set_row_data(const uint row_id, BaseMatrixT<T>* mat) = 0;
    virtual bool insert_row_data(const int row_id, BaseMatrixT<T>* mat) = 0;

    bool get_col_data(const uint col_id, BaseMatrixT<T>* out_mat);
    bool set_col_data(const uint col_id, BaseMatrixT<T>* mat);
    
    void reset(const T value = 0);
    void reset(const T value,
               uint rows,
               uint cols);

    virtual bool extend(BaseMatrixT<T>* mat, bool col_dim = true) = 0;

    bool add(BaseMatrixT<T>* mat);
    bool subtract(BaseMatrixT<T>* mat);

    bool dot(BaseMatrixT<T>* mat);
    void outer(BaseMatrixT<T>* mat);

    bool add(const T value);
    bool subtract(const T value);
    bool multiply(const T value);
    bool multiply(BaseMatrixT<T>* mat);
    bool division(const T value);

    void pow(const T exponent);
    void log();
    void exp();

    void sigmoid();
    void derivative_sigmoid();

	void softmax();
	void tanh();
	void relu();

    virtual T sum() const = 0;
    void x_sum();
    void y_sum();

    BaseMatrixT<int>* argmax(const uint axis);
    uint argmax(const uint id, const uint axis);

	bool isnan();
	bool isinf();

    virtual bool swap(const uint a_row,
                      const uint a_col,
                      const uint b_row,
                      const uint b_col) = 0;
    virtual bool swap_row(const uint a, const uint b) = 0;
    virtual bool swap_col(const uint a, const uint b) = 0;

    virtual BaseMatrixT<T>* transpose() = 0;

    bool reshape(uint row, uint col){
        if(row * col == _rows * _cols){
            _rows = row;
            _cols = col;
            return true;
        }
        return false;
    }

    void expand(uint row_dim, uint col_dim);

    /*
     * shape is full or valid, default full.
     */
    bool convn(ccma::algebra::BaseMatrixT<T>* kernal,
               uint stride = 1,
               std::string shape = "full");

    /*
     * convert dim
     * 1 is row dim, 2 is col dim
     */
    void flipdim(uint dim = 1);
    void flip180();

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

    inline uint get_num_thread(uint size){
        if(size <= _num_per_thread){
            return 1;
        }
        if(_num_hardware_concurrency == 0){
            return 2;
        }else{
            uint num =  size / _num_per_thread;
            if( size % _num_per_thread != 0){
                num += 1;
            }
            return std::min(num, _num_hardware_concurrency);
        }
    }

protected:
    uint _rows;
    uint _cols;

    uint _num_per_thread = 1000;
private:
    const uint _num_hardware_concurrency = std::thread::hardware_concurrency();
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

    void clone(BaseMatrixT<T>* out_mat);
    void clear_matrix();

    inline T* get_data(){
        return _data;
    }

    void set_data(const T* data,
                  const uint rows,
                  const uint cols);

    inline T get_data(const int idx){
        int index = idx;
        if(check_range(&index)){
            return _data[index];
        }
        //todo out_of_range exception
        return _data[idx];
    }
    inline bool set_data(const T& value, const int idx){
        int index = idx;
        if(check_range(&index)){
            _data[index] = value;
            clear_cache();
            return true;
        }
        return false;
    }

    inline T get_data(const int row, const int col){
        int r = row;
		int c = col;
        if(check_range(&r, &c)){
            return _data[r * this->_cols + c];
        }else{
            //todo out_of_range exception
            return _data[row * this->_cols + col];
        }
    }

    inline bool set_data(const T& value,
                         const int row,
                         const int col){
        int r = row, c = col;
        if(check_range(&r, &c)){
            _data[r * this->_cols + c] = value;
            clear_cache();

            return true;
        }
        return false;
    }

    void set_shallow_data(T* data,
                          const uint rows,
                          const uint cols);


    bool get_row_data(const int row_id, BaseMatrixT<T>* out_mat);
    bool set_row_data(const uint row_id, BaseMatrixT<T>* mat);
    bool insert_row_data(const int row_id, BaseMatrixT<T>* mat);

    bool extend(BaseMatrixT<T>* mat, bool col_dim = true);

    //bool dot(BaseMatrixT<T>* mat);

    T sum() const;

    bool swap(const uint a_row,
              const uint a_col,
              const uint b_row,
              const uint b_col);
    bool swap_row(const uint a, const uint b);
    bool swap_col(const uint a, const uint b);

    BaseMatrixT<T>* transpose();

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

    inline bool check_range(int* idx){
        int mat_size = this->_rows * this->_cols;
        if(*idx >= 0 && *idx < mat_size){
            return true;
        }else if(*idx < 0 && *idx + mat_size >= 0){
            *idx += mat_size;
            return true;
        }
        return false;
    }
    inline bool check_range(int* row, int* col){
        int r = *row, c = *col;
        if(r < 0){
            r += this->_rows;
        }
        if(c < 0){
            c += this->_cols;
        }

        if(r >= 0 && r < (int)this->_rows && c >= 0 && c < (int)this->_cols){
            if(*row < 0){
                *row = r;
            }
            if(*col < 0){
                *col = c;
            }
            return true;
        }
        return false;
    }

    inline void clear_cache(){
        _cache_matrix_det = ccma::utils::get_max_value<T>();

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
class DenseRandomMatrixT :public DenseMatrixT<T>{
public:
    DenseRandomMatrixT(uint rows,
                       uint cols,
                       const T mean_value,
                       const T stddev,
                       const T min_value = 0,
                       const T max_value = 0) : DenseMatrixT<T>(rows, cols){
        std::default_random_engine engine(std::chrono::system_clock::now().time_since_epoch().count());
        std::normal_distribution<T> distribution(mean_value, stddev);
        uint size = rows * cols;

        if(min_value != max_value){
            T scale = max_value - min_value;
            for(uint i = 0; i != size; i++){
                T value = static_cast<T>(distribution(engine));
                if(value > max_value){
                    value -= (int)((value - min_value)/scale) * scale; 
                }else if(value < min_value){
                    value =+ (int)((max_value - value))/scale * scale;
                }
                this->_data[i] = value;
            }
        }else{
            for(uint i = 0; i != size; i++){
                this->_data[i] = static_cast<T>(distribution(engine));
            }
        }
    }
};//class DenseRandomMatrixT

template<class T>
class DenseMatrixMNT : public DenseMatrixT<T>{
public:
    DenseMatrixMNT(const T* data, uint rows, uint cols):DenseMatrixT<T>(data, rows, cols){}

    DenseMatrixMNT(uint rows, uint cols, T value){
        this->_data = new T[rows * cols];

        //memset only support 0 or -1.
        if(value == 0 || value == -1){
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
                _value  = it->second;
                _key    = it->first;
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

    inline T get_label(const int idx) const{
        int index = idx;
        if(index < 0){
            index += this->_rows;
        }
        if(index >= 0 && index < (int)this->_rows){
            return this->_labels[index];
        }else{
            //todo out_of_range exception.
            return static_cast<T>(0);
        }
    }

    inline uint get_feature_name(const int idx) const{
        int index = idx;
        if(index < 0){
            index += this->_rows;
        }
        if(index >= 0 && index < (int)this->_rows){
            return this->_feature_names[index];
        }else{
            //todo out_of_range exception.
            return 0;
        }
    }

    void get_data_matrix(DenseMatrixT<T>* out_mat);

    void get_labels(DenseMatrixT<T>* out_mat);

    void set_data(const T* data,
                  const T* labels,
                  const uint rows,
                  const uint cols);

    void set_data(const T* data,
                  const T* labels,
                  const uint* feature_names,
                  const uint rows,
                  const uint cols);

    bool get_row_data(const int row_id, LabeledDenseMatrixT<T>* out_mat);
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

    bool operator==(LabeledDenseMatrixT<T>* mat);

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
