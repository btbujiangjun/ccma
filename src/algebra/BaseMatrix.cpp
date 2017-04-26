/*********************************************
* Author: Jun Jiang - jiangjun4@sina.com
* Created: 2017-04-13 15:47
* Last modified: 2017-04-13 15:47
* Filename: BaseMatrix.cpp
* Description: Implemention of BaseMatrixT
**********************************************/

#include "BaseMatrix.h"
#include <cmath>

namespace ccma{
namespace algebra{

template<class T>
bool BaseMatrixT<T>::add(BaseMatrixT<T>* mat){
    if(this->_rows != mat->get_rows() || this->_cols != mat->get_cols()){
        printf("Add matrix dim Error:[%d-%d][%d-%d]\n", this->_rows, this->_cols, mat->get_rows(), mat->get_cols());
        return false;
    }

    uint size = this->get_size();
    for(int i = 0; i < size; i++){
        T value = mat->get_data(i);
        if(value != 0){
            set_data(this->get_data(i) + mat->get_data(i), i);
        }
    }

    return true;
}

template<class T>
bool BaseMatrixT<T>::subtract(BaseMatrixT<T>* mat){
    if( this->_rows != mat->get_rows() || this->_cols != mat->get_cols()){
        printf("Subtract matrix dim Error:[%d-%d][%d-%d]\n", this->_rows, this->_cols, mat->get_rows(), mat->get_cols());
        return false;
    }

    uint size = this->get_size();
    for(int i = 0; i < size; i++){
        T value = mat->get_data(i);
        if(value != 0){
            set_data(get_data(i) - mat->get_data(i), i);
        }
    }

    return true;
}

template<class T>
bool BaseMatrixT<T>::add(const T value){
    uint size = this->get_size();
    for(int i = 0; i < size; i++){
        set_data(get_data(i) + value, i);
    }

    return true;
}
template<class T>
bool BaseMatrixT<T>::subtract(const T value){
    uint size = this->get_size();
    for(int i = 0; i < size; i++){
        set_data(get_data(i) - value, i);
    }

    return true;
}

template<class T>
bool BaseMatrixT<T>::multiply(const T value){
    uint size = this->get_size();
    for(int i = 0; i < size; i++){
        set_data(get_data(i) * value, i);
    }

    return true;
}

template<class T>
bool BaseMatrixT<T>::multiply(BaseMatrixT<T>* mat){
    if(_rows != mat->get_rows() || _cols != mat->get_cols()){
        printf("multiply matrix dim Error:[%d-%d][%d-%d]\n", this->_rows, this->_cols, mat->get_rows(), mat->get_cols());
        return false;
    }

    uint size = this->get_size();
    for(int i = 0; i < _rows * _cols; i++){
        T value = mat->get_data(i);
        if(value != 1){
           set_data(get_data(i) * mat->get_data(i), i);
        }
    }

    return true;
}

template<class T>
bool BaseMatrixT<T>::division(const T value){
    uint size = this->get_size();
    for(int i = 0; i < size; i++){
        set_data(get_data(i) / value, i);
    }

    return true;
}

template<class T>
bool BaseMatrixT<T>::sigmoid(){
    uint size = this->get_size();
    for(int i = 0; i < _rows * _cols; i++){
        set_data(1/(1 + exp(-get_data(i))), i);
    }

    return true;
}


template class BaseMatrixT<int>;
template class BaseMatrixT<real>;

}//namespace ccma
}//namespace algebra
