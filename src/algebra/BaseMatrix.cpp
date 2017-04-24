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
        return false;
    }

    for(int i = 0; i < this->_rows * this->_cols; i++){
        set_data(this->get_data(i) + mat->get_data(i), i);
    }

    return true;
}

template<class T>
bool BaseMatrixT<T>::subtract(BaseMatrixT<T>* mat){
    if( this->_rows != mat->get_rows() || this->_cols != mat->get_cols()){
        return false;
    }

    for(int i = 0; i < this->_rows * this->_cols; i++){
        set_data(get_data(i) - mat->get_data(i), i);
    }

    return true;
}

template<class T>
bool BaseMatrixT<T>::add(const T value){
    for(int i = 0; i < _rows * _cols; i++){
        set_data(get_data(i) + value, i);
    }

    return true;
}
template<class T>
bool BaseMatrixT<T>::subtract(const T value){
    for(int i = 0; i < _rows * _cols; i++){
        set_data(get_data(i) - value, i);
    }

    return true;
}

template<class T>
bool BaseMatrixT<T>::multiply(const T value){
    for(int i = 0; i < _rows * _cols; i++){
        set_data(get_data(i) * value, i);
    }

    return true;
}

template<class T>
bool BaseMatrixT<T>::multiply(BaseMatrixT<T>* mat){
    if(_rows != mat->get_rows() || _cols != mat->get_cols()){
        return false;
    }
    for(int i = 0; i < _rows * _cols; i++){
        set_data(get_data(i) * mat->get_data(i), i);
    }

    return true;
}

template<class T>
bool BaseMatrixT<T>::division(const T value){
    for(int i = 0; i < _rows * _cols; i++){
        set_data(get_data(i) / value, i);
    }

    return true;
}

template<class T>
bool BaseMatrixT<T>::sigmoid(){
    for(int i = 0; i < _rows * _cols; i++){
        set_data(1/(1 + exp(-get_data(i))), i);
    }

    return true;
}


template class BaseMatrixT<int>;
template class BaseMatrixT<real>;

}//namespace ccma
}//namespace algebra
