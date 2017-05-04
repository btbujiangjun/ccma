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
bool BaseMatrixT<T>::add(const T value){
    uint size = get_size();
    T* data = get_data();
    for(uint i = 0; i < size; i++){
        data[i] += value;
    }
    return true;
}
template<class T>
bool BaseMatrixT<T>::add(BaseMatrixT<T>* mat){
    if(_rows != mat->get_rows() || _cols != mat->get_cols()){
        printf("Add matrix dim Error:[%d-%d][%d-%d]\n", _rows, _cols, mat->get_rows(), mat->get_cols());
        return false;
    }

    uint size = get_size();
    T* data_a = get_data();
    T* data_b = mat->get_data();

    for(uint i = 0; i < size; i++){
        data_a[i] += data_b[i];
    }

    return true;
}

template<class T>
bool BaseMatrixT<T>::subtract(const T value){
    uint size = get_size();
    T* data = get_data();
    for(uint i = 0; i < size; i++){
        data[i] -= value;
    }

    return true;
}
template<class T>
bool BaseMatrixT<T>::subtract(BaseMatrixT<T>* mat){
    if( _rows != mat->get_rows() || _cols != mat->get_cols()){
        printf("Subtract matrix dim Error:[%d-%d][%d-%d]\n", _rows, _cols, mat->get_rows(), mat->get_cols());
        return false;
    }

    uint size = get_size();

    T* data_a = get_data();
    T* data_b = mat->get_data();

    for(uint i = 0; i < size; i++){
        data_a[i] -= data_b[i];
    }

    return true;
}

template<class T>
bool BaseMatrixT<T>::multiply(const T value){
    uint size = get_size();
    T* data = get_data();

    for(uint i = 0; i < size; i++){
        data[i] *= value;
    }

    return true;
}
template<class T>
bool BaseMatrixT<T>::multiply(BaseMatrixT<T>* mat){
    if(_rows != mat->get_rows() || _cols != mat->get_cols()){
        printf("multiply matrix dim Error:[%d-%d][%d-%d]\n", _rows, _cols, mat->get_rows(), mat->get_cols());
        return false;
    }

    uint size = get_size();
    T* data_a = get_data();
    T* data_b = mat->get_data();

    for(uint i = 0; i < size; i++){
        data_a[i] *= data_b[i];
    }

    return true;
}

template<class T>
bool BaseMatrixT<T>::division(const T value){

    uint size   = get_size();
    T* data     = get_data();

    for(uint i = 0; i < size; i++){
        data[i] /=  value;
    }

    return true;
}

template<class T>
bool BaseMatrixT<T>::sigmoid(){

    uint size   = get_size();
    T one       = static_cast<T>(1);
    T* data     = get_data();

    for(uint i = 0; i < size; i++){
        data[i] = one / (one + std::exp(-data[i]));
    }

    return true;
}


template class BaseMatrixT<int>;
template class BaseMatrixT<real>;

}//namespace ccma
}//namespace algebra
