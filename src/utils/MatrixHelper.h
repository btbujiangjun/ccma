/*********************************************
* Author: Jun Jiang - jiangjun4@sina.com
* Created: 2017-01-16 14:34
* Last modified: 2017-01-16 14:34
* Filename: MatrixHelper.h
* Description:Matrix algorithm Helper
**********************************************/
#ifndef _CCMA_UTILS_MATRIXHELPER_H_
#define _CCMA_UTILS_MATRIXHELPER_H_

#include "algebra/BaseMatrix.h"
#include <cmath>


namespace ccma{
namespace utils{

class MatrixHelper{
public:
    template<class T1, class T2, class T3>
    bool add(const ccma::algebra::BaseMatrixT<T1>* mat1,
             const ccma::algebra::BaseMatrixT<T2>* mat2,
             ccma::algebra::BaseMatrixT<T3>* result);

    template<class T1, class T2, class T3>
    bool subtract(const ccma::algebra::BaseMatrixT<T1>* mat1,
                  const ccma::algebra::BaseMatrixT<T2>* mat2,
                  ccma::algebra::BaseMatrixT<T3>* result);

    template<class T1, class T2, class T3>
    bool product(const ccma::algebra::BaseMatrixT<T1>* mat1,
                 const ccma::algebra::BaseMatrixT<T2>* mat2,
                 ccma::algebra::BaseMatrixT<T3>* result);

    template<class T>
    bool product(const ccma::algebra::BaseMatrixT<T>* mat,
                 const T value,
                 ccma::algebra::BaseMatrixT<T>* result);

    template<class T1, class T2, class T3>
    bool multiply(const ccma::algebra::BaseMatrixT<T1>* mat1,
                  const ccma::algebra::BaseMatrixT<T2>* mat2,
                  ccma::algebra::BaseMatrixT<T3>* result);

    template<class T1, class T2, class T3>
    bool pow(const ccma::algebra::BaseMatrixT<T1>* mat,
             const T2 exponent,
             ccma::algebra::BaseMatrixT<T3>* result);

    template<class T1, class T2>
    bool log(const ccma::algebra::BaseMatrixT<T1>* mat,
             ccma::algebra::BaseMatrixT<T2>* result);

    template<class T1, class T2>
    bool exp(const ccma::algebra::BaseMatrixT<T1>* mat,
             ccma::algebra::BaseMatrixT<T2>* result);

    template<class T>
    void transpose(const ccma::algebra::BaseMatrixT<T>* mat, ccma::algebra::BaseMatrixT<T>* result);

    template<class T>
    bool signmod(const ccma::algebra::BaseMatrixT<T>* mat, ccma::algebra::BaseMatrixT<real>* result);
};//class MatrixHelper

template<class T1, class T2, class T3>
bool MatrixHelper::add(const ccma::algebra::BaseMatrixT<T1>* mat1,
                       const ccma::algebra::BaseMatrixT<T2>* mat2,
                       ccma::algebra::BaseMatrixT<T3>* result){
    if(mat1->get_rows() != mat2->get_rows() || mat1->get_cols() != mat2->get_cols()){
        return false;
    }

    T3* data = new T3[mat1->get_rows() * mat1->get_cols()];
    for(uint i = 0; i < mat1->get_rows() * mat1->get_cols(); i++){
        data[i] = static_cast<T3>(mat1->get_data(i)) + static_cast<T3>(mat2->get_data(i));
    }

    if(result == nullptr){
        result = new ccma::algebra::DenseMatrixT<T3>();
    }
    result->set_shallow_data(data, mat1->get_rows(), mat1->get_cols());

    return true;
}

template<class T1, class T2, class T3>
bool MatrixHelper::subtract(const ccma::algebra::BaseMatrixT<T1>* mat1,
                            const ccma::algebra::BaseMatrixT<T2>* mat2,
                            ccma::algebra::BaseMatrixT<T3>* result){
    if(mat1->get_rows() != mat2->get_rows() || mat1->get_cols() != mat2->get_cols()){
        return false;
    }

    T3* data = new T3[mat1->get_rows() * mat1->get_cols()];
    for(uint i = 0; i < mat1->get_rows() * mat1->get_cols(); i++){
        data[i] = static_cast<T3>(mat1->get_data(i)) - static_cast<T3>(mat2->get_data(i));
    }

    if(result == nullptr){
        result = new ccma::algebra::DenseMatrixT<T3>();
    }
    result->set_shallow_data(data, mat1->get_rows(), mat1->get_cols());

    return true;
}

template<class T1, class T2, class T3>
bool MatrixHelper::product(const ccma::algebra::BaseMatrixT<T1>* mat1,
                           const ccma::algebra::BaseMatrixT<T2>* mat2,
                           ccma::algebra::BaseMatrixT<T3>* result){
    if(mat1->get_cols() != mat2->get_rows()){
        printf("Product Matrix Dim ERROR:[%d-%d][%d-%d]\n", mat1->get_rows(), mat1->get_cols(), mat2->get_rows(), mat2->get_cols());
        return false;
    }

    T3* data = new T3[mat1->get_rows() * mat2->get_cols()];
    for(int i = 0; i < mat1->get_rows(); i++){
        for(int j = 0; j < mat2->get_cols(); j++){
            T3 value = static_cast<T3>(0);
            for(int k = 0; k < mat1->get_cols(); k++){
                value += static_cast<T3>(mat1->get_data(i, k) * mat2->get_data(k, j));
            }
            data[i * mat2->get_cols() + j] = value;
        }
    }

    if(result == nullptr){
        result = new ccma::algebra::DenseMatrixT<T3>();
    }
    result->set_shallow_data(data, mat1->get_rows(), mat2->get_cols());

    return true;
}

template<class T>
bool MatrixHelper::product(const ccma::algebra::BaseMatrixT<T>* mat,
                           const T value,
                           ccma::algebra::BaseMatrixT<T>* result){
    T* data = new T[mat->get_rows() * mat->get_cols()];
    for(uint i = 0; i < mat->get_rows() * mat->get_cols(); i++){
        data[i] = mat->get_data(i) * value;
    }

    if(result == nullptr){
        result = new ccma::algebra::DenseMatrixT<T>();
    }

    result->set_shallow_data(data, mat->get_rows(), mat->get_cols());

    return true;
}

template<class T1, class T2, class T3>
bool MatrixHelper::multiply(const ccma::algebra::BaseMatrixT<T1>* mat1,
                            const ccma::algebra::BaseMatrixT<T2>* mat2,
                            ccma::algebra::BaseMatrixT<T3>* result){
    if(mat1->get_rows() != mat2->get_rows() || mat1->get_cols() != mat2->get_cols()){
        return false;
    }

    int size = mat1->get_rows() * mat1->get_cols();
    T3* data = new T3[size];

    for(uint i = 0; i < size; i++){
        data[i] = static_cast<T3>(mat1->get_data(i) * mat2->get_data(i));
    }

    if(!result){
        result = new ccma::algebra::DenseMatrixT<T3>();
    }

    result->set_shallow_data(data, mat1->get_rows(), mat1->get_cols());

    return true;
}

template<class T1, class T2, class T3>
bool MatrixHelper::pow(const ccma::algebra::BaseMatrixT<T1>* mat,
                       const T2 exponent,
                       ccma::algebra::BaseMatrixT<T3>* result){
    T3* data = new T3[mat->get_rows() * mat->get_cols()];
    for(uint i = 0; i < mat->get_rows() * mat->get_cols(); i++){
        data[i] = static_cast<T3>(std::pow(mat->get_data(i), exponent));
    }

    if(result == nullptr){
        result = new ccma::algebra::DenseMatrixT<T3>();
    }

    result->set_shallow_data(data, mat->get_rows(), mat->get_cols());

    return true;
}

template<class T1, class T2>
bool MatrixHelper::log(const ccma::algebra::BaseMatrixT<T1>* mat,
                       ccma::algebra::BaseMatrixT<T2>* result){
    T2* data = new T2[mat->get_rows() * mat->get_cols()];
    for(uint i = 0; i < mat->get_rows() * mat->get_cols(); i++){
        data[i] = static_cast<T2>(std::log(mat->get_data(i)));
    }

    if(result == nullptr){
        result = new ccma::algebra::DenseMatrixT<T2>();
    }

    result->set_shallow_data(data, mat->get_rows(), mat->get_cols());

    return true;
}

template<class T1, class T2>
bool MatrixHelper::exp(const ccma::algebra::BaseMatrixT<T1>* mat,
                       ccma::algebra::BaseMatrixT<T2>* result){
    T2* data = new T2[mat->get_rows() * mat->get_cols()];
    for(uint i = 0; i < mat->get_rows() * mat->get_cols(); i++){
        data[i] = static_cast<T2>(std::exp(mat->get_data(i)));
    }

    if(result == nullptr){
        result = new ccma::algebra::DenseMatrixT<T2>();
    }

    result->set_shallow_data(data, mat->get_rows(), mat->get_cols());

    return true;
}

template<class T>
bool MatrixHelper::signmod(const ccma::algebra::BaseMatrixT<T>* mat, ccma::algebra::BaseMatrixT<real>* result){
    real* data = new real[mat->get_rows() * mat->get_cols()];
    for(uint i = 0; i < mat->get_rows() * mat->get_cols(); i++){
        data[i] = 1.0f/(1.0f + std::exp(-mat->get_data(i)));
    }

    if(result == nullptr){
        result = new ccma::algebra::DenseMatrixT<real>();
    }

    result->set_shallow_data(data, mat->get_rows(), mat->get_cols());

    return true;
}

template<class T>
void MatrixHelper::transpose(const ccma::algebra::BaseMatrixT<T>* mat, ccma::algebra::BaseMatrixT<T>* result){

    T* data = new T[mat->get_rows() * mat->get_cols()];

    for(int i = 0; i < mat->get_cols(); i++){
        for(int j = 0; j < mat->get_rows(); j++){
            data[i * mat->get_rows() + j] = mat->get_data(j, i);
        }
    }

    if(result == nullptr){
        result = new ccma::algebra::DenseMatrixT<T>();
    }
    result->set_shallow_data(data, mat->get_cols(), mat->get_rows());
}

}//namespace utils
}//namespace ccma

#endif
