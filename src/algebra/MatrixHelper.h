/*********************************************
* Author: Jun Jiang - jiangjun4@sina.com
* Created: 2017-01-16 14:34
* Last modified: 2017-01-16 14:34
* Filename: MatrixHelper.h
* Description:Matrix algorithm Helper
**********************************************/

#include "BaseMatrix.h"
#include <cmath>


namespace ccma{
namespace algebra{

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
    bool pow(const ccma::algebra::BaseMatrixT<T1>* mat,
             const T2 exponent,
             ccma::algebra::BaseMatrixT<T3>* result);

    template<class T1, class T2>
    bool log(const ccma::algebra::BaseMatrixT<T1>* mat,
             ccma::algebra::BaseMatrixT<T2>* result);

    template<class T1, class T2>
    bool exp(const ccma::algebra::BaseMatrixT<T1>* mat,
             ccma::algebra::BaseMatrixT<T2>* result);
};//class MatrixHelper

template<class T1, class T2, class T3>
bool MatrixHelper::add(const ccma::algebra::BaseMatrixT<T1>* mat1,
                       const ccma::algebra::BaseMatrixT<T2>* mat2,
                       ccma::algebra::BaseMatrixT<T3>* result){
    if(mat1->get_rows() != mat2->get_rows() || mat1->get_cols() != mat2->get_cols()){
        return false;
    }

    T3* data = new T3[mat1->get_rows() * mat2->get_cols()];
    for(uint i = 0; i < mat1->get_rows() * mat1->get_cols(); i++){
        data[i] = static_cast<T3>(mat1->get_data(i)) + static_cast<T3>(mat2->get_data(i));
    }

    if(result == nullptr){
        result = new ccma::algebra::DenseMatrixT<T3>();
    }
    result->set_shallow_data(data, mat1->get_rows(), mat2->get_cols());

    return true;
}

template<class T1, class T2, class T3>
bool MatrixHelper::subtract(const ccma::algebra::BaseMatrixT<T1>* mat1,
                            const ccma::algebra::BaseMatrixT<T2>* mat2,
                            ccma::algebra::BaseMatrixT<T3>* result){
    if(mat1->get_rows() != mat2->get_rows() || mat1->get_cols() != mat2->get_cols()){
        return false;
    }

    T3* data = new T3[mat1->get_rows() * mat2->get_cols()];
    for(uint i = 0; i < mat1->get_rows() * mat1->get_cols(); i++){
        data[i] = static_cast<T3>(mat1->get_data(i)) - static_cast<T3>(mat2->get_data(i));
    }

    if(result == nullptr){
        result = new ccma::algebra::DenseMatrixT<T3>();
    }
    result->set_shallow_data(data, mat1->get_rows(), mat2->get_cols());

    return true;
}

template<class T1, class T2, class T3>
bool MatrixHelper::pow(const ccma::algebra::BaseMatrixT<T1>* mat,
                       const T2 exponent,
                       ccma::algebra::BaseMatrixT<T3>* result){
    T3* data = new T3[mat->get_rows() * mat->get_cols()];
    for(uint i = 0; i < mat->get_rows() * mat->get_cols(); i++){
        data[i] = static_cast<T3>(pow(mat->get_data(i), exponent));
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
        data[i] = static_cast<T2>(log(mat->get_data(i)));
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
        data[i] = static_cast<T2>(exp(mat->get_data(i)));
    }

    if(result == nullptr){
        result = new ccma::algebra::DenseMatrixT<T2>();
    }

    result->set_shallow_data(data, mat->get_rows(), mat->get_cols());

    return true;
}


}//namespace algebra
}//namespace ccma

