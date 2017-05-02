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
    bool add(ccma::algebra::BaseMatrixT<T1>* mat1,
             ccma::algebra::BaseMatrixT<T2>* mat2,
             ccma::algebra::BaseMatrixT<T3>* result);

    template<class T1, class T2, class T3>
    bool subtract(ccma::algebra::BaseMatrixT<T1>* mat1,
                  ccma::algebra::BaseMatrixT<T2>* mat2,
                  ccma::algebra::BaseMatrixT<T3>* result);

    template<class T1, class T2, class T3>
    bool dot(ccma::algebra::BaseMatrixT<T1>* mat1,
                 ccma::algebra::BaseMatrixT<T2>* mat2,
                 ccma::algebra::BaseMatrixT<T3>* result);

    template<class T>
    bool dot(ccma::algebra::BaseMatrixT<T>* mat,
                 const T value,
                 ccma::algebra::BaseMatrixT<T>* result);

    template<class T1, class T2, class T3>
    bool multiply(ccma::algebra::BaseMatrixT<T1>* mat1,
                  ccma::algebra::BaseMatrixT<T2>* mat2,
                  ccma::algebra::BaseMatrixT<T3>* result);

    template<class T1, class T2, class T3>
    bool pow(ccma::algebra::BaseMatrixT<T1>* mat,
             const T2 exponent,
             ccma::algebra::BaseMatrixT<T3>* result);

    template<class T1, class T2>
    bool log(ccma::algebra::BaseMatrixT<T1>* mat,
             ccma::algebra::BaseMatrixT<T2>* result);

    template<class T1, class T2>
    bool exp(ccma::algebra::BaseMatrixT<T1>* mat,
             ccma::algebra::BaseMatrixT<T2>* result);

    template<class T>
    void transpose(ccma::algebra::BaseMatrixT<T>* mat, ccma::algebra::BaseMatrixT<T>* result);

    template<class T>
    bool signmod(ccma::algebra::BaseMatrixT<T>* mat, ccma::algebra::BaseMatrixT<real>* result);
};//class MatrixHelper

template<class T1, class T2, class T3>
bool MatrixHelper::add(ccma::algebra::BaseMatrixT<T1>* mat1,
                       ccma::algebra::BaseMatrixT<T2>* mat2,
                       ccma::algebra::BaseMatrixT<T3>* result){

    uint row1 = mat1->get_rows();
    uint col1 = mat1->get_cols();
    uint row2 = mat2->get_rows();
    uint col2 = mat2->get_cols();

    if(row1 != row2 || col1 != col2){
        printf("MatrixHelper::add, Matrix Dim Error:[%d-%d][%d-%d]\n", row1, col1, row2, col2);
        return false;
    }

    uint size = row1 * col1;
    T3* data = new T3[size];
    for(uint i = 0; i < size; i++){
        data[i] = static_cast<T3>(mat1->get_data(i)) + static_cast<T3>(mat2->get_data(i));
    }

    result->set_shallow_data(data, row1, col1);

    return true;
}

template<class T1, class T2, class T3>
bool MatrixHelper::subtract(ccma::algebra::BaseMatrixT<T1>* mat1,
                            ccma::algebra::BaseMatrixT<T2>* mat2,
                            ccma::algebra::BaseMatrixT<T3>* result){

    uint row1 = mat1->get_rows();
    uint col1 = mat1->get_cols();
    uint row2 = mat2->get_rows();
    uint col2 = mat2->get_cols();

    if(row1 != row2 || col1 != col2){
        printf("MatrixHelper::subtract, Matrix Dim Error:[%d-%d][%d-%d]\n", row1, col1, row2, col2);
        return false;
    }

    uint size = row1 * col1;
    T3* data = new T3[size];
    for(uint i = 0; i < size; i++){
        data[i] = static_cast<T3>(mat1->get_data(i)) - static_cast<T3>(mat2->get_data(i));
    }

    result->set_shallow_data(data, row1, col1);

    return true;
}

template<class T1, class T2, class T3>
bool MatrixHelper::dot(ccma::algebra::BaseMatrixT<T1>* mat1,
                           ccma::algebra::BaseMatrixT<T2>* mat2,
                           ccma::algebra::BaseMatrixT<T3>* result){

    uint row1 = mat1->get_rows();
    uint col1 = mat1->get_cols();
    uint row2 = mat2->get_rows();
    uint col2 = mat2->get_cols();

    if(col1 != row2){
        printf("MatrixHelper::dot, Matrix Dim ERROR:[%d-%d][%d-%d]\n", row1, col1, row2, col2);
        return false;
    }

    T3* data = new T3[row1 * col2];
    for(int i = 0; i < row1; i++){
        for(int j = 0; j < col2; j++){
            T3 value = static_cast<T3>(0);
            for(int k = 0; k < col1; k++){
                value += static_cast<T3>(mat1->get_data(i, k) * mat2->get_data(k, j));
            }
            data[i * col2 + j] = value;
        }
    }

    result->set_shallow_data(data, row1, col2);

    return true;
}

template<class T>
bool MatrixHelper::dot(ccma::algebra::BaseMatrixT<T>* mat,
                           const T value,
                           ccma::algebra::BaseMatrixT<T>* result){
    int size = mat->get_rows() * mat->get_cols();
    T* data = new T[size];
    for(uint i = 0; i < size; i++){
        data[i] = mat->get_data(i) * value;
    }
    result->set_shallow_data(data, mat->get_rows(), mat->get_cols());

    return true;
}

template<class T1, class T2, class T3>
bool MatrixHelper::multiply(ccma::algebra::BaseMatrixT<T1>* mat1,
                            ccma::algebra::BaseMatrixT<T2>* mat2,
                            ccma::algebra::BaseMatrixT<T3>* result){

    uint row1 = mat1->get_rows();
    uint col1 = mat1->get_cols();
    uint row2 = mat2->get_rows();
    uint col2 = mat2->get_cols();

    if(row1 != row2 || col1 != col2){
        printf("MatrixHelper::multiply, Matrix Dim ERROR:[%d-%d][%d-%d]\n", row1, col1, row2, col2);
        return false;
    }

    int size = row1 * col1;
    T3* data = new T3[size];

    for(uint i = 0; i < size; i++){
        data[i] = static_cast<T3>(mat1->get_data(i) * mat2->get_data(i));
    }

    result->set_shallow_data(data, row1, col1);

    return true;
}

template<class T1, class T2, class T3>
bool MatrixHelper::pow(ccma::algebra::BaseMatrixT<T1>* mat,
                       const T2 exponent,
                       ccma::algebra::BaseMatrixT<T3>* result){
    uint size = mat->get_rows() * mat->get_cols();
    T3* data = new T3[size];
    for(uint i = 0; i < size; i++){
        data[i] = static_cast<T3>(std::pow(mat->get_data(i), exponent));
    }

    result->set_shallow_data(data, mat->get_rows(), mat->get_cols());

    return true;
}

template<class T1, class T2>
bool MatrixHelper::log(ccma::algebra::BaseMatrixT<T1>* mat,
                       ccma::algebra::BaseMatrixT<T2>* result){
    uint size = mat->get_rows() * mat->get_cols();
    T2* data = new T2[size];
    for(uint i = 0; i < size; i++){
        data[i] = static_cast<T2>(std::log(mat->get_data(i)));
    }

    result->set_shallow_data(data, mat->get_rows(), mat->get_cols());

    return true;
}

template<class T1, class T2>
bool MatrixHelper::exp(ccma::algebra::BaseMatrixT<T1>* mat,
                       ccma::algebra::BaseMatrixT<T2>* result){
    uint size = mat->get_rows() * mat->get_cols();
    T2* data = new T2[size];
    for(uint i = 0; i < size; i++){
        data[i] = static_cast<T2>(std::exp(mat->get_data(i)));
    }

    result->set_shallow_data(data, mat->get_rows(), mat->get_cols());

    return true;
}

template<class T>
bool MatrixHelper::signmod(ccma::algebra::BaseMatrixT<T>* mat, ccma::algebra::BaseMatrixT<real>* result){
    uint size = mat->get_rows() * mat->get_cols();
    real* data = new real[size];
    for(uint i = 0; i < size; i++){
        data[i] = 1.0f/(1.0f + std::exp(-mat->get_data(i)));
    }

    result->set_shallow_data(data, mat->get_rows(), mat->get_cols());

    return true;
}

template<class T>
void MatrixHelper::transpose(ccma::algebra::BaseMatrixT<T>* mat, ccma::algebra::BaseMatrixT<T>* result){

    uint row    = mat->get_rows();
    uint col    = mat->get_cols();
    uint size   = row * col;
    T* data     = new T[size];

    for(int i = 0; i < col; i++){
        for(int j = 0; j < row; j++){
            data[i * row + j] = mat->get_data(j, i);
        }
    }
    result->set_shallow_data(data, col, row);
}

}//namespace utils
}//namespace ccma

#endif
