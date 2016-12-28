/*********************************************
* Author: Jun Jiang - jiangjun4@sina.com
* Created: 2016-12-07 15:14
* Last modified: 2016-12-07 15:14
* Filename: LinearRegress.h
* Description:linear regression
**********************************************/
#include "LinearRegress.h"
#include <stdio.h>

namespace ccma{
namespace algorithm{
namespace regression{


template<class T>
bool LinearRegression::standard_regression(ccma::algebra::LabeledDenseMatrixT<T>* train_data, ccma::algebra::DenseColMatrixT<real>* weights){
    ccma::algebra::BaseMatrixT<T>* x = train_data->copy_data();

    ccma::algebra::BaseMatrixT<T>* y = train_data->get_labels();

    ccma::algebra::BaseMatrixT<T>* xT = new ccma::algebra::DenseMatrixT<T>();
    x->transpose(xT);

    ccma::algebra::BaseMatrixT<T>* xTx = new ccma::algebra::DenseMatrixT<T>();
    if(!xT->product(x, xTx)){
        delete x, y, xT, xTx;
        return false;
    }

    ccma::algebra::BaseMatrixT<real>* xTxI = new ccma::algebra::DenseMatrixT<real>();
    if(!xTx->inverse(xTxI)){
        delete x, y, xT, xTx, xTxI;
        return false;
    }

    ccma::algebra::BaseMatrixT<T>* xTy = new ccma::algebra::DenseMatrixT<T>();
    xT->product(y, xTy);
    xTxI->product(xTy, weights);

    delete x, y, xT, xTx, xTy, xTxI;

    return true;
}

template bool LinearRegression::standard_regression(ccma::algebra::LabeledDenseMatrixT<int>* train_data, ccma::algebra::DenseColMatrixT<real>* weights);
template bool LinearRegression::standard_regression(ccma::algebra::LabeledDenseMatrixT<real>* train_data, ccma::algebra::DenseColMatrixT<real>* weights);

}//regression
}//namespace
}//namespace ccma
