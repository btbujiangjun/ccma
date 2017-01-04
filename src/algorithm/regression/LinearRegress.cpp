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


template<class T>
bool LinearRegression::local_weight_logistic_regresion(ccma::algebra::LabeledDenseMatrixT<T>* train_data,
                                                     ccma::algebra::DenseMatrixT<T>* predict_data,
                                                     real k,
                                                     ccma::algebra::DenseColMatrixT<real>* predict_labels){
    if(train_data->get_cols() != predict_data->get_cols()){
        return false;
    }

    real* labels = new real[predict_data->get_rows()];

    ccma::algebra::DenseMatrixT<T>* x = train_data->get_data_matrix();
    ccma::algebra::DenseMatrixT<T>* y = train_data->get_labels();

    ccma::algebra::DenseMatrixT<T>* xT = new ccma::algebra::DenseMatrixT<T>();
    x->transpose(xT);

    ccma::algebra::DenseMatrixT<T>* xTx = new ccma::algebra::DenseMatrixT<T>();
    xT->product(x, xTx);

    for(uint i = 0 ; i < predict_data->get_rows(); i++){
        ccma::algebra::BaseMatrixT<real>* weight = new ccma::algebra::DenseEyeMatrixT<real>(train_data->get_rows());
        ccma::algebra::DenseMatrixT<T>* predict_row_mat = predict_data->get_row_data(i);

        for(uint j = 0; j < train_data->get_rows(); j++){
            ccma::algebra::DenseMatrixT<T>* train_row_mat = predict_data->get_row_data(j);

            ccma::algebra::DenseMatrixT<T>* diff_mat = new ccma::algebra::DenseMatrixT<T>();
            predict_row_mat->subtract(train_row_mat, diff_mat);

            ccma::algebra::DenseMatrixT<T>* diff_mat_t = new ccma::algebra::DenseMatrixT<T>();
            diff_mat->transpose(diff_mat_t);

            ccma::algebra::DenseMatrixT<T>* diff_mat_diff_mat_t = new ccma::algebra::DenseMatrixT<T>();
            diff_mat->product(diff_mat_t, diff_mat_diff_mat_t);

            //personalized weight for every train data with gaussian kernal
            weight->set_data(exp((real)diff_mat_diff_mat_t->get_data(0, 0) / (-2.0 * k * k)), j, j);
//printf("[%d][%d][%f]\n", j, j, weight->get_data(j, j));
            delete train_row_mat, diff_mat, diff_mat_t, diff_mat_diff_mat_t;
        }

        ccma::algebra::DenseMatrixT<real>* xTx_weight = new ccma::algebra::DenseMatrixT<real>();
        xTx->product(weight, xTx_weight);

        real* det;
        if(!xTx_weight->det(det) || *det == 0.0){
            delete x, y, xT, xTx, xTx_weight, predict_row_mat;
            return false;
        }

        ccma::algebra::DenseMatrixT<real>* weight_y = new ccma::algebra::DenseMatrixT<real>();
        weight->product(y, weight_y);

        ccma::algebra::DenseMatrixT<real>* xT_weight_y = new ccma::algebra::DenseMatrixT<real>();
        xT->product(weight_y, xT_weight_y);

        ccma::algebra::DenseMatrixT<real>* xTx_weight_I = new ccma::algebra::DenseMatrixT<real>();
        xTx_weight->inverse(xTx_weight_I);

        ccma::algebra::DenseMatrixT<real>* weight_i = new ccma::algebra::DenseMatrixT<real>();
        xTx_weight_I->product(xT_weight_y, weight_i);

        ccma::algebra::DenseMatrixT<real>* predict_mat_i = new ccma::algebra::DenseMatrixT<real>();
        predict_row_mat->product(weight_i, predict_mat_i);

        labels[i] = predict_mat_i->get_data(0);

        delete  xTx_weight, weight_y, xT_weight_y, xTx_weight_I, weight_i, predict_row_mat, predict_mat_i;
    }
    delete x, y , xT, xTx;

    if(predict_labels == nullptr){
        predict_labels = new ccma::algebra::DenseColMatrixT<real>(0.0, predict_data->get_rows());
    }
    predict_labels->set_shallow_data(labels, predict_data->get_rows(), 1);

    return true;
}




template bool LinearRegression::standard_regression(ccma::algebra::LabeledDenseMatrixT<int>* train_data, ccma::algebra::DenseColMatrixT<real>* weights);
template bool LinearRegression::standard_regression(ccma::algebra::LabeledDenseMatrixT<real>* train_data, ccma::algebra::DenseColMatrixT<real>* weights);

template bool LinearRegression::local_weight_logistic_regresion(ccma::algebra::LabeledDenseMatrixT<int>* train_data,
                                                                ccma::algebra::DenseMatrixT<int>* predict_data,
                                                                real k,
                                                                ccma::algebra::DenseColMatrixT<real>* predict_labels);
template bool LinearRegression::local_weight_logistic_regresion(ccma::algebra::LabeledDenseMatrixT<real>* train_data,
                                                                ccma::algebra::DenseMatrixT<real>* predict_data,
                                                                real k,
                                                                ccma::algebra::DenseColMatrixT<real>* predict_labels);
}//regression
}//namespace
}//namespace ccma
