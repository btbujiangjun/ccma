/*********************************************
* Author: Jun Jiang - jiangjun4@sina.com
* Created: 2016-12-07 15:14
* Last modified: 2016-12-07 15:14
* Filename: LinearRegress.h
* Description:linear regression
**********************************************/
#include "LinearRegress.h"
#include <stdio.h>
#include "utils/MatrixHelper.h"

namespace ccma{
namespace algorithm{
namespace regression{


template<class T>
bool LinearRegression::standard_regression(ccma::algebra::LabeledDenseMatrixT<T>* train_data, ccma::algebra::DenseColMatrixT<real>* weights){

    auto x = new ccma::algebra::DenseMatrixT<T>();
    train_data->clone(x);

    auto y = new ccma::algebra::DenseMatrixT<T>();
    train_data->get_labels(y);

    ccma::algebra::BaseMatrixT<T>* xT = new ccma::algebra::DenseMatrixT<T>();
    _helper->transpose(x, xT);

    ccma::algebra::BaseMatrixT<T>* xTx = new ccma::algebra::DenseMatrixT<T>();
    if(_helper->dot(xT, x, xTx)){
        delete x;
	delete y;
	delete xT;
	delete xTx;
        return false;
    }

    ccma::algebra::BaseMatrixT<real>* xTxI = new ccma::algebra::DenseMatrixT<real>();
    if(!xTx->inverse(xTxI)){
        delete x;
	delete y;
	delete xT;
	delete xTx;
	delete xTxI;
        return false;
    }

    ccma::algebra::BaseMatrixT<T>* xTy = new ccma::algebra::DenseMatrixT<T>();
    _helper->dot(xT, y, xTy);
    _helper->dot(xTxI, xTy, weights);

    delete x;
    delete y;
    delete xT;
    delete xTx;
    delete xTy;
    delete xTxI;

    return true;
}


template<class T>
bool LinearRegression::local_weight_logistic_regression(ccma::algebra::LabeledDenseMatrixT<T>* train_data,
                                                        ccma::algebra::DenseMatrixT<T>* predict_data,
                                                        const real k,
                                                        ccma::algebra::DenseColMatrixT<real>* predict_labels){
    if(train_data->get_cols() != predict_data->get_cols()){
        return false;
    }

    real* labels = new real[predict_data->get_rows()];

    auto x = new ccma::algebra::DenseMatrixT<T>();
    train_data->get_data_matrix(x);

    auto y = new ccma::algebra::DenseMatrixT<T>();
    train_data->get_labels(y);

    ccma::algebra::DenseMatrixT<T>* xT = new ccma::algebra::DenseMatrixT<T>();
    _helper->transpose(x, xT);

    for(uint i = 0 ; i < predict_data->get_rows(); i++){
        ccma::algebra::BaseMatrixT<real>* weight = new ccma::algebra::DenseEyeMatrixT<real>(train_data->get_rows());
        auto predict_row_mat = new ccma::algebra::DenseMatrixT<T>();
        predict_data->get_row_data(i, predict_row_mat);

        for(uint j = 0; j < train_data->get_rows(); j++){
            auto train_row_mat = new ccma::algebra::LabeledDenseMatrixT<T>();
            train_data->get_row_data(j, train_row_mat);

            ccma::algebra::DenseMatrixT<T>* diff_mat = new ccma::algebra::DenseMatrixT<T>();
            _helper->subtract(predict_row_mat, train_row_mat, diff_mat);

            ccma::algebra::DenseMatrixT<T>* diff_mat_t = new ccma::algebra::DenseMatrixT<T>();
            _helper->transpose(diff_mat, diff_mat_t);

            ccma::algebra::DenseMatrixT<T>* diff_mat_diff_mat_t = new ccma::algebra::DenseMatrixT<T>();
            _helper->dot(diff_mat, diff_mat_t, diff_mat_diff_mat_t);

            //personalized weight for every train data with gaussian kernal
            weight->set_data(exp((real)diff_mat_diff_mat_t->get_data(0, 0) / (-2.0 * k * k)), j, j);
            delete train_row_mat;
	    delete diff_mat;
	    delete diff_mat_t;
	    delete diff_mat_diff_mat_t;
        }

        ccma::algebra::DenseMatrixT<real>* weight_x = new ccma::algebra::DenseMatrixT<real>();
        _helper->dot(weight, x, weight_x);

        ccma::algebra::DenseMatrixT<real>* xTx = new ccma::algebra::DenseMatrixT<real>();
        _helper->dot(xT, weight_x, xTx);

        real det = 0.0;
        if(!xTx->det(&det) || det == 0.0){
            delete x;
	    delete y;
	    delete xT;
	    delete weight_x;
	    delete xTx;
	    delete predict_row_mat;
            return false;
        }

        ccma::algebra::DenseMatrixT<real>* weight_y = new ccma::algebra::DenseMatrixT<real>();
        _helper->dot(weight, y, weight_y);

        ccma::algebra::DenseMatrixT<real>* xT_weight_y = new ccma::algebra::DenseMatrixT<real>();
        _helper->dot(xT, weight_y, xT_weight_y);

        ccma::algebra::DenseMatrixT<real>* xTxI = new ccma::algebra::DenseMatrixT<real>();
        xTx->inverse(xTxI);

        ccma::algebra::DenseMatrixT<real>* weight_i = new ccma::algebra::DenseMatrixT<real>();
        _helper->dot(xTxI, xT_weight_y, weight_i);

        ccma::algebra::DenseMatrixT<real>* predict_mat_i = new ccma::algebra::DenseMatrixT<real>();
        _helper->dot(predict_row_mat, weight_i, predict_mat_i);

        labels[i] = predict_mat_i->get_data(0);

        delete weight_x;
	delete xTx;
	delete weight_y;
	delete xT_weight_y;
	delete xTxI;
	delete weight_i;
	delete predict_mat_i;
	delete predict_row_mat;
    }
    delete x;
    delete y;
    delete xT;

    predict_labels->set_shallow_data(labels, predict_data->get_rows(), 1);

    return true;
}

template<class T>
bool LinearRegression::ridge_regression(ccma::algebra::LabeledDenseMatrixT<T>* train_data,
                                        const real lamda,
                                        ccma::algebra::DenseColMatrixT<real>* weights){

    auto x = new ccma::algebra::DenseMatrixT<T>();
    train_data->get_data_matrix(x);

    auto y = new ccma::algebra::DenseMatrixT<T>();
    train_data->get_labels(y);

    ccma::algebra::DenseMatrixT<T>* xT = new ccma::algebra::DenseMatrixT<T>();
    _helper->transpose(x, xT);

    ccma::algebra::DenseMatrixT<T>* xTx = new ccma::algebra::DenseMatrixT<T>();
    _helper->dot(xT, x, xTx);

    ccma::algebra::DenseEyeMatrixT<real>* eye = new ccma::algebra::DenseEyeMatrixT<real>(x->get_cols());
    eye->add(lamda);
    _helper->add(eye, xTx, eye);

    real* result;
    if(!eye->det(result) || *result == 0.0){
        delete x;
	delete y;
	delete xT;
	delete xTx;
	delete eye;

        return false;
    }

    ccma::algebra::DenseMatrixT<real>* eyeI = new ccma::algebra::DenseMatrixT<real>();
    eye->inverse(eyeI);

    ccma::algebra::DenseMatrixT<T>* xTy = new ccma::algebra::DenseMatrixT<T>();
    _helper->dot(xT, y, xTy);

    _helper->dot(eyeI, xTy, weights);

    delete x;
    delete y;
    delete xT;
    delete xTx;
    delete eye;
    delete eyeI;
    delete xTy;

    return true;
}


template bool LinearRegression::standard_regression(ccma::algebra::LabeledDenseMatrixT<int>* train_data, ccma::algebra::DenseColMatrixT<real>* weights);
template bool LinearRegression::standard_regression(ccma::algebra::LabeledDenseMatrixT<real>* train_data, ccma::algebra::DenseColMatrixT<real>* weights);

template bool LinearRegression::local_weight_logistic_regression(ccma::algebra::LabeledDenseMatrixT<int>* train_data, ccma::algebra::DenseMatrixT<int>* predict_data, const real k, ccma::algebra::DenseColMatrixT<real>* predict_labels);
template bool LinearRegression::local_weight_logistic_regression(ccma::algebra::LabeledDenseMatrixT<real>* train_data, ccma::algebra::DenseMatrixT<real>* predict_data, const real k, ccma::algebra::DenseColMatrixT<real>* predict_labels);

template bool LinearRegression::ridge_regression(ccma::algebra::LabeledDenseMatrixT<int>* train_data, const real lamda, ccma::algebra::DenseColMatrixT<real>* weights);
template bool LinearRegression::ridge_regression(ccma::algebra::LabeledDenseMatrixT<real>* train_data, const real lamda, ccma::algebra::DenseColMatrixT<real>* weights);
}//regression
}//namespace
}//namespace ccma
