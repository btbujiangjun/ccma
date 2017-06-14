/*********************************************
* Author: Jun Jiang - jiangjun4@sina.com
* Created: 2016-12-07 15:14
* Last modified: 2016-12-07 15:14
* Filename: LinearRegress.h
* Description:linear regression
**********************************************/
#include "algorithm/regression/LinearRegress.h"
#include <stdio.h>
#include "utils/MatrixHelper.h"

namespace ccma{
namespace algorithm{
namespace regression{


template<class T>
bool LinearRegression::standard_regression(ccma::algebra::LabeledDenseMatrixT<T>* train_data, ccma::algebra::DenseColMatrixT<real>* weights){

    bool result_value = false;

    auto x = new ccma::algebra::DenseMatrixT<T>();
    train_data->clone(x);

    auto xT = new ccma::algebra::DenseMatrixT<T>();
    x->clone(xT);
    xT->transpose();

    //xTx = x .* x.T
    auto xTx = new ccma::algebra::DenseMatrixT<T>();
    xT->clone(xTx);
    xTx->dot(x);

    auto xTxI = new ccma::algebra::DenseMatrixT<real>();
    //xTx.inverse()
    if(xTx->inverse(xTxI)){
        auto y = new ccma::algebra::DenseMatrixT<T>();
        train_data->get_labels(y);

        xT->dot(y);//xTy
		_helper->dot(xTxI, xT, weights);
        
        delete y;

        result_value = true;
    }

    delete x;
    delete xT;
    delete xTx;
    delete xTxI;

    return result_value;
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

    auto xT = new ccma::algebra::DenseMatrixT<T>();
    x->clone(xT);
    xT->transpose();

    uint predict_row = predict_data->get_rows();
    uint train_row = train_data->get_rows();

    for(uint i = 0 ; i != predict_row; i++){
        auto weight = new ccma::algebra::DenseEyeMatrixT<real>(train_data->get_rows());
        auto predict_row_mat = new ccma::algebra::DenseMatrixT<T>();
        predict_data->get_row_data(i, predict_row_mat);

        for(uint j = 0; j != train_row; j++){
            auto train_row_mat = new ccma::algebra::LabeledDenseMatrixT<T>();
            train_data->get_row_data(j, train_row_mat);

            auto diff_mat = new ccma::algebra::DenseMatrixT<T>();
            _helper->subtract(predict_row_mat, train_row_mat, diff_mat);

            auto diff_mat_t = new ccma::algebra::DenseMatrixT<T>();
            _helper->transpose(diff_mat, diff_mat_t);

            auto diff_mat_diff_mat_t = new ccma::algebra::DenseMatrixT<T>();
            _helper->dot(diff_mat, diff_mat_t, diff_mat_diff_mat_t);

            //personalized weight for every train data with gaussian kernal
            weight->set_data(exp((real)diff_mat_diff_mat_t->get_data(0, 0) / (-2.0 * k * k)), j, j);
            delete train_row_mat;
	        delete diff_mat;
    	    delete diff_mat_t;
	        delete diff_mat_diff_mat_t;
        }

        auto weight_x = new ccma::algebra::DenseMatrixT<real>();
        _helper->dot(weight, x, weight_x);

        auto xTx = new ccma::algebra::DenseMatrixT<real>();
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

        auto weight_y = new ccma::algebra::DenseMatrixT<real>();
        _helper->dot(weight, y, weight_y);

        auto xT_weight_y = new ccma::algebra::DenseMatrixT<real>();
        _helper->dot(xT, weight_y, xT_weight_y);

        auto xTxI = new ccma::algebra::DenseMatrixT<real>();
        xTx->inverse(xTxI);

        auto weight_i = new ccma::algebra::DenseMatrixT<real>();
        _helper->dot(xTxI, xT_weight_y, weight_i);

        auto predict_mat_i = new ccma::algebra::DenseMatrixT<real>();
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
