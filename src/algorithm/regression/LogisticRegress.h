/*********************************************
* Author: Jun Jiang - jiangjun4@sina.com
* Created: 2016-12-01 15:35
* Last modified: 2016-12-01 15:35
* Filename: regress.h
* Description: regression model 
**********************************************/
#ifndef _CCMA_ALGORITHM_REGRESSION_REGRESS_H
#define _CCMA_ALGORITHM_REGRESSION_REGRESS_H

#include <time.h>
#include "algebra/BaseMatrix.h"
#include "utils/MatrixHelper.h"

namespace ccma{
namespace algorithm{
namespace regression{

class LogisticRegress{
public:
    LogisticRegress(){
        _helper = new ccma::utils::MatrixHelper();
    }
    ~LogisticRegress(){
        if(_weights != nullptr){
            delete _weights;
        }
        if(_helper != nullptr){
            delete _helper;
        }
    }

    template<class T>
    void batch_grad_desc(ccma::algebra::LabeledDenseMatrixT<T>* train_data, uint epoch);
    template<class T>
    void stoc_grad_desc(ccma::algebra::LabeledDenseMatrixT<T>* train_data, uint epoch);
    template<class T>
    void smooth_stoc_grad_desc(ccma::algebra::LabeledDenseMatrixT<T>* train_data, uint epoch);

    template<class T>
    uint classify(ccma::algebra::BaseMatrixT<T>* mat, ccma::algebra::BaseMatrixT<T>* result);

private:
    ccma::algebra::BaseMatrixT<real>* _weights = nullptr;
    ccma::utils::MatrixHelper* _helper = nullptr;

    template<class T>
    uint evaluate(ccma::algebra::LabeledDenseMatrixT<T>* train_data);

    void init_weights(uint size);
};

template<class T>
void LogisticRegress::batch_grad_desc(ccma::algebra::LabeledDenseMatrixT<T>* train_data, uint epoch){
    real alpha = 0.001;
    init_weights(train_data->get_cols());

    ccma::algebra::BaseMatrixT<T>* data_mat = train_data->clone();

    ccma::algebra::BaseMatrixT<T>* data_t_mat = new ccma::algebra::DenseMatrixT<T>();
    _helper->transpose(data_mat, data_t_mat);

    ccma::algebra::BaseMatrixT<T>* label_mat = train_data->get_labels();

    for(uint i = 0; i < epoch; i++){

        ccma::algebra::BaseMatrixT<T>* weight_mat = new ccma::algebra::DenseMatrixT<T>();
        if(!_helper->product(data_mat,_weights, weight_mat)){
            delete[] data_mat, data_t_mat, label_mat;
        }

        ccma::algebra::BaseMatrixT<real>* h_mat = new ccma::algebra::DenseMatrixT<real>();
        _helper->signmod(weight_mat, h_mat);

        ccma::algebra::BaseMatrixT<real>* error_mat = new ccma::algebra::DenseMatrixT<real>();
        _helper->subtract(label_mat, h_mat, error_mat);

        error_mat->multiply(alpha);

        ccma::algebra::BaseMatrixT<real>* step_mat = new ccma::algebra::DenseMatrixT<real>();
        _helper->product(data_t_mat, error_mat, step_mat);

        _weights->add(step_mat);

        delete weight_mat, h_mat, error_mat, step_mat;

//        printf("error:[%d]\t", evaluate(train_data));
//        _weights->display();
    }
    delete data_mat, data_t_mat, label_mat;

    printf("batch error:[%d]\t", evaluate(train_data));
    _weights->display();
}

template<class T>
void LogisticRegress::stoc_grad_desc(ccma::algebra::LabeledDenseMatrixT<T>* train_data, uint epoch){
    real alpha = 0.01;
    init_weights(train_data->get_cols());

    uint rows = train_data->get_rows();
    for(uint i = 0; i < epoch; i++){
        for(uint j = 0; j < rows; j++){
            ccma::algebra::BaseMatrixT<T>* row_mat = train_data->get_row_data(j);

            ccma::algebra::BaseMatrixT<T>* dp_mat = new ccma::algebra::DenseMatrixT<T>();
            _helper->product(row_mat, _weights, dp_mat);

            ccma::algebra::BaseMatrixT<real>* sm_mat = new ccma::algebra::DenseMatrixT<real>();
            _helper->signmod(dp_mat, sm_mat);

            real error = (real)train_data->get_label(j) - sm_mat->get_data(0);

            ccma::algebra::BaseMatrixT<real>* h_mat = new ccma::algebra::DenseMatrixT<real>();
            _helper->product(row_mat, error * alpha, h_mat);

            ccma::algebra::BaseMatrixT<real>* h_t_mat = new ccma::algebra::DenseMatrixT<real>();
            _helper->transpose(h_mat, h_t_mat);
            _weights->add(h_t_mat);

            delete row_mat, dp_mat, sm_mat, h_mat, h_t_mat;
            row_mat = dp_mat = sm_mat = h_mat = h_t_mat = nullptr;

//            printf("error:[%d]\n", evaluate(train_data));
//            _weights->display();
        }
    }
    printf("stoc error:[%d]\n", evaluate(train_data));
    _weights->display();
}

template<class T>
void LogisticRegress::smooth_stoc_grad_desc(ccma::algebra::LabeledDenseMatrixT<T>* train_data, uint epoch){
    real alpha = 0.1f;
    init_weights(train_data->get_cols());

    srand((int)time(0));
    uint rows = train_data->get_rows();
    int* rand_row_idx = new int[rows];
    uint row = 0;

    for(uint i = 0; i < epoch; i++){
        //init random sampling array, avoid cyclical fluctuations
        for(uint k = 0; k < rows; k++){
            rand_row_idx[k] = k;
        }

        for(uint j = 0; j < rows; j++){
            alpha = 4/(1.0 + i + j) + 0.0001;

            uint raw_row = (uint)rand() % (rows - j);
            uint m = 0, n = 0;
            for(; m < rows; m++){
                if(rand_row_idx[m] >= 0){
                    n++;
                }
                if(n == raw_row + 1){
                    break;
                }
            }
            row = rand_row_idx[m];
            rand_row_idx[m] = -1;//avoid sampling twice

            ccma::algebra::BaseMatrixT<T>* row_mat = train_data->get_row_data(row);

            ccma::algebra::BaseMatrixT<T>* dp_mat = new ccma::algebra::DenseMatrixT<T>();
            _helper->product(row_mat, _weights, dp_mat);

            ccma::algebra::BaseMatrixT<real>* sm_mat = new ccma::algebra::DenseMatrixT<real>();
            _helper->signmod(dp_mat, sm_mat);
            real error = (real)train_data->get_label(row) - sm_mat->get_data(0);

            ccma:algebra::BaseMatrixT<real>* h_mat = new ccma::algebra::DenseMatrixT<real>();
            _helper->product(row_mat, error * alpha, h_mat);

            ccma::algebra::BaseMatrixT<real>* h_t_mat = new ccma::algebra::DenseMatrixT<real>();
            _helper->transpose(h_mat, h_t_mat);
            _weights->add(h_t_mat);

            delete row_mat, dp_mat, sm_mat, h_mat, h_t_mat;
            row_mat = nullptr;
            dp_mat = sm_mat = h_mat = h_t_mat = nullptr;

//            printf("iterator[%d]row[%d]error:[%d]\n", i, row, evaluate(train_data));
//            _weights->display();
        }
    }
    delete[] rand_row_idx;
    rand_row_idx = nullptr;

    printf("smooth stoc error:[%d]\n", evaluate(train_data));
    _weights->display();
}

template<class T>
uint LogisticRegress::classify(ccma::algebra::BaseMatrixT<T>* mat, ccma::algebra::BaseMatrixT<T>* result){
    if(mat->get_cols() != _weights->get_rows()){
        return -1;
    }

    ccma::algebra::BaseMatrixT<real>* prob_mat = new ccma::algebra::DenseMatrixT<real>();
    _helper->product(mat, _weights, prob_mat);

    ccma::algebra::BaseMatrixT<real>* signmod_mat = new ccma::algebra::DenseMatrixT<real>();
    _helper->signmod(prob_mat, signmod_mat);
    delete prob_mat;

    T* data = new T[signmod_mat->get_rows()];
    for(uint i = 0; i < signmod_mat->get_rows(); i++){
        if(signmod_mat->get_data(i, 0) < 0.5){
            data[i] = (T)0;
        }else{
            data[i] = (T)1;
        }
    }

    result->set_shallow_data(data, mat->get_rows(), 1);

    return 1;
}

template<class T>
uint LogisticRegress::evaluate(ccma::algebra::LabeledDenseMatrixT<T>* train_data){
    ccma::algebra::BaseMatrixT<T>* predict = new ccma::algebra::DenseMatrixT<T>();
    if(classify(train_data, predict) > 0){
        uint k = 0;
        for(uint i = 0; i < train_data->get_rows(); i++){
            if(predict->get_data(i, 0) != train_data->get_label(i)){
                k++;
            }
        }
        return k;
    }
    return 999999;
}

void LogisticRegress::init_weights(uint size){
    if(_weights != nullptr){
        delete _weights;
        _weights = nullptr;
    }
    _weights = new ccma::algebra::DenseOneMatrixT<real>(size);
}

}//namespace regression
}//namespace algorithm
}//namespace ccml

#endif //_CCMA_ALGORITHM_REGRESSION_REGRESS_H
