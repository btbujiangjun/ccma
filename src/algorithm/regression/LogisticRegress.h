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

namespace ccma{
namespace algorithm{
namespace regression{

template<class T, class LT, class FT>
class LogisticRegress{
public:
    LogisticRegress(ccma::algebra::LabeledMatrixT<T, LT, FT>* train_data);
    ~LogisticRegress(){};

    void batch_grad_desc(uint epoch);
    void stoc_grad_desc(uint epoch);
    void smooth_stoc_grad_desc(uint epoch);

    uint classify(ccma::algebra::BaseMatrixT<T>* mat, ccma::algebra::BaseMatrixT<LT>* result);

private:
    ccma::algebra::LabeledMatrixT<T, LT, FT>* _train_data;
    ccma::algebra::BaseMatrixT<real>* _weights = nullptr;

    int signmod(ccma::algebra::BaseMatrixT<T>* mat, ccma::algebra::BaseMatrixT<real>* result);

    uint evaluate();

    void init_weights();
};

template<class T, class LT, class FT>
LogisticRegress<T, LT, FT>::LogisticRegress(ccma::algebra::LabeledMatrixT<T, LT, FT>* train_data): _train_data(train_data){
    _weights = new ccma::algebra::OneMatrix<real>(train_data->get_cols());
}

template<class T, class LT, class FT>
void LogisticRegress<T, LT, FT>::batch_grad_desc(uint epoch){
    real alpha = 0.001;
    init_weights();

    ccma::algebra::BaseMatrixT<T>* data_mat = new ccma::algebra::BaseMatrixT<T>();
    _train_data->copy_data(data_mat);

    ccma::algebra::BaseMatrixT<T>* data_t_mat = new ccma::algebra::BaseMatrixT<T>();
    data_mat->transpose(data_t_mat);

    ccma::algebra::BaseMatrixT<LT>* label_mat = new ccma::algebra::BaseMatrixT<LT>();
    _train_data->get_labels(label_mat);

    for(uint i = 0; i < epoch; i++){

        ccma::algebra::BaseMatrixT<T>* weight_mat = new ccma::algebra::BaseMatrixT<T>();
        int ret = data_mat->dot_product(_weights, weight_mat);

        ccma::algebra::BaseMatrixT<real>* h_mat = new ccma::algebra::BaseMatrixT<real>();
        signmod(weight_mat, h_mat);

        ccma::algebra::BaseMatrixT<real>* error_mat = new ccma::algebra::BaseMatrixT<real>();
        ret = label_mat->subtract(h_mat, error_mat);

        ret = error_mat->product(alpha);

        ccma::algebra::BaseMatrixT<real>* step_mat = new ccma::algebra::BaseMatrixT<real>();
        ret = data_t_mat->dot_product(error_mat, step_mat);

        ret = _weights->sum(step_mat);

        delete weight_mat, h_mat, error_mat, step_mat;

//        printf("error:[%d]\t", evaluate());
//        _weights->display();
    }
    delete data_mat, data_t_mat, label_mat;

    printf("batch error:[%d]\t", evaluate());
    _weights->display();
}

template<class T, class LT, class FT>
void LogisticRegress<T, LT, FT>::stoc_grad_desc(uint epoch){
    real alpha = 0.01;
    init_weights();

    uint rows = _train_data->get_rows();
    for(uint i = 0; i < epoch; i++){
        for(uint j = 0; j < rows; j++){
            ccma::algebra::BaseMatrixT<T>* row_mat = new ccma::algebra::BaseMatrixT<T>();
            _train_data->get_row_data(j, row_mat);

            ccma::algebra::BaseMatrixT<T>* dp_mat = new ccma::algebra::BaseMatrixT<T>();
            row_mat->dot_product(_weights, dp_mat);

            ccma::algebra::BaseMatrixT<real>* sm_mat = new ccma::algebra::BaseMatrixT<real>();
            signmod(dp_mat, sm_mat);

            real error = (real)_train_data->get_label(j) - sm_mat->get_data(0);

            ccma::algebra::BaseMatrixT<real>* h_mat = new ccma::algebra::BaseMatrixT<real>();
            row_mat->product(error * alpha, h_mat);

            ccma::algebra::BaseMatrixT<real>* h_t_mat = new ccma::algebra::BaseMatrixT<real>();
            h_mat->transpose(h_t_mat);
            _weights->sum(h_t_mat);

            delete row_mat, dp_mat, sm_mat, h_mat, h_t_mat;
            row_mat = dp_mat = sm_mat = h_mat = h_t_mat = nullptr;

//            printf("error:[%d]\n", evaluate());
//            _weights->display();
        }
    }
    printf("stoc error:[%d]\n", evaluate());
    _weights->display();
}

template<class T, class LT, class FT>
void LogisticRegress<T, LT, FT>::smooth_stoc_grad_desc(uint epoch){
    real alpha = 0.1f;
    init_weights();

    srand((int)time(0));
    uint rows = _train_data->get_rows();
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

            ccma::algebra::BaseMatrixT<T>* row_mat = new ccma::algebra::BaseMatrixT<T>();
            _train_data->get_row_data(row, row_mat);

            ccma::algebra::BaseMatrixT<T>* dp_mat = new ccma::algebra::BaseMatrixT<T>();
            row_mat->dot_product(_weights, dp_mat);

            ccma::algebra::BaseMatrixT<real>* sm_mat = new ccma::algebra::BaseMatrixT<real>();
            signmod(dp_mat, sm_mat);
            real error = (real)_train_data->get_label(row) - sm_mat->get_data(0);

            ccma:algebra::BaseMatrixT<real>* h_mat = new ccma::algebra::BaseMatrixT<real>();
            row_mat->product(error * alpha, h_mat);

            ccma::algebra::BaseMatrixT<real>* h_t_mat = new ccma::algebra::BaseMatrixT<real>();
            h_mat->transpose(h_t_mat);
            _weights->sum(h_t_mat);

            delete row_mat, dp_mat, sm_mat, h_mat, h_t_mat;
            row_mat = nullptr;
            dp_mat = sm_mat = h_mat = h_t_mat = nullptr;

//            printf("iterator[%d]row[%d]error:[%d]\n", i, row, evaluate());
//            _weights->display();
        }
    }
    delete[] rand_row_idx;
    rand_row_idx = nullptr;

    printf("smooth stoc error:[%d]\n", evaluate());
    _weights->display();
}

template<class T, class LT, class FT>
uint LogisticRegress<T, LT, FT>::classify(ccma::algebra::BaseMatrixT<T>* mat, ccma::algebra::BaseMatrixT<LT>* result){
    if(mat->get_cols() != _weights->get_rows()){
        return -1;
    }

    ccma::algebra::BaseMatrixT<real>* prob_mat = new ccma::algebra::BaseMatrixT<real>();
    mat->dot_product(_weights, prob_mat);

    ccma::algebra::BaseMatrixT<real>* signmod_mat = new ccma::algebra::BaseMatrixT<real>();
    signmod(prob_mat, signmod_mat);
    delete prob_mat;

    LT* data = new LT[signmod_mat->get_rows()];
    for(uint i = 0; i < signmod_mat->get_rows(); i++){
        if(signmod_mat->get_data(i, 0) < 0.5){
            data[i] = (LT)0;
        }else{
            data[i] = (LT)1;
        }
    }

    result->set_data(data, mat->get_rows(), 1);
    delete[] data;

    return 1;
}

template<class T, class LT, class FT>
int LogisticRegress<T, LT, FT>::signmod(ccma::algebra::BaseMatrixT<T>* mat, ccma::algebra::BaseMatrixT<real>* result){
    real* data = new real[mat->get_rows() * mat->get_cols()];
    uint new_data_idx = 0;

    for(uint i = 0; i < mat->get_rows(); i++){
        for(uint j = 0; j < mat->get_cols(); j++){
            data[new_data_idx++] = 1.0f/(1.0f+exp(-mat->get_data(i, j)));
        }
    }

    result->set_data(data, mat->get_rows(), mat->get_cols());
    delete[] data;

    return 1;
}

template<class T, class LT, class FT>
uint LogisticRegress<T, LT, FT>::evaluate(){
    ccma::algebra::BaseMatrixT<LT>* predict = new ccma::algebra::BaseMatrixT<LT>();
    if(classify(_train_data, predict) > 0){
        uint k = 0;
        for(uint i = 0; i < _train_data->get_rows(); i++){
            if(predict->get_data(i, 0) != _train_data->get_label(i)){
                k++;
            }
        }
        return k;
    }
    return 999999;
}

template<class T, class LT, class FT>
void LogisticRegress<T, LT, FT>::init_weights(){
    if(_weights != nullptr){
        delete _weights;
        _weights = nullptr;
    }
    _weights = new ccma::algebra::OneMatrix<real>(_train_data->get_cols());
}

}//namespace regression
}//namespace algorithm
}//namespace ccml

#endif //_CCMA_ALGORITHM_REGRESSION_REGRESS_H
