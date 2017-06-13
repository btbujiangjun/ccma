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
#include "utils/Shuffler.h"

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
			_weights = nullptr;
        }
        if(_helper != nullptr){
            delete _helper;
			_helper = nullptr;
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

    auto data_t_mat = new ccma::algebra::DenseMatrixT<T>();
    train_data->clone(data_t_mat);
	data_t_mat->transpose();

    auto label_mat = new ccma::algebra::DenseMatrixT<T>();
    train_data->get_labels(label_mat);

    auto activation_mat = new ccma::algebra::DenseMatrixT<T>();

    for(uint i = 0; i < epoch; i++){
		train_data->clone(activation_mat);
		//activation = sigmoid(x .* weight)
		activation_mat->dot(_weights);
		activation_mat->sigmoid();

		//error = activation - label
		activation_mat->subtract(label_mat);

		//step = data_t_mat * error_mat * alpha
        activation_mat->multiply(alpha);
        _helper->dot(data_t_mat, activation_mat, activation_mat);
		//update weight
        _weights->subtract(activation_mat);
    }
    delete data_t_mat;
    delete label_mat;
	delete activation_mat;

    printf("batch error:[%d]/[%d]\t", evaluate(train_data), train_data->get_rows());
    _weights->display();
}

template<class T>
void LogisticRegress::stoc_grad_desc(ccma::algebra::LabeledDenseMatrixT<T>* train_data, uint epoch){
    real alpha = 0.01;
    init_weights(train_data->get_cols());
    uint rows = train_data->get_rows();

	auto train_data_mat = new ccma::algebra::DenseMatrixT<T>();
	train_data->get_data_matrix(train_data_mat);

    auto activation_mat = new ccma::algebra::DenseMatrixT<T>();
    auto row_t_mat = new ccma::algebra::DenseMatrixT<T>();

    for(uint i = 0; i < epoch; i++){
        for(uint j = 0; j < rows; j++){
            train_data_mat->get_row_data(j, activation_mat);
            train_data_mat->get_row_data(j, row_t_mat);
			row_t_mat->transpose();

			//activation = sigmoid(x .* weight)
			activation_mat->dot(_weights);
			activation_mat->sigmoid();

			//error = activation - label
            real error = activation_mat->get_data(0) - train_data->get_label(j);
			//step = x.T * error * alpha
            row_t_mat->multiply(error * alpha);
			//update weight
            _weights->subtract(row_t_mat);
        }
    }

	delete activation_mat;
	delete row_t_mat;

    printf("stoc error:[%d][%d]\n", evaluate(train_data),train_data->get_rows());
    _weights->display();
}

template<class T>
void LogisticRegress::smooth_stoc_grad_desc(ccma::algebra::LabeledDenseMatrixT<T>* train_data, uint epoch){
    real alpha = 0.1f;
    init_weights(train_data->get_cols());

	auto train_data_mat = new ccma::algebra::DenseMatrixT<T>();
	train_data->get_data_matrix(train_data_mat);
	
    uint rows = train_data->get_rows();
	ccma::utils::Shuffler shuffler(rows);
    auto activation_mat = new ccma::algebra::DenseMatrixT<T>();
    auto data_t_mat = new ccma::algebra::DenseMatrixT<T>();

    for(uint i = 0; i < epoch; i++){
		shuffler.shuffle();

        for(uint j = 0; j < rows; j++){
            alpha = 4/(1.0 + i + j) + 0.0001;

            train_data_mat->get_row_data(shuffler.get_row(j), activation_mat);
            train_data_mat->get_row_data(shuffler.get_row(j), data_t_mat);
			data_t_mat->transpose();

			//activation = sigmoid(x.* weight)
			activation_mat->dot(_weights);
			activation_mat->sigmoid();
			//error = activation -label
            real error = activation_mat->get_data(0) - train_data->get_label(shuffler.get_row(j));
			//step = x.T * error * alpha;
			data_t_mat->multiply(error * alpha);

            _weights->subtract(data_t_mat);
        }
    }
	delete activation_mat;
	delete data_t_mat;

    printf("smooth stoc error:[%d][%d]\n", evaluate(train_data), train_data->get_rows());
    _weights->display();
}

template<class T>
uint LogisticRegress::classify(ccma::algebra::BaseMatrixT<T>* mat, ccma::algebra::BaseMatrixT<T>* result){
    if(mat->get_cols() != _weights->get_rows()){
        return -1;
    }

    auto activation_mat = new ccma::algebra::DenseMatrixT<real>();
	mat->clone(activation_mat);

	activation_mat->dot(_weights);
	activation_mat->sigmoid();

	uint rows = activation_mat->get_rows();
    T* data = new T[rows];
	auto predict_data = activation_mat->get_data();
    for(uint i = 0; i != rows; i++){
        if(predict_data[i] < 0.5){
            data[i] = (T)0;
        }else{
            data[i] = (T)1;
        }
    }
	delete activation_mat;

    result->set_shallow_data(data, mat->get_rows(), 1);

    return 1;
}

template<class T>
uint LogisticRegress::evaluate(ccma::algebra::LabeledDenseMatrixT<T>* train_data){
    ccma::algebra::BaseMatrixT<T>* predict = new ccma::algebra::DenseMatrixT<T>();
    if(classify(train_data, predict) > 0){
        uint k = 0;
		uint rows = train_data->get_rows();
        for(uint i = 0; i != rows; i++){
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
