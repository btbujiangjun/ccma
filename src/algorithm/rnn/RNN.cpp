/***********************************************
 * Author: Jun Jiang - jiangjun4@sina.com
 * Create: 2017-06-23 14:32
 * Last modified : 2017-06-23 14:32
 * Filename      : RNN.cpp
 * Description   : 
 **********************************************/

#include "algorithm/rnn/RNN.h"

namespace ccma{
namespace algorithm{
namespace rnn{

void RNN::sgd(std::vector<ccma::algebra::BaseMatrixT<real>*>* train_seq_data,
              std::vector<ccma::algebra::BaseMatrixT<real>*>* train_seq_label, 
              uint epoch, 
              real alpha){
	auto seq_data = new ccma::algebra::DenseMatrixT<real>();
	auto seq_label = new ccma::algebra::DenseMatrixT<real>();
	
	uint num_train_data = train_seq_data->size();
	bool debug = (num_train_data <= 5);
	for(uint i = 0; i != epoch; i++){
		for(uint j = 0; j != num_train_data; j++){
			train_seq_data->at(j)->clone(seq_data);
			train_seq_label->at(j)->clone(seq_label);
			

			if(seq_data->get_rows() != seq_label->get_rows()){
				printf("Data Error[%d][%d-%d].\n", j, seq_data->get_rows(), seq_label->get_rows());
				break;
			}

			sgd_step(seq_data, seq_label, debug,alpha);

			if(j % 1 == 0){
                printf("Epoch[%d][%d/%d]training, loss[%f]...\r", i, j, num_train_data, loss(train_seq_data, train_seq_label));
			}
		}
	}

	printf("training finished.\n");

    delete seq_data;
	delete seq_label;
}

void RNN::sgd_step(ccma::algebra::BaseMatrixT<real>* train_seq_data,
              ccma::algebra::BaseMatrixT<real>* train_seq_label, 
              bool debug,
			  real alpha){
	_layer->back_propagation(train_seq_data, train_seq_label, debug);
	
	auto u = _layer->get_derivate_weight();
	u->multiply(alpha);
	_U->subtract(u);

	auto w = _layer->get_derivate_pre_weight();
	w->multiply(alpha);
	_W->subtract(w);

	auto v = _layer->get_derivate_act_weight();
	v->multiply(alpha);
	_V->subtract(v);
}
real RNN::loss(std::vector<ccma::algebra::BaseMatrixT<real>*>* train_seq_data,
               std::vector<ccma::algebra::BaseMatrixT<real>*>* train_seq_label){
	real l = 0;
	
	auto seq_data = new ccma::algebra::DenseMatrixT<real>();
	auto seq_label = new ccma::algebra::DenseMatrixT<real>();
	auto predict_mat = new ccma::algebra::DenseMatrixT<real>();
	
	uint num_train_data = train_seq_data->size();
	for(uint j = 0; j != num_train_data; j++){
		train_seq_data->at(j)->clone(seq_data);
		train_seq_label->at(j)->clone(seq_label);
		_layer->feed_farward(seq_data, seq_label);

		_layer->get_activation()->clone(predict_mat);
		predict_mat->dot(seq_label);

		predict_mat->log();
		l -= predict_mat->sum();
	}

    delete seq_data;
	delete seq_label;
	delete predict_mat;

	l /= num_train_data;

	return l;
}

}//namespace rnn
}//namespace algorithm
}//namespace ccma
