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

    auto seq_data  = new ccma::algebra::DenseMatrixT<real>();
	auto seq_label = new ccma::algebra::DenseMatrixT<real>();
	
	uint num_train_data = train_seq_data->size();
	bool debug = (num_train_data <= 5);
	for(uint i = 0; i != epoch; i++){
		for(uint j = 0; j != num_train_data; j++){
			train_seq_data->at(j)->clone(seq_data);
			train_seq_label->at(j)->clone(seq_label);

            printf("starting [%d][%d].\n", i, j);

			if(seq_data->get_rows() != seq_label->get_rows()){
				printf("Data Error[%d][%d-%d].\n", j, seq_data->get_rows(), seq_label->get_rows());
				continue;
			}

			sgd_step(seq_data, seq_label, alpha, debug);

//			if(j % 1 == 0){
                printf("Epoch[%d][%d/%d]training, loss[%f]...\n", i, j, num_train_data, loss(train_seq_data, train_seq_label));
//			}
		}
	}

	printf("training finished.\n");

    delete seq_data;
	delete seq_label;
}

void RNN::sgd_step(ccma::algebra::BaseMatrixT<real>* train_seq_data,
              ccma::algebra::BaseMatrixT<real>* train_seq_label, 
			  real alpha,
              bool debug){

    auto derivate_weight     = new ccma::algebra::DenseMatrixT<real>();
    auto derivate_pre_weight = new ccma::algebra::DenseMatrixT<real>();
    auto derivate_act_weight = new ccma::algebra::DenseMatrixT<real>();

	_layer->back_propagation(train_seq_data, train_seq_label, _U, _W, _V, derivate_weight, derivate_pre_weight, derivate_act_weight, debug);
	
	derivate_weight->multiply(alpha);
	_U->subtract(derivate_weight);

	derivate_pre_weight->multiply(alpha);
	_W->subtract(derivate_pre_weight);

	derivate_act_weight->multiply(alpha);
	_V->subtract(derivate_act_weight);

    delete derivate_weight;
    delete derivate_pre_weight;
    delete derivate_act_weight;
}

real RNN::loss(std::vector<ccma::algebra::BaseMatrixT<real>*>* train_seq_data,
               std::vector<ccma::algebra::BaseMatrixT<real>*>* train_seq_label){
	real l = 0;
	
	auto seq_data  = new ccma::algebra::DenseMatrixT<real>();
	auto seq_label = new ccma::algebra::DenseMatrixT<real>();
	
	auto store      = new ccma::algebra::DenseMatrixT<real>();
	auto activation = new ccma::algebra::DenseMatrixT<real>();
	
    uint num_train_data = train_seq_data->size();
	for(uint j = 0; j != num_train_data; j++){
        printf("start loss1 %d:[%f]\n", j, l);

		train_seq_data->at(j)->clone(seq_data);
		train_seq_label->at(j)->clone(seq_label);
		_layer->feed_farward(seq_data, seq_label, _U, _W, _V, store, activation, debug);

		auto mat_label = seq_label->argmax(0);
       
        mat_label->display("|");

        printf("start loss3 %d:[%f]\n", j, l);

        uint rows = mat_label->get_rows();

        for(uint row = 0; row != rows; row++){
            l -= std::log(activation->get_data(mat_label->get_data(row, 0), row)); 
        }

        printf("loss %d:[%f]\n", j, l);
	}

    delete seq_data;
	delete seq_label;

    delete store;
    delete activation;

	l /= num_train_data;

	return l;
}

}//namespace rnn
}//namespace algorithm
}//namespace ccma
