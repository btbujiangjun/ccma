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
			_layer->back_propagation(seq_data, seq_label, debug);

			if(j % 1 == 0){
                printf("Epoch[%d][%d/%d]training...\r", i, j, num_train_data);
			}
		}
	}

	printf("training finished.\n");

    delete seq_data;
	delete seq_label;
}


real RNN::loss(std::vector<ccma::algebra::BaseMatrixT<real>*>* train_seq_data,
               std::vector<ccma::algebra::BaseMatrixT<real>*>* train_seq_data){
	real l = 0;
	
	auto seq_data = new ccma::algebra::DenseMatrixT<real>();
	auto seq_label = new ccma::algebra::DenseMatrixT<real>();
	
	uint num_train_data = train_seq_data->size();
	for(uint j = 0; j != num_train_data; j++){
		train_seq_data->at(j)->clone(seq_data);
		train_seq_label->at(j)->clone(seq_label);
		_layer->feed_farward(seq_data, seq_label, false);
	}

    delete seq_data;
	delete seq_label;

	return l;
}

}//namespace rnn
}//namespace algorithm
}//namespace ccma
