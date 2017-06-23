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
	
	uint train_data_size = train_seq_data->size();
	bool debug = (train_data_size <= 5);
	for(uint i = 0; i != epoch; i++){
		for(uint j = 0; j != train_data_size; j++){
			train_seq_data->at(j)->clone(seq_data);
			train_seq_label->at(j)->clone(seq_label);
			_layer->back_propagation(seq_data, seq_label, debug);
		}
	}
	delete seq_data;
	delete seq_label;
}

}//namespace rnn
}//namespace algorithm
}//namespace ccma
