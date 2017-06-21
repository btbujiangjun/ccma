/***********************************************
 * Author: Jun Jiang - jiangjun4@sina.com
 * Create: 2017-06-20 17:05
 * Last modified : 2017-06-20 17:05
 * Filename      : Layer.cpp
 * Description   : RNN network Layer 
 **********************************************/

#include "algorithm/rnn/Layer.h"

namespace ccma{
namespace algorithm{
namespace rnn{

void Layer::initialize(){
	for(auto&& mat : _store){
		delete mat;
	}
	_store.clear();

	for(auto&& mat : _activation){
		delete mat;
	}
	_activation.clear();
}

void feed_farward(ccma::algebra::BaseMatrixT<real>* train_seq_data, bool debug){
	
	initialize();

	uint seq_size = train_seq_data->get_rows();

	auto seq_data = ccma::algebra::DenseMatrixT<real>();
	for(uint i = 0; i != seq_size; i++){
		train_seq_data->get_row_data(i, seq_data);
		auto store = ccma::algebra::DenseMatrixT<real>();
		auto activation = ccma::algebra::DenseMatrixT<real>();

		_weight->clone(store);
		store->dot(seq_data);

		if(i > 0){
			auto pre_activation = ccma::algebra::DenseMatrixT<real>();
			_pre_weight->clone(pre_activation);
			pre_activation->dot(_activation[i-1]);
			store->add(pre_activation);
			delete pre_activation;
		}

		_store.push_back(store);

		_act_weight->clone(activation);
		activation->dot(store);
		activation.tanh();
		_activation.push_back(activation);
	}
	delete seq_data;
}

void Layer::back_propagation(ccma::algebra::BaseMatrixT<real>* train_seq_data,
					         ccma::algebra::BaseMatrixT<real>* train_seq_label,
						     bool debug = false){
	feed_farward(train_seq_data, debug);

	uint seq_size = train_seq_data->get_rows();

}


}//namespace rnn
}//namespace algorithm
}//namespace ccma
