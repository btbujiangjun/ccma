/***********************************************
 * Author: Jun Jiang - jiangjun4@sina.com
 * Create: 2017-06-20 17:05
 * Last modified : 2017-06-20 17:05
 * Filename      : Layer.cpp
 * Description   : RNN network Layer 
 **********************************************/
#include "algebra/BaseMatrix.h"
#include "algorithm/rnn/Layer.h"

namespace ccma{
namespace algorithm{
namespace rnn{

void Layer::initialize(ccma::algebra::BaseMatrixT<real>* train_seq_data){
    if(_store != nullptr){
        delete _store;
    }
    _store = new ccma::algebra::DenseMatrixT<real>(train_seq_data->get_rows(), _hidden_dim);

    if(_activation != nullptr){
        delete _activation;
    }
    _activation = new ccma::algebra::DenseMatrixT<real>(train_seq_data->get_rows(), train_seq_data->get_cols());
}

void Layer::feed_farward(ccma::algebra::BaseMatrixT<real>* train_seq_data, bool debug){

	initialize(train_seq_data);
	uint seq_size = train_seq_data->get_rows();

	auto store = new ccma::algebra::DenseMatrixT<real>();
	auto activation = new ccma::algebra::DenseMatrixT<real>();
    auto seq_time_data = new ccma::algebra::DenseMatrixT<real>();
    auto pre_store = new ccma::algebra::DenseMatrixT<real>();
    auto pre_weight = new ccma::algebra::DenseMatrixT<real>();

	for(uint t = 0; t != seq_size; t++){
        //s[t] = tanh(U*x[t] + W*s[t-1])
        //o[t] = softmax(V* s[t])
		train_seq_data->get_row_data(t, seq_time_data);
		_weight->clone(store);
        seq_time_data->transpose();

		store->dot(seq_time_data);
        store->transpose();

		if(t > 0){
			_pre_weight->clone(pre_weight);
            _store->get_row_data(t-1, pre_store);
            pre_store->transpose();

			pre_weight->dot(pre_store);
            pre_weight->transpose();
            store->add(pre_weight);
		}

        _store->set_row_data(store, t);
        
		_act_weight->clone(activation);

		store->transpose();
        activation->dot(store);
		activation->tanh();
		activation->transpose();
		_activation->set_row_data(activation, t);
	}

	delete store;
    delete activation;
    delete seq_time_data;
    delete pre_store;
    delete pre_weight;
}

void Layer::back_propagation(ccma::algebra::BaseMatrixT<real>* train_seq_data,
					         ccma::algebra::BaseMatrixT<real>* train_seq_label,
						     bool debug){
	feed_farward(train_seq_data, debug);

    auto derivate_weight = new ccma::algebra::DenseMatrixT<real>(_weight->get_rows(), _weight->get_cols());
    auto derivate_pre_weight = new ccma::algebra::DenseMatrixT<real>(_pre_weight->get_rows(), _pre_weight->get_cols());
    auto derivate_act_weight = new ccma::algebra::DenseMatrixT<real>(_act_weight->get_rows(), _act_weight->get_cols());

    auto derivate_output = new ccma::algebra::DenseMatrixT<real>();
    _activation->clone(derivate_output);
    derivate_output->subtract(train_seq_label);

    auto derivate_output_t = new ccma::algebra::DenseMatrixT<real>();
    auto derivate_store_t = new ccma::algebra::DenseMatrixT<real>();
    auto derivate_t = new ccma::algebra::DenseMatrixT<real>();

    auto derivate_pre_weight_t = new ccma::algebra::DenseMatrixT<real>();
    auto derivate_weight_t = new ccma::algebra::DenseMatrixT<real>();
    auto train_data_t = new ccma::algebra::DenseMatrixT<real>();

	uint seq_size = train_seq_data->get_rows();
    for(int t = seq_size - 1; t >= 0 ; t--){
        derivate_output->get_row_data(t, derivate_output_t);
        _store->get_row_data(t, derivate_store_t);
        derivate_store_t->transpose();

        derivate_output_t->outer(derivate_store_t);
        derivate_act_weight->add(derivate_output_t);

        _store->get_row_data(t, derivate_store_t);
        derivate_store_t->multiply(derivate_store_t);
        derivate_store_t->multiply(-1);
        derivate_store_t->add(1);

        derivate_output->get_row_data(t, derivate_output_t);
		derivate_output_t->transpose();

		_act_weight->clone(derivate_t);
		derivate_t->transpose();

        derivate_t->dot(derivate_output_t);
		derivate_t->transpose();

		derivate_t->multiply(derivate_store_t);

        //back_propagation steps
        for(int step = 0; step < _bptt_truncate && step <= t; step++){
            int bptt_step = t - step;
            printf("Backpropagation step t=%d bptt step=%d\n", t, bptt_step);
            if(bptt_step > 0){
                _store->get_row_data(bptt_step -1, derivate_store_t);
            }else{
                real* data = new real[_store->get_cols()];
                memset(data, 0, sizeof(real)*_store->get_cols());
                derivate_store_t->set_shallow_data(data, 1, _store->get_cols());
            }
            derivate_t->clone(derivate_pre_weight_t);
            derivate_pre_weight_t->outer(derivate_store_t);
            derivate_pre_weight->add(derivate_pre_weight_t);
           



            //for(uint i = 0; i < derivate_t->get_cols(); i++){
				//todo
                //derivate_weight->set_data(i, derivate_weight->get_data(i,) + derivate_t->get_data(0, i));
            //}
            
            if(bptt_step > 0){
                derivate_store_t->multiply(derivate_store_t);
                derivate_store_t->multiply(-1);
                derivate_store_t->add(1);

                derivate_pre_weight_t->transpose();
                derivate_pre_weight_t->dot(derivate_t);
                derivate_pre_weight_t->multiply(derivate_store_t);
                derivate_pre_weight_t->clone(derivate_t);
            }
        }
    }
    delete derivate_output;
    delete derivate_output_t;
    delete derivate_t;
    delete derivate_pre_weight_t;
    delete derivate_weight_t;
    delete derivate_store_t;
	
	delete derivate_weight;
	delete derivate_pre_weight;
	delete derivate_act_weight;
}


}//namespace rnn
}//namespace algorithm
}//namespace ccma
