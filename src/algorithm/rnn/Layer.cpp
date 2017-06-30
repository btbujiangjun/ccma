/***********************************************
 * Author: Jun Jiang - jiangjun4@sina.com
 * Create: 2017-06-20 17:05
 * Last modified : 2017-06-20 17:05
 * Filename	  : Layer.cpp
 * Description   : RNN network Layer 
 **********************************************/
#include "algebra/BaseMatrix.h"
#include "algorithm/rnn/Layer.h"

namespace ccma{
namespace algorithm{
namespace rnn{

void Layer::feed_farward(ccma::algebra::BaseMatrixT<real>* train_seq_data,
						 ccma::algebra::BaseMatrixT<real>* weight,
						 ccma::algebra::BaseMatrixT<real>* pre_weight,
						 ccma::algebra::BaseMatrixT<real>* act_weight,
						 ccma::algebra::BaseMatrixT<real>* store,
						 ccma::algebra::BaseMatrixT<real>* activation,
                         bool debug){
	
	uint seq_rows = train_seq_data->get_rows();
	uint seq_cols = train_seq_data->get_cols();
	
	store->reset(0, _hidden_dim, seq_rows);
	activation->reset(0, seq_cols, seq_rows);

	auto store_t = new ccma::algebra::DenseMatrixT<real>();
	auto activation_t = new ccma::algebra::DenseMatrixT<real>();
	auto seq_time_data = new ccma::algebra::DenseMatrixT<real>();
	auto pre_store_t = new ccma::algebra::DenseMatrixT<real>();
	auto pre_weight_t = new ccma::algebra::DenseMatrixT<real>();

	for(uint t = 0; t != seq_rows; t++){
		//s[t] = tanh(U*x[t] + W*s[t-1])
		//o[t] = softmax(V* s[t])
		train_seq_data->get_row_data(t, seq_time_data);
		weight->clone(store_t);
		store_t->dot(seq_time_data->transpose());

		if(t > 0){
			pre_weight->clone(pre_weight_t);
			store->get_col_data(t-1, pre_store_t);

			pre_weight_t->dot(pre_store_t);
			store_t->add(pre_weight_t);
		}
        store_t->tanh();
		store->set_col_data(t, store_t);
	
		act_weight->clone(activation_t);
		activation_t->dot(store_t);
		activation_t->softmax();
		activation->set_col_data(t, activation_t);
	}

	delete store_t;
	delete activation_t;
	delete seq_time_data;
	delete pre_store_t;
	delete pre_weight_t;
}

void Layer::back_propagation(ccma::algebra::BaseMatrixT<real>* train_seq_data,
							 ccma::algebra::BaseMatrixT<real>* train_seq_label,
							 ccma::algebra::BaseMatrixT<real>* weight,
							 ccma::algebra::BaseMatrixT<real>* pre_weight,
							 ccma::algebra::BaseMatrixT<real>* act_weight,
							 ccma::algebra::BaseMatrixT<real>* derivate_weight,
							 ccma::algebra::BaseMatrixT<real>* derivate_pre_weight,
							 ccma::algebra::BaseMatrixT<real>* derivate_act_weight,
                             bool debug){

	auto store		= new ccma::algebra::DenseMatrixT<real>();
	auto activation = new ccma::algebra::DenseMatrixT<real>();

	derivate_weight->reset(0, weight->get_rows(), weight->get_cols());
	derivate_pre_weight->reset(0, pre_weight->get_rows(), pre_weight->get_cols());
	derivate_act_weight->reset(0, act_weight->get_rows(), act_weight->get_cols());

	feed_farward(train_seq_data, weight, pre_weight, act_weight, store, activation, debug);

	auto derivate_output = new ccma::algebra::DenseMatrixT<real>();
	activation->clone(derivate_output);

	derivate_output->subtract(train_seq_label->transpose());

	auto derivate_pre_weight_t  = new ccma::algebra::DenseMatrixT<real>();
	auto derivate_output_t      = new ccma::algebra::DenseMatrixT<real>();
	auto derivate_weight_t      = new ccma::algebra::DenseMatrixT<real>();
	auto derivate_weight_t_c    = new ccma::algebra::DenseMatrixT<real>();
	auto derivate_store_t       = new ccma::algebra::DenseMatrixT<real>();
	auto derivate_t             = new ccma::algebra::DenseMatrixT<real>();
	auto train_data_t           = new ccma::algebra::DenseMatrixT<real>();

	uint seq_size = train_seq_data->get_rows();

	for(int t = seq_size - 1; t >= 0 ; t--){
        //update derivate_act_weight
		derivate_output->get_col_data(t, derivate_output_t);
		store->get_col_data(t, derivate_store_t);

		derivate_output_t->outer(derivate_store_t);
		derivate_act_weight->add(derivate_output_t);

        //calc derivate_t
		store->get_col_data(t, derivate_store_t);
		derivate_store_t->multiply(derivate_store_t);
		derivate_store_t->multiply(-1);
		derivate_store_t->add(1);

		activation->get_col_data(t, derivate_output_t);
		act_weight->clone(derivate_t);

		derivate_t->transpose()->dot(derivate_output_t);

		derivate_t->multiply(derivate_store_t);

		//back_propagation steps
		for(uint step = 0; step < _bptt_truncate && (int)step <= t; step++){
			int bptt_step = t - step;

            if(debug){
    			printf("Backpropagation step t=%d bptt step=%d\n", t, bptt_step);
            }

			if(bptt_step > 0){
				store->get_col_data(bptt_step -1, derivate_store_t);
			}else{
				derivate_store_t->reset(0, store->get_rows(), 1);
			}

            //update derivate_pre_weight
			derivate_t->clone(derivate_pre_weight_t);
			derivate_pre_weight_t->outer(derivate_store_t);
			derivate_pre_weight->add(derivate_pre_weight_t);

            //update derivate_weight
			train_seq_data->get_row_data(bptt_step, train_data_t);
            derivate_weight->clone(derivate_weight_t);

			derivate_weight_t->dot(train_data_t->transpose());
			derivate_weight_t->add(derivate_t);

            uint idx = train_data_t->argmax(bptt_step, 0);
            derivate_weight->get_col_data(idx, derivate_weight_t_c);
            derivate_weight_t_c->add(derivate_weight_t);
            derivate_weight->set_col_data(idx, derivate_weight_t_c);

			//update delta
			if(bptt_step > 0){
				derivate_store_t->multiply(derivate_store_t);
				derivate_store_t->multiply(-1);
				derivate_store_t->add(1);

				derivate_pre_weight_t->transpose()->dot(derivate_t);
				derivate_pre_weight_t->multiply(derivate_store_t);
				derivate_pre_weight_t->clone(derivate_t);
			}
		}
	}
	delete derivate_output;
	delete derivate_output_t;
    delete derivate_weight_t;
    delete derivate_weight_t_c;
	delete derivate_t;
	delete derivate_pre_weight_t;
	delete derivate_store_t;
	delete train_data_t;
}


}//namespace rnn
}//namespace algorithm
}//namespace ccma
