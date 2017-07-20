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
						 ccma::algebra::BaseMatrixT<real>* state,
						 ccma::algebra::BaseMatrixT<real>* activation,
                         bool debug){
	
	uint seq_rows = train_seq_data->get_rows();
	uint seq_cols = train_seq_data->get_cols();
	
	state->reset(0, seq_rows, _hidden_dim);
	activation->reset(0, seq_rows, seq_cols);

	auto state_t = new ccma::algebra::DenseMatrixT<real>();
	auto activation_t = new ccma::algebra::DenseMatrixT<real>();
	auto seq_time_data = new ccma::algebra::DenseMatrixT<real>();
	auto pre_state_t = new ccma::algebra::DenseMatrixT<real>();
	auto pre_weight_t = new ccma::algebra::DenseMatrixT<real>();

	for(uint t = 0; t != seq_rows; t++){
		//s[t] = tanh(U*x[t] + W*s[t-1])
		//o[t] = softmax(V* s[t])
		train_seq_data->get_row_data(t, seq_time_data);
		weight->clone(state_t);
		state_t->dot(seq_time_data->transpose());

		if(t > 0){
			pre_weight->clone(pre_weight_t);
			state->get_row_data(t-1, pre_state_t);

			pre_weight_t->dot(pre_state_t->transpose());
			state_t->add(pre_weight_t);
		}
        state_t->tanh();
		state->set_row_data(t, state_t->transpose());
	
		act_weight->clone(activation_t);
		activation_t->dot(state_t->transpose());

        auto m = new ccma::algebra::DenseMatrixT<real>();
        activation_t->clone(m);

		activation_t->softmax();

		if(activation_t->isnan()){
            uint size = activation_t->get_size();
            uint max_idx = 0;
            real max_value = 0.0;
            for(uint i = 0; i != size; i++){
                if(i == 0 || max_value < activation_t->get_data(i) || std::isnan(activation_t->get_data(i))){
                    max_value = activation_t->get_data(i);
                    max_idx = i;
                }
            }

            uint rows = max_idx / activation_t->get_cols();
            uint cols = (max_idx - rows * activation_t->get_cols());

            printf("isnan[%d][%d][%f][%f][%f]\n", rows, cols, act_weight->get_data(rows, cols),m->get_data(rows, cols), max_value);

            auto max_mat = new ccma::algebra::DenseMatrixT<real>();
            act_weight->get_row_data(rows, max_mat);
            
            printf("m_mat");
            m->display();
            printf("state_t");
            state_t->transpose()->display();
            state_t->transpose();

            delete max_mat;
        }

        delete m;

		activation->set_row_data(t, activation_t->transpose());
	}

	delete state_t;
	delete activation_t;
	delete seq_time_data;
	delete pre_state_t;
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
	//derivate_weight->reset(0, weight->get_rows(), weight->get_cols());
	//derivate_pre_weight->reset(0, pre_weight->get_rows(), pre_weight->get_cols());
	//derivate_act_weight->reset(0, act_weight->get_rows(), act_weight->get_cols());
	
    auto state		     = new ccma::algebra::DenseMatrixT<real>();
	auto derivate_output = new ccma::algebra::DenseMatrixT<real>();

	feed_farward(train_seq_data, weight, pre_weight, act_weight, state, derivate_output, debug);

	derivate_output->subtract(train_seq_label);

	auto derivate_pre_weight_t  = new ccma::algebra::DenseMatrixT<real>();
	auto derivate_output_t      = new ccma::algebra::DenseMatrixT<real>();
	auto derivate_weight_t      = new ccma::algebra::DenseMatrixT<real>();
	auto derivate_weight_t_c    = new ccma::algebra::DenseMatrixT<real>();
	auto derivate_state_t       = new ccma::algebra::DenseMatrixT<real>();
	auto derivate_t             = new ccma::algebra::DenseMatrixT<real>();
	auto train_data_t           = new ccma::algebra::DenseMatrixT<real>();

	uint seq_size = train_seq_data->get_rows();

	for(int t = seq_size - 1; t >= 0 ; t--){
        //update derivate_act_weight
		derivate_output->get_row_data(t, derivate_output_t);
		state->get_row_data(t, derivate_state_t);

		derivate_output_t->transpose()->outer(derivate_state_t->transpose());
		derivate_act_weight->add(derivate_output_t);

        //calc derivate_t
		state->get_row_data(t, derivate_state_t);
		derivate_state_t->transpose();
		derivate_state_t->multiply(derivate_state_t);
		derivate_state_t->multiply(-1);
		derivate_state_t->add(1);

		derivate_output->get_row_data(t, derivate_output_t);
		act_weight->clone(derivate_t);

		derivate_t->transpose()->dot(derivate_output_t->transpose());

		derivate_t->multiply(derivate_state_t);

		//back_propagation steps
		for(uint step = 0; step < _bptt_truncate && (int)step <= t; step++){
			int bptt_step = t - step;

            if(debug){
    			printf("Backpropagation step t=%d bptt step=%d\n", t, bptt_step);
            }

			if(bptt_step > 0){
				state->get_row_data(bptt_step -1, derivate_state_t);
			}else{
				derivate_state_t->reset(0, state->get_cols(), 1);
			}

            //update derivate_pre_weight
			derivate_t->clone(derivate_pre_weight_t);
			derivate_pre_weight_t->outer(derivate_state_t->transpose());
			derivate_pre_weight->add(derivate_pre_weight_t);
			
            //update derivate_weight
			train_seq_data->get_row_data(bptt_step, train_data_t);
            derivate_weight->clone(derivate_weight_t);

			derivate_weight_t->dot(train_data_t->transpose());
			derivate_weight_t->add(derivate_t);

            uint idx = train_data_t->argmax(0, 1);
            derivate_weight->get_col_data(idx, derivate_weight_t_c);
            derivate_weight_t_c->add(derivate_weight_t);
            derivate_weight->set_col_data(idx, derivate_weight_t_c);

			//update delta
			if(bptt_step > 0){
				derivate_state_t->multiply(derivate_state_t);
				derivate_state_t->multiply(-1);
				derivate_state_t->add(1);

				pre_weight->clone(derivate_pre_weight_t);
				derivate_pre_weight_t->transpose()->dot(derivate_t);
				derivate_pre_weight_t->multiply(derivate_state_t);
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
	delete derivate_state_t;
	delete train_data_t;
}


}//namespace rnn
}//namespace algorithm
}//namespace ccma
