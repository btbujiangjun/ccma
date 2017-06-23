/***********************************************
 * Author: Jun Jiang - jiangjun4@sina.com
 * Create: 2017-06-20 16:23
 * Last modified : 2017-06-20 16:23
 * Filename      : Layer.h
 * Description   : RNN network Layer 
 **********************************************/

#ifndef _CCMA_ALGORITHM_RNN_LAYER_H
#define _CCMA_ALGORITHM_RNN_LAYER_H_

#include <vector>
#include "algebra/BaseMatrix.h"

namespace ccma{
namespace algorithm{
namespace rnn{

class Layer{
public:
	Layer(uint hidden_dim,
          uint bptt_truncate,
          ccma::algebra::BaseMatrixT<real>* weight,
          ccma::algebra::BaseMatrixT<real>* pre_weight,
          ccma::algebra::BaseMatrixT<real>* act_weight){
	    _hidden_dim = hidden_dim;
        _bptt_truncate = bptt_truncate;
		_weight = weight;
		_pre_weight = pre_weight;
		_act_weight = act_weight;
	}

	~Layer(){
		if(_store != nullptr){
			delete _store;
		}
		if(_activation != nullptr){
			delete _activation;
		}
	}

	void feed_farward(ccma::algebra::BaseMatrixT<real>* train_seq_data, bool debug = false);
	void back_propagation(ccma::algebra::BaseMatrixT<real>* train_seq_data,
						  ccma::algebra::BaseMatrixT<real>* train_seq_label,
						  bool debug = false);

private:
	void initialize(ccma::algebra::BaseMatrixT<real>* train_seq_data);

private:
	uint _hidden_dim;
    uint _bptt_truncate;
	ccma::algebra::BaseMatrixT<real>* _weight;
	ccma::algebra::BaseMatrixT<real>* _pre_weight;
	ccma::algebra::BaseMatrixT<real>* _act_weight;

	ccma::algebra::BaseMatrixT<real>* _store;
	ccma::algebra::BaseMatrixT<real>* _activation;
};//class Layer

}//namespace rnn
}//namespace algorithm
}//namespace ccma

#endif
