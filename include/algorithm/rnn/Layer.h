/***********************************************
 * Author: Jun Jiang - jiangjun4@sina.com
 * Create: 2017-06-20 16:23
 * Last modified : 2017-06-20 16:23
 * Filename      : Layer.h
 * Description   : RNN network Layer 
 **********************************************/

#ifndef _CCMA_ALGORITHM_RNN_LAYER_H_
#define _CCMA_ALGORITHM_RNN_LAYER_H_

#include <vector>
#include "algebra/BaseMatrix.h"

namespace ccma{
namespace algorithm{
namespace rnn{

class Layer{
public:
	Layer(uint hidden_dim,
          uint bptt_truncate){
	    _hidden_dim = hidden_dim;
        _bptt_truncate = bptt_truncate;
	}

	~Layer() = default;

	void feed_farward(ccma::algebra::BaseMatrixT<real>* train_seq_data,
                      ccma::algebra::BaseMatrixT<real>* weight,
                      ccma::algebra::BaseMatrixT<real>* pre_weight,
                      ccma::algebra::BaseMatrixT<real>* act_weight,
                      ccma::algebra::BaseMatrixT<real>* state,
                      ccma::algebra::BaseMatrixT<real>* activation,
                      bool debug = false);

	void back_propagation(ccma::algebra::BaseMatrixT<real>* train_seq_data,
						  ccma::algebra::BaseMatrixT<real>* train_seq_label,
                          ccma::algebra::BaseMatrixT<real>* weight,
                          ccma::algebra::BaseMatrixT<real>* pre_weight,
                          ccma::algebra::BaseMatrixT<real>* act_weight,
                          ccma::algebra::BaseMatrixT<real>* derivate_weight,
                          ccma::algebra::BaseMatrixT<real>* derivate_pre_weight,
                          ccma::algebra::BaseMatrixT<real>* derivate_act_weight,
                          bool debug = false);

private:
	uint _hidden_dim;
    uint _bptt_truncate;
};//class Layer

}//namespace rnn
}//namespace algorithm
}//namespace ccma

#endif
