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
          uint bptt_truncate,
          ccma::algebra::BaseMatrixT<real>* weight,
          ccma::algebra::BaseMatrixT<real>* pre_weight,
          ccma::algebra::BaseMatrixT<real>* act_weight){

	    _hidden_dim = hidden_dim;
        _bptt_truncate = bptt_truncate;

//		_weight = weight;
//		_pre_weight = pre_weight;
//		_act_weight = act_weight;

//        _store = new ccma::algebra::DenseMatrixT<real>();
//        _activation = new ccma::algebra::DenseMatrixT<real>();
	
//		_derivate_weight = new ccma::algebra::DenseMatrixT<real>(_weight->get_rows(), _weight->get_cols());
//		_derivate_pre_weight = new ccma::algebra::DenseMatrixT<real>(_pre_weight->get_rows(), _pre_weight->get_cols());
//		_derivate_act_weight = new ccma::algebra::DenseMatrixT<real>(_act_weight->get_rows(), _act_weight->get_cols());
	}

	~Layer() = default;

	void feed_farward(ccma::algebra::BaseMatrixT<real>* train_seq_data,
                      ccma::algebra::BaseMatrixT<real>* weight,
                      ccma::algebra::BaseMatrixT<real>* pre_weight,
                      ccma::algebra::BaseMatrixT<real>* act_weight,
                      ccma::algebra::BaseMatrixT<real>* store,
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

//	ccma::algebra::BaseMatrixT<real>* _store;
//	ccma::algebra::BaseMatrixT<real>* _activation;

//	ccma::algebra::BaseMatrixT<real>* _weight;
//	ccma::algebra::BaseMatrixT<real>* _pre_weight;
//	ccma::algebra::BaseMatrixT<real>* _act_weight;
	
//	ccma::algebra::BaseMatrixT<real>* _derivate_weight;
//	ccma::algebra::BaseMatrixT<real>* _derivate_pre_weight;
//	ccma::algebra::BaseMatrixT<real>* _derivate_act_weight;
	
};//class Layer

}//namespace rnn
}//namespace algorithm
}//namespace ccma

#endif
