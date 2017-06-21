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

namespace ccma{
namespace algorithm{
namespace rnn{

class Layer{
public:
	Layer(ccma::algebra::BaseMatrixT<real>* weight,
		  ccma::algebra::BaseMatrixT<real>* pre_weight,
		  ccma::algebra::BaseMatrixT<real>* act_weight){
		  _weight = weight;
		  _pre_weight = pre_weight;
		  _act_weight = act_weight;
	}

	~Layer(){
		initialize();
	}

	void feed_farward(ccma::algebra::BaseMatrixT<real>* train_seq_data, bool debug = false);
	void back_propagation(ccma::algebra::BaseMatrixT<real>* train_seq_data,
						  ccma::algebra::BaseMatrixT<real>* train_seq_label,
						  bool debug = false);

private:
	void initialize();
private:

	std::vector<ccma::algebra::BaseMatrixT<real>*> _store;
	std::vector<ccma::algebra::BaseMatrixT<real>*> _activation;

	ccma::algebra::BaseMatrixT<real>* _weight;
	ccma::algebra::BaseMatrixT<real>* _pre_weight;
	ccma::algebra::BaseMatrixT<real>* _act_weight;
};//class Layer

}//namespace rnn
}//namespace algorithm
}//namespace ccma

#endif
