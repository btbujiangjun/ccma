/***********************************************
 * Author: Jun Jiang - jiangjun4@sina.com
 * Create: 2017-06-20 16:08
 * Last modified : 2017-06-20 16:08
 * Filename      : RNN.h
 * Description   : Recurrent Neural Network 
 **********************************************/

#ifndef _CCMA_ALGORITHM_RNN_RNN_H_
#define _CCMA_ALGORITHM_RNN_RNN_H_

#include <vector>
#include "algorithm/rnn/Layer.h"

namespace ccma{
namespace algorithm{
namespace rnn{

class RNN{
public:
    RNN(uint feature_dim,
        uint hidden_dim,
        uint bptt_truncate = 4){

        _feature_dim = feature_dim;
        _hidden_dim = hidden_dim;

        _U = new ccma::algebra::DenseRandomMatrixT<real>(hidden_dim, feature_dim, 0, 1, -std::sqrt(1.0/feature_dim), std::sqrt(1.0/feature_dim));
        _W = new ccma::algebra::DenseRandomMatrixT<real>(hidden_dim, hidden_dim, 0, 1, -std::sqrt(1.0/hidden_dim), std::sqrt(1.0/hidden_dim));
        _V = new ccma::algebra::DenseRandomMatrixT<real>(feature_dim, hidden_dim, 0, 1, -std::sqrt(1.0/hidden_dim), std::sqrt(1.0/hidden_dim));

        _layer = new ccma::algorithm::rnn::Layer(hidden_dim, bptt_truncate, _U, _W, _V);
    }
    ~RNN(){
        if(_U != nullptr){
            delete _U;
        }
        if(_W != nullptr){
            delete _W;
        }
        if(_V != nullptr){
            delete _V;
        }
        delete _layer;
    }

    void sgd(std::vector<ccma::algebra::BaseMatrixT<real>*>* train_seq_data,
             std::vector<ccma::algebra::BaseMatrixT<real>*>* train_seq_label, 
             uint epoch = 5, 
             real alpha = 0.1);

private:
	void sgd_step(ccma::algebra::BaseMatrixT<real>* train_seq_data,
              	  ccma::algebra::BaseMatrixT<real>* train_seq_label, 
			  	  real alpha,
                  bool debug,
				  int j);

    real loss(std::vector<ccma::algebra::BaseMatrixT<real>*>* train_seq_data,
              std::vector<ccma::algebra::BaseMatrixT<real>*>* train_seq_label);

    real total_loss(std::vector<ccma::algebra::BaseMatrixT<real>*>* train_seq_data,
                    std::vector<ccma::algebra::BaseMatrixT<real>*>* train_seq_label);
    
	bool check_data(std::vector<ccma::algebra::BaseMatrixT<real>*>* train_seq_data,
             		std::vector<ccma::algebra::BaseMatrixT<real>*>* train_seq_label); 
private:
    uint _feature_dim;
    uint _hidden_dim;

    ccma::algebra::BaseMatrixT<real>* _U;
    ccma::algebra::BaseMatrixT<real>* _W;
    ccma::algebra::BaseMatrixT<real>* _V;

    Layer* _layer;
};//class RNN 

}//namespace rnn
}//namespace algorithm
}//namespace ccma

#endif
