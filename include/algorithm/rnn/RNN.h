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

        _U = new ccma::algebra::DenseRandomMatrixT<real>(hidden_dim, feature_dim, 0, 0.5);
        _W = new ccma::algebra::DenseRandomMatrixT<real>(hidden_dim, hidden_dim, 0, 0.5);
        _V = new ccma::algebra::DenseRandomMatrixT<real>(feature_dim, hidden_dim, 0, 0.5);

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
    uint _feature_dim;
    uint _hidden_dim;

    ccma::algebra::DenseRandomMatrixT<real>* _U;
    ccma::algebra::DenseRandomMatrixT<real>* _W;
    ccma::algebra::DenseRandomMatrixT<real>* _V;

    Layer* _layer;
};//class RNN 

}//namespace rnn
}//namespace algorithm
}//namespace ccma

#endif
