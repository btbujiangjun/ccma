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
#include "algebra/BaseMatrix.h"

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
        _bptt_truncate = bptt_truncate;

        _U = new ccma::algebra::DenseRandomMatrixT<real>(hidden_dim, feature_dim, 0, 0.5);
        _V = new ccma::algebra::DenseRandomMatrixT<real>(feature_dim, hidden_dim, 0, 0.5);
        _W = new ccma::algebra::DenseRandomMatrixT<real>(hidden_dim, hidden_dim, 0, 0.5);
    }
    ~RNN(){
        if(_U != nullptr){
            delete _U;
        }
        if(_V != nullptr){
            delete _V;
        }
        if(_W != nullptr){
            delete _W;
        }
    }

private:

    void feed_farward(ccma::algebra::BaseMatrix<real>* train_seq_data, bool debug = false);
    void back_propagation(ccma::algebra::BaseMatrix<real>* train_seq_data,
                          ccma::algebra::BaseMatrix<real>* train_seq_label,
                          bool debug = false);

private:
    uint _feature_dim;
    uint _hidden_dim;

    uint _bptt_truncate;

    ccma::algebra::DenseRandomMatrixT<real>* _U;
    ccma::algebra::DenseRandomMatrixT<real>* _V;
    ccma::algebra::DenseRandomMatrixT<real>* _W;

};//class RNN 
}//namespace rnn
}//namespace algorithm
}//namespace ccma

#endif
