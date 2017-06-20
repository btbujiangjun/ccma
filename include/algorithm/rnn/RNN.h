/***********************************************
 * Author: Jun Jiang - jiangjun4@sina.com
 * Create: 2017-06-20 16:08
 * Last modified : 2017-06-20 16:08
 * Filename      : RNN.h
 * Description   : Recurrent Neural Network 
 **********************************************/

#ifndef _CCMA_ALGORITHM_RNN_RNN_H_
#define _CCMA_ALGORITHM_RNN_RNN_H_

#include "algebra/BaseMatrix.h"
#include "algorithm/rnn/Layer.h"

namespace ccma{
namespace algorithm{
namespace rnn{
class RNN{
public:
    RNN(){}
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
    ccma::algebra::DenseRandomMatrixT<real>* _U;
    ccma::algebra::DenseRandomMatrixT<real>* _V;
    ccma::algebra::DenseRandomMatrixT<real>* _W;

};//class RNN 
}//namespace rnn
}//namespace algorithm
}//namespace ccma

#endif
