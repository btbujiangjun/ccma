/*********************************************
* Author: Jun Jiang - jiangjun4@sina.com
* Created: 2017-05-31 17:02
* Last modified: 2017-05-31 17:02
* Filename: CNN.h
* Description: Convolution Neural Network 
**********************************************/

#ifndef _CCMA_ALGORITHM_CNN_CNN_H_
#define _CCMA_ALGORITHM_CNN_CNN_H_

#include "algebra/BaseMatrix.h"
#include "algorithm/cnn/Layer.h"

namespace ccma{
namespace algorithm{
namespace cnn{
class CNN{
public:
    CNN(){}
    ~CNN(){
        for(auto layer : _layers){
            delete layer;
        }
        _layers.clear();
    }

    bool add_layer(Layer* layer);
    void train(ccma::algebra::BaseMatrixT<real>* train_data,
               ccma::algebra::BaseMatrixT<real>* train_label,
               uint epoch = 1,
               ccma::algebra::BaseMatrixT<real>* test_data = nullptr,
               ccma::algebra::BaseMatrixT<real>* test_label = nullptr);
    void feedforward(ccma::algebra::BaseMatrixT<real>* mat);

private:
    std::vector<Layer*> _layers;

};//class CNN

}//namespace cnn
}//namespace algorithm
}//namespace ccma

#endif
