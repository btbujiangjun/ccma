/*********************************************
* Author: Jun Jiang - jiangjun4@sina.com
* Created: 2017-04-06 16:27
* Last modified: 2017-04-06 16:27
* Filename: nn.h
* Description: deep neural networks, 
* feedforward and backpropagation
**********************************************/

#ifndef _CCMA_ALGORITHM_NN_DNN_H_
#define _CCMA_ALGORITHM_NN_DNN_H_

#include <vector>
#include "algebra/BaseMatrix.h"

namespace ccma{
namespace algorithm{
namespace nn{


class DNN{
public:
    DNN(){}

    ~DNN(){
        _sizes.clear();

        for(int i = 0; i < _num_layers; i++){
            delete _weights[i];
        }
        _weights.clear();

        for(int i = 0; i < _num_layers  - 1; i++){
            delete _biases[i];
        }
        _biases.clear();
    }

    int add_layer(int neural_size);

    void init_networks();

    bool sgd(ccma::algebra::BaseMatrixT<real>* train_data,
            int epochs,
            real eta,
            int mini_batch_size = 1,
            ccma::algebra::BaseMatrixT<real>* test_data = nullptr);

private:
    bool mini_batch_update(ccma::algebra::BaseMatrixT<real>* mini_batch, real eta);

    void back_propagation(const ccma::algebra::LabeledDenseMatrixT<real>* train_data,
                          std::vector<ccma::algebra::DenseColMatrixT<real>*>* out_weights,
                          std::vector<ccma::algebra::DenseColMatrixT<real>*>* out_biases);


    void init_parameter(std::vector<ccma::algebra::DenseColMatrixT<real>*>* out_parameters,real init_value);

private:
    int _num_layers;

    std::vector<uint> _sizes;
    std::vector<ccma::algebra::DenseColMatrixT<real>*> _weights;
    std::vector<ccma::algebra::DenseColMatrixT<real>*> _biases;
};//class DNN

}//namespace nn
}//namespace algorithm
}//namespace ccma
#endif
