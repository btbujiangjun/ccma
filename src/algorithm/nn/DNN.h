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
#include "utils/MatrixHelper.h"

namespace ccma{
namespace algorithm{
namespace nn{


class DNN{
public:
    DNN(){
         _num_layers = 0;
    }

    ~DNN(){
        _sizes.clear();

        clear_parameter(&_weights);
        _weights.clear();

        clear_parameter(&_biases);
        _biases.clear();
    }

    int add_layer(int neural_size);

    void init_networks_weights();

    bool sgd(ccma::algebra::LabeledDenseMatrixT<real>* train_data,
            int epochs,
            real eta,
            int mini_batch_size = 1,
            ccma::algebra::LabeledDenseMatrixT<real>* test_data = nullptr);


    void feedforward(ccma::algebra::BaseMatrixT<real>* mat);

    int evaluate(ccma::algebra::LabeledDenseMatrixT<real>* test_data);

private:
    bool mini_batch_update(ccma::algebra::LabeledDenseMatrixT<real>* mini_batch, real eta);

    void back_propagation(ccma::algebra::LabeledDenseMatrixT<real>* train_data,
                          std::vector<ccma::algebra::BaseMatrixT<real>*>* out_weights,
                          std::vector<ccma::algebra::BaseMatrixT<real>*>* out_biases);

    void init_parameter(std::vector<ccma::algebra::BaseMatrixT<real>*>* weight_parameter,
                        real weight_init_value,
                        std::vector<ccma::algebra::BaseMatrixT<real>*>* biases_parameter,
                        real bias_init_value);

    void clear_parameter(std::vector<ccma::algebra::BaseMatrixT<real>*>* parameters);

    void cost_derivative(ccma::algebra::BaseMatrixT<real>* output_activation,real y);

    void sigmoid_derivative(ccma::algebra::BaseMatrixT<real>* z);
private:
    int _num_layers;

    std::vector<uint> _sizes;
    std::vector<ccma::algebra::BaseMatrixT<real>*> _weights;
    std::vector<ccma::algebra::BaseMatrixT<real>*> _biases;

    ccma::utils::MatrixHelper helper;
};//class DNN

}//namespace nn
}//namespace algorithm
}//namespace ccma
#endif
