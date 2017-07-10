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
#include <thread>
#include "Cost.h"
#include "algebra/BaseMatrix.h"
#include "utils/MatrixHelper.h"
#include "utils/ModelLoader.h"

namespace ccma{
namespace algorithm{
namespace nn{


class DNN{
public:
    DNN(const std::string& path = ""){
         _num_layers = 0;
         _path = path;
         _cost = new CrossEntropyCost();
    }

    ~DNN(){
        _sizes.clear();

        clear_parameter(&_weights);
        _weights.clear();

        clear_parameter(&_biases);
        _biases.clear();

        delete _cost;
    }

    int add_layer(int neural_size);

    void init_networks_weights();

    bool sgd(ccma::algebra::BaseMatrixT<real>* train_data,
             ccma::algebra::BaseMatrixT<real>* train_label,
             uint epochs,
             real eta,
             real lamda = 0.0,
             uint mini_batch_size = 1,
             ccma::algebra::BaseMatrixT<real>* test_data = nullptr,
             ccma::algebra::BaseMatrixT<real>* test_label = nullptr);


    void feedforward(ccma::algebra::BaseMatrixT<real>* mat);

    int evaluate(ccma::algebra::BaseMatrixT<real>* test_data, ccma::algebra::BaseMatrixT<real>* test_label);

    bool load_model(const std::string& path);
    bool write_model(const std::string& path);

private:
    void mini_batch_update(ccma::algebra::BaseMatrixT<real>* mini_batch_data,
                           ccma::algebra::BaseMatrixT<real>* mini_batch_label,
                           real eta,
                           real lamda,
                           uint num_train_data);

    void back_propagation(ccma::algebra::BaseMatrixT<real>* train_data,
                          ccma::algebra::BaseMatrixT<real>* train_label,
                          std::vector<ccma::algebra::BaseMatrixT<real>*>* batch_weights,
                          std::vector<ccma::algebra::BaseMatrixT<real>*>* batch_biases);

    void init_parameter(std::vector<ccma::algebra::BaseMatrixT<real>*>* weight_parameter,
                        std::vector<ccma::algebra::BaseMatrixT<real>*>* biases_parameter);

    void clear_parameter(std::vector<ccma::algebra::BaseMatrixT<real>*>* parameters);

private:
    uint _num_layers;
    std::vector<uint> _sizes;
    std::string _path;
    Cost* _cost;

    const uint _num_hardware_concurrency = std::thread::hardware_concurrency() == 0 ? 1 : std::thread::hardware_concurrency();

    std::vector<ccma::algebra::BaseMatrixT<real>*> _weights;
    std::vector<ccma::algebra::BaseMatrixT<real>*> _biases;

    ccma::utils::ModelLoader loader;
    ccma::utils::MatrixHelper helper;
};//class DNN

}//namespace nn
}//namespace algorithm
}//namespace ccma
#endif
