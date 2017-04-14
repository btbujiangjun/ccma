/*********************************************
* Author: Jun Jiang - jiangjun4@sina.com
* Created: 2017-04-06 17:27
* Last modified: 2017-04-06 17:27
* Filename: DNN.cpp
* Description: 
**********************************************/

#include "DNN.h"
#include <random>
#include <ctime>
#include "algebra/MatrixShuffler.h"

namespace ccma{
namespace algorithm{
namespace nn{

int DNN::add_layer(int neural_size){

    _sizes.push_back(neural_size);
    return ++_num_layers;
}

void DNN::init_networks_weights(){

    init_parameter(&_weights, 0.0, &_biases, 0.0);

    std::default_random_engine engine(time(0));
    //mean & stddev
    std::normal_distribution<real> distribution(0.0, 1.0);

    int len = _weights.size();
    for(int i = 0; i < len; i++){
        for(int j = 0; j < _weights[i]->get_sizes(); j++){
            _weights[i]->set_data(distribution(engine), j, 0);
        }
        for(int j = 0; j < _biases[i]->get_sizes(); j++){
            _biases[i]->set_data(distribution(engine), j, 0);
        }

        _weights[i]->display();
        _biases[i]->display();

    }
}

bool DNN::sgd(ccma::algebra::BaseMatrixT<real>* train_data,
              int epochs,
              real eta,
              int mini_batch_size,
              ccma::algebra::BaseMatrixT<real>* test_data){

    int num_train_data = train_data->get_rows();
    int num_test_data = 0;
    if(test_data != nullptr){
        num_test_data = test_data->get_rows();
    }

    if(_num_layers <= 1 || _sizes[0] != train_data->get_cols() - 1 || (num_test_data > 0 && _sizes[0] != test_data->get_cols() - 1)){
        return false;
    }

    ccma::algebra::MatrixShuffler<real>* shuffler = new ccma::algebra::MatrixShuffler<real>(train_data);

    ccma::algebra::DenseMatrixT<real>* mini_batch = new ccma::algebra::DenseMatrixT<real>(0,train_data->get_cols());

    for(int i = 0; i < epochs; i++){

        shuffler->shuffle();

        for(int j = 0; j < num_train_data; j++){
            ccma::algebra::BaseMatrixT<real>* row_data = train_data->get_row_data(j);
            mini_batch->set_row_data(row_data, j % mini_batch_size);
            delete row_data;

            if(mini_batch->get_rows() == mini_batch_size || j == (num_train_data - 1) ){
                mini_batch_update(mini_batch, eta);
                mini_batch->clear_matrix();
            }
        }

    }

    delete mini_batch;
    delete shuffler;

    return true;
}


bool DNN::mini_batch_update(ccma::algebra::BaseMatrixT<real>* mini_batch, real eta){
    std::vector<ccma::algebra::BaseMatrixT<real>*> batch_weights;
    std::vector<ccma::algebra::BaseMatrixT<real>*> batch_biases;
    init_parameter(&batch_weights, 0.0);
    init_parameter(&batch_biases, 0.0);

    ccma::algebra::LabeledDenseMatrixT<real>* label_data = new ccma::algebra::LabeledDenseMatrixT<real>();
    for(int i = 0; i < mini_batch->get_rows(); i++){

        real* data = new real[_sizes[0]];
        memcpy(data, mini_batch->get_row_data(i)->get_data(), sizeof(real) * _sizes[0]);
        real label = mini_batch->get_row_data(i)->get_data()[_sizes[0]];
        label_data->set_shallow_data(data, &label, 1, _sizes[0]);

        std::vector<ccma::algebra::BaseMatrixT<real>*> train_weight;
        std::vector<ccma::algebra::BaseMatrixT<real>*> train_bias;

        back_propagation(label_data, &train_weight, &train_bias);

        //sum weights & biases
        for(int i = 0; i < _num_layers; i++){
            batch_weights[i]->add(train_weight[i]);
            batch_biases[i]->add(train_bias[i]);
        }

        //clear train_weight & train_bias
    }
    delete label_data;

    //batch update with average grad
    for(int i = 0; i < _num_layers; i++){
        batch_weights[i]->multiply(eta);
        batch_weights[i]->division(mini_batch->get_rows());
        _weights[i]->subtract(batch_weights[i]);

        batch_biases[i]->multiply(eta);
        batch_biases[i]->division(mini_batch->get_rows());
        _biases[i]->subtract(batch_biases[i]);
    }
}

void DNN::back_propagation(const ccma::algebra::LabeledDenseMatrixT<real>* train_data,
                           std::vector<ccma::algebra::BaseMatrixT<real>*>* out_weights,
                           std::vector<ccma::algebra::BaseMatrixT<real>*>* out_biases){

    std::vector<ccma::algebra::BaseMatrixT<real>*> local_weights;
    std::vector<ccma::algebra::BaseMatrixT<real>*> local_biases;
    init_parameter(&local_weights, 0.0, &local_biases, 0.0);

    std::vector<ccma::algebra::BaseMatrixT<real>*> zs;
    std::vector<ccma::algebra::BaseMatrixT<real>*> activations;
    ccma::algebra::BaseMatrixT<real>* activation = train_data->get_data_matrix();

    for(int i = 0; i < _weights.size(); i++){
        activation->multiply(_weights[i]);
        activation->add(_biases[i]);
        zs.push_back(activation->copy_matrix());

        activation->signmoid();
        activations.push_back(activation->copy_matrix());
    }

}
real DNN::cost_derivative(real output_activation, y){
    return output_activation - y;
}

void DNN::sigmoid_derivative(ccma::algebra::BaseMatrixT<real>* z){

}

void DNN::init_parameter(std::vector<ccma::algebra::BaseMatrixT<real>*>* weight_parameter,
                         real weight_init_value
                         std::vector<ccma::algebra::BaseMatrixT<real>*>* biases_parameter,
                         real bias_init_value){

    clear_parameter(weight_parameter);
    clear_parameter(bias_parameter);

    for(int i = 1; i < _num_layers; i++){
        ccma::algebra::BaseMatrixT<real>* layer = new ccma::algebra::DenseMatrixT<real>(_sizes[i], _sizes[i - 1]);
        _weights.push_back(layer);

        ccma::algebra::BaseMatrixT<real>* bias = new ccma::algebra::DenseMatrixT<real>(1, _sizes[i]);
        _biases.push_back(bias);
    }
}

void DNN::clear_parameter(std::vector<ccma::algebra::BaseMatrixT<real>*>* parameters){
    size_t len = parameters->size();
    for(size_t i = 0; i < len; i++){
        delete parameters[i];
    }
    parameters->clear();
}


}//namespace nn
}//namespace algorithm
}//namespace ccma
