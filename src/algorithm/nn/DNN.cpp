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
#include "utils/Shuffler.h"

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
            _weights[i]->set_data(distribution(engine), j);
        }
        for(int j = 0; j < _biases[i]->get_sizes(); j++){
            _biases[i]->set_data(distribution(engine), j);
        }

        _weights[i]->display();
        _biases[i]->display();

    }
}

bool DNN::sgd(ccma::algebra::BaseMatrixT<real>* train_data,
              ccma::algebra::BaseMatrixT<real>* train_label,
              int epochs,
              real eta,
              int mini_batch_size,
              ccma::algebra::BaseMatrixT<real>* test_data,
              ccma::algebra::BaseMatrixT<real>* test_label){

    int num_train_data = train_data->get_rows();
    int num_test_data = 0;
    if(test_data != nullptr){
        num_test_data = test_data->get_rows();
    }

    //check nn structure and data dims
    if(_num_layers <= 1 || _sizes[0] != train_data->get_cols() 
            || (num_test_data > 0 && _sizes[0] != test_data->get_cols())){
        return false;
    }

    auto shuffler           = new ccma::utils::Shuffler(train_data->get_rows());

    auto mini_batch_data    = new ccma::algebra::DenseMatrixT<real>();
    auto mini_batch_label   = new ccma::algebra::DenseMatrixT<real>();

    for(int i = 0; i < epochs; i++){

        shuffler->shuffle();

        for(int j = 0; j < num_train_data; j++){

            auto row_data = train_data->get_row_data(shuffler->get_row(j));
            mini_batch_data->set_row_data(row_data, j % mini_batch_size);
            delete row_data;

            auto row_label = train_label->get_row_data(shuffler->get_row(j));
            mini_batch_label->set_row_data(row_label, j % mini_batch_size);
            delete row_label;

            if(mini_batch_data->get_rows() == mini_batch_size || j == (num_train_data - 1) ){
                mini_batch_update(mini_batch_data, mini_batch_label, eta);

                mini_batch_data->clear_matrix();
                mini_batch_label->clear_matrix();
            }
        }

        if(num_test_data > 0){
            int num_predict = evaluate(test_data, test_label);
            printf("Epoch %d: %d / %d\n", i, num_predict, num_test_data);
        }

    }

    delete mini_batch_data;
    delete mini_batch_label;
    delete shuffler;

    return true;
}

void DNN::feedforward(ccma::algebra::BaseMatrixT<real>* mat){
    for(int i = 0; i < _weights.size(); i++){
        if(!mat->product(_weights[i])){
            printf("Matrix Dim Error: weights[%d][%d:%d]-mat[%d:%d]\n", i, _weights[i]->get_rows(), _weights[i]->get_cols(), mat->get_rows(), mat->get_cols());
        }
        mat->add(_biases[i]);
    }
}

int DNN::evaluate(ccma::algebra::BaseMatrixT<real>* test_data, ccma::algebra::BaseMatrixT<real>* test_label){

    ccma::algebra::BaseMatrixT<real>* predict = test_data->clone();

    feedforward(predict);

    int num = 0;
    if(predict->get_rows() == test_label->get_rows() && predict->get_cols() == test_label->get_cols()){
        for(int i = 0; i < predict->get_sizes(); i++){
            if(predict->get_row_data(i) == test_label->get_row_data(i)){
                num++;
            }
        }
    }

    delete predict;

    return num;
}

bool DNN::mini_batch_update(ccma::algebra::BaseMatrixT<real>* mini_batch_data,
                            ccma::algebra::BaseMatrixT<real>* mini_batch_label,
                            real eta){
    std::vector<ccma::algebra::BaseMatrixT<real>*> batch_weights;
    std::vector<ccma::algebra::BaseMatrixT<real>*> batch_biases;
    init_parameter(&batch_weights, 0.0, &batch_biases, 0.0);

    for(int i = 0; i < mini_batch_data->get_rows(); i++){

        std::vector<ccma::algebra::BaseMatrixT<real>*> train_weight;
        std::vector<ccma::algebra::BaseMatrixT<real>*> train_bias;

        auto train_data     = mini_batch_data->get_row_data(i);
        auto train_label    = mini_batch_label->get_row_data(i);

        back_propagation(train_data, train_label, &train_weight, &train_bias);

        delete train_data;
        delete train_label;

        //sum weights & biases
        for(int i = 0; i < _weights.size(); i++){
            batch_weights[i]->add(train_weight[i]);
            batch_biases[i]->add(train_bias[i]);
        }

        //clear train_weight & train_bias
        clear_parameter(&train_weight);
        clear_parameter(&train_bias);
    }

    //batch update with average grad
    for(int i = 0; i < _weights.size(); i++){
        batch_weights[i]->multiply(eta);
        batch_weights[i]->division(mini_batch_data->get_rows());
        _weights[i]->subtract(batch_weights[i]);

        batch_biases[i]->multiply(eta);
        batch_biases[i]->division(mini_batch_data->get_rows());
        _biases[i]->subtract(batch_biases[i]);
    }
}

void DNN::back_propagation(ccma::algebra::BaseMatrixT<real>* train_data,
                           ccma::algebra::BaseMatrixT<real>* train_label,
                           std::vector<ccma::algebra::BaseMatrixT<real>*>* out_weights,
                           std::vector<ccma::algebra::BaseMatrixT<real>*>* out_biases){

    init_parameter(out_weights, 0.0, out_biases, 0.0);

    std::vector<ccma::algebra::BaseMatrixT<real>*> zs;
    std::vector<ccma::algebra::BaseMatrixT<real>*> activations;

    auto activation = train_data->clone();
    activations.push_back(activation);

    //feedforward
    for(int i = 0; i < _weights.size(); i++){
        if(!helper.product(activation, _weights[i], activation)){
            printf("Matrix Dim Error: weights[%d][%d:%d]-activation[%d:%d]\n", i, _weights[i]->get_rows(), _weights[i]->get_cols(), activation->get_rows(), activation->get_cols());
        }
        activation->add(_biases[i]);
        zs.push_back(activation->clone());

        activation->sigmoid();
        activations.push_back(activation->clone());
    }
    delete activation;

    //back pass
    int last_layer = activations.size() - 1;

    auto act = activations[last_layer];
    cost_derivative(act, train_label);

    sigmoid_derivative(zs[last_layer - 1]);

    if(!act->multiply(zs[last_layer - 1])){
        printf("Matrix Dim Error: zs layer[%d][%d:%d]-act[%d:%d]\n", last_layer, zs[last_layer]->get_rows(), zs[last_layer]->get_cols(), act->get_rows(), act->get_cols());
    }
    auto delta = act;
    out_biases->at(last_layer - 1)->set_data(delta);
    delta = out_biases->at(last_layer - 1);

    act = activations[last_layer - 1];
    act->multiply(delta);
    out_weights->at(last_layer - 1)->set_data(act);

    auto mat = new ccma::algebra::DenseMatrixT<real>();
    for(int i = _weights.size() - 2; i >= 0; i--){
        sigmoid_derivative(zs[i]);
        helper.multiply(_weights[i + 1], out_biases->at(i), mat);
        mat->multiply(zs[i]);
        out_biases->at(i)->set_data(mat);

        helper.multiply(activations[i], out_biases->at(i), mat);
        out_weights->at(i)->set_data(mat);
    }
    delete mat;
}
void DNN::cost_derivative(ccma::algebra::BaseMatrixT<real>* output_activation, ccma::algebra::BaseMatrixT<real>* y){
    output_activation->subtract(y);
}

void DNN::sigmoid_derivative(ccma::algebra::BaseMatrixT<real>* z){

    //sigmod(z)*(1-sigmod(z))
    z->sigmoid();

    ccma::algebra::BaseMatrixT<real>* mat = z->clone();
    mat->multiply(-1);
    mat->add(1);
    z->multiply(mat);

    delete mat;
}

void DNN::init_parameter(std::vector<ccma::algebra::BaseMatrixT<real>*>* weight_parameter,
                         real weight_init_value,
                         std::vector<ccma::algebra::BaseMatrixT<real>*>* bias_parameter,
                         real bias_init_value){

    clear_parameter(weight_parameter);
    clear_parameter(bias_parameter);

    for(int i = 1; i < _num_layers; i++){
        ccma::algebra::BaseMatrixT<real>* layer = new ccma::algebra::DenseMatrixT<real>(_sizes[i - 1], _sizes[i]);
        weight_parameter->push_back(layer);

        ccma::algebra::BaseMatrixT<real>* bias = new ccma::algebra::DenseMatrixT<real>(1, _sizes[i]);
        bias_parameter->push_back(bias);
    }
}

void DNN::clear_parameter(std::vector<ccma::algebra::BaseMatrixT<real>*>* parameters){
    size_t len = parameters->size();
    for(size_t i = 0; i < len; i++){
        ccma::algebra::BaseMatrixT<real>* mat = parameters->at(i);
        delete mat;
    }
    parameters->clear();
}


}//namespace nn
}//namespace algorithm
}//namespace ccma
