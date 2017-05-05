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
#include <thread>
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

    uint len = _weights.size();
    for(uint i = 0; i < len; i++){
        uint size = _weights[i]->get_size();
        for(uint j = 0; j < size; j++){
            _weights[i]->set_data(distribution(engine), j);
        }

        size = _biases[i]->get_size();
        for(uint j = 0; j < size; j++){
            _biases[i]->set_data(distribution(engine), j);
        }
    }
}

bool DNN::sgd(ccma::algebra::BaseMatrixT<real>* train_data,
              ccma::algebra::BaseMatrixT<real>* train_label,
              uint epochs,
              real eta,
              uint mini_batch_size,
              ccma::algebra::BaseMatrixT<real>* test_data,
              ccma::algebra::BaseMatrixT<real>* test_label){

    int num_train_data = train_data->get_rows();
    int num_test_data = 0;
    if(test_data != nullptr){
        num_test_data = test_data->get_rows();
    }

    //check nn structure and data dims
    if(_num_layers <= 1 || _sizes[0] != train_data->get_cols() 
            || (num_test_data > 0 && _sizes[0] != test_data->get_cols())
            || train_label->get_cols() != _weights[_weights.size() - 1]->get_cols()){
        return false;
    }

    auto shuffler           = new ccma::utils::Shuffler(num_train_data);

    auto mini_batch_data    = new ccma::algebra::DenseMatrixT<real>();
    auto mini_batch_label   = new ccma::algebra::DenseMatrixT<real>();

    auto row_data           = new ccma::algebra::DenseMatrixT<real>();
    auto row_label          = new ccma::algebra::DenseMatrixT<real>();

    for(uint i = 0; i < epochs; i++){

        clock_t start_time = clock();

        shuffler->shuffle();

        for(uint j = 0; j < num_train_data; j++){

            train_data->get_row_data(shuffler->get_row(j), row_data);
            mini_batch_data->set_row_data(row_data, j % mini_batch_size);

            train_label->get_row_data(shuffler->get_row(j), row_label);
            mini_batch_label->set_row_data(row_label, j % mini_batch_size);

            if(mini_batch_data->get_rows() == mini_batch_size || j == (num_train_data - 1) ){
                mini_batch_update(mini_batch_data, mini_batch_label, eta);

                mini_batch_data->clear_matrix();
                mini_batch_label->clear_matrix();
            }
        }

        clock_t training_time = clock();
        printf("Epoch %d train runtime: %f ms\n", i, static_cast<double>(training_time - start_time)/CLOCKS_PER_SEC*1000);

        if(num_test_data > 0){
            printf("Epoch %d: %d / %d\n", i, evaluate(test_data, test_label), num_test_data);
            printf("Epoch %d predict runtime: %f ms\n", i, static_cast<double>(clock() - training_time)/CLOCKS_PER_SEC*1000);
        }

        printf("Epoch %d runtime: %f ms\n", i, static_cast<double>(clock() - start_time)/CLOCKS_PER_SEC*1000);
    }

    delete row_data;
    delete row_label;

    delete mini_batch_data;
    delete mini_batch_label;

    delete shuffler;

    return true;
}

void DNN::feedforward(ccma::algebra::BaseMatrixT<real>* mat){
    for(uint i = 0; i < _weights.size(); i++){
        mat->dot(_weights[i]);
        mat->add(_biases[i]);
        mat->sigmoid();
    }
}

int DNN::evaluate(ccma::algebra::BaseMatrixT<real>* test_data, ccma::algebra::BaseMatrixT<real>* test_label){

    uint num            = 0;
    auto predict_mat    = new ccma::algebra::DenseMatrixT<real>();
    uint num_test_data  = test_data->get_rows();
    real max_value, value;
    uint max_index;

    for(uint i = 0; i < num_test_data; i++){
        test_data->get_row_data(i, predict_mat);

        feedforward(predict_mat);

        max_value = 0;
        max_index = 0;

        uint size = predict_mat->get_cols();
        for(uint j = 0; j < size; j++){
            value = predict_mat->get_data(j);
            if(value > max_value){
                max_value = value;
                max_index = j;
            }
        }

        if(max_index == test_label->get_data(i)){
            num++;
        }
    }
    delete predict_mat;

    return num;
}

bool DNN::mini_batch_update(ccma::algebra::BaseMatrixT<real>* mini_batch_data,
                            ccma::algebra::BaseMatrixT<real>* mini_batch_label,
                            real eta){

    std::vector<ccma::algebra::BaseMatrixT<real>*> batch_weights;
    std::vector<ccma::algebra::BaseMatrixT<real>*> batch_biases;
    init_parameter(&batch_weights, 0.0, &batch_biases, 0.0);

    uint row = mini_batch_data->get_rows();
    uint weight_size = _weights.size();

    uint num_thread = 10;
    if(num_thread > row){
        num_thread = row;
    }

    /*
    auto train_data  = new ccma::algebra::DenseMatrixT<real>[num_thread];
    auto train_label = new ccma::algebra::DenseMatrixT<real>[num_thread];

    for(uint i = 0; i < row; i++){
        mini_batch_data->get_row_data(i, &train_data[i % num_thread]);
        mini_batch_label->get_row_data(i, &train_label[i % num_thread]);

        if(i % num_thread == num_thread - 1 || i == row - 1){
            uint size = i % num_thread + 1;
            for(uint k = 0; k < size; k++){
//                std::thread t();
            }
        }
    }

    delete[] train_data;
    delete[] train_label;
    */

    auto train_data     = new ccma::algebra::DenseMatrixT<real>();
    auto train_label    = new ccma::algebra::DenseMatrixT<real>();

    std::vector<ccma::algebra::BaseMatrixT<real>*> train_weight;
    std::vector<ccma::algebra::BaseMatrixT<real>*> train_bias;

    for(uint i = 0; i < row; i++){

        mini_batch_data->get_row_data(i, train_data);
        mini_batch_label->get_row_data(i, train_label);

        back_propagation(train_data, train_label, &train_weight, &train_bias);

        //sum weights & biases
        for(uint i = 0; i < weight_size; i++){
            batch_weights[i]->add(train_weight[i]);
            batch_biases[i]->add(train_bias[i]);
        }

        //clear train_weight & train_bias
        clear_parameter(&train_weight);
        clear_parameter(&train_bias);
    }

    delete train_data;
    delete train_label;


    //batch update with average grad
    for(uint i = 0; i < weight_size; i++){
        batch_weights[i]->multiply(eta);
        batch_weights[i]->division(mini_batch_data->get_rows());
        _weights[i]->subtract(batch_weights[i]);

        batch_biases[i]->multiply(eta);
        batch_biases[i]->division(mini_batch_data->get_rows());
        _biases[i]->subtract(batch_biases[i]);
    }

    clear_parameter(&batch_weights);
    clear_parameter(&batch_biases);
}

void DNN::back_propagation(ccma::algebra::BaseMatrixT<real>* train_data,
                           ccma::algebra::BaseMatrixT<real>* train_label,
                           std::vector<ccma::algebra::BaseMatrixT<real>*>* out_weights,
                           std::vector<ccma::algebra::BaseMatrixT<real>*>* out_biases){

    init_parameter(out_weights, 0.0, out_biases, 0.0);

    std::vector<ccma::algebra::BaseMatrixT<real>*> zs;
    std::vector<ccma::algebra::BaseMatrixT<real>*> activations;

    auto activation = new ccma::algebra::DenseMatrixT<real>();
    train_data->clone(activation);

    auto as = new ccma::algebra::DenseMatrixT<real>();
    activation->clone(as);
    activations.push_back(as);//store all activation layer by layer

    //feedforward
    for(uint i = 0; i < _weights.size(); i++){

        activation->dot(_weights[i]);
        activation->add(_biases[i]);

        auto a_clone = new ccma::algebra::DenseMatrixT<real>();
        activation->clone(a_clone);
        zs.push_back(a_clone);//store all z value(not sigmoid) layer by layer

        activation->sigmoid();

        auto as_clone = new ccma::algebra::DenseMatrixT<real>();
        activation->clone(as_clone);
        activations.push_back(as_clone);
    }
    delete activation;

    //back pass

    int last_layer = activations.size() - 1;

    auto act = activations[last_layer];

    //last layer, to calc cost func
    cost_derivative(act, train_label);

    //derivative of the sigmod funtion
    sigmoid_derivative(zs[last_layer - 1]);

    act->multiply(zs[last_layer - 1]);
    auto delta = act;
    out_biases->at(last_layer - 1)->set_data(delta);
    delta = out_biases->at(last_layer - 1);

    act = activations[last_layer - 1];
    act->transpose();
    act->dot(delta);
    out_weights->at(last_layer - 1)->set_data(act);

    auto mat = new ccma::algebra::DenseMatrixT<real>();
    auto delta_weight = new ccma::algebra::DenseMatrixT<real>();
    auto delta_bias = new ccma::algebra::DenseMatrixT<real>();

    for(int i = _weights.size() - 2; i >= 0; i--){
        sigmoid_derivative(zs[i]);

        _weights[i + 1]->clone(delta_weight);
        delta_weight->transpose();

        out_biases->at(i+1)->clone(delta_bias);

        helper.dot(delta_bias, delta_weight, mat);
        mat->multiply(zs[i]);
        out_biases->at(i)->set_data(mat);

        activations[i]->transpose();
        helper.dot(activations[i], out_biases->at(i), mat);
        out_weights->at(i)->set_data(mat);
    }

    delete mat;
    delete delta_weight;
    delete delta_bias;

    clear_parameter(&zs);
    clear_parameter(&activations);
}
void DNN::cost_derivative(ccma::algebra::BaseMatrixT<real>* output_activation, ccma::algebra::BaseMatrixT<real>* y){
    output_activation->subtract(y);
}

void DNN::sigmoid_derivative(ccma::algebra::BaseMatrixT<real>* z){

    //sigmod(z)*(1-sigmod(z))
    z->sigmoid();

    auto mat = new ccma::algebra::DenseMatrixT<real>();
    z->clone(mat);

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
        auto layer = new ccma::algebra::DenseMatrixT<real>(_sizes[i - 1], _sizes[i]);
        weight_parameter->push_back(layer);

        auto bias = new ccma::algebra::DenseMatrixT<real>(1, _sizes[i]);
        bias_parameter->push_back(bias);
    }
}

void DNN::clear_parameter(std::vector<ccma::algebra::BaseMatrixT<real>*>* parameters){
    for(auto parameter : *parameters){
        delete parameter;
    }
    parameters->clear();
}


}//namespace nn
}//namespace algorithm
}//namespace ccma
