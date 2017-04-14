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
    _num_layers++;

    if(_num_layers > 1){
        ccma::algebra::DenseColMatrixT<real>* layer = new ccma::algebra::DenseColMatrixT<real>(neural_size, 0);
        _weights.push_back(layer);

        ccma::algebra::DenseColMatrixT<real>* bias = new ccma::algebra::DenseColMatrixT<real>(neural_size, 0);
        _biases.push_back(bias);
    }

    return _num_layers;
}

void DNN::init_networks(){
    std::default_random_engine engine(time(0));
    //mean & stddev
    std::normal_distribution<real> distribution(0.0, 1.0);

    for(int i = 1; i < _num_layers; i++){
        ccma::algebra::DenseColMatrixT<real>* weight = _weights[i-1];
        ccma::algebra::DenseColMatrixT<real>* bias = _biases[i-1];
        for(int j = 0; j < weight->get_rows(); j++){
            weight->set_data(distribution(engine), j, 0);
            bias->set_data(distribution(engine), j, 0);
        }

        weight->display();
        bias->display();

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
                //batch_update

                mini_batch->clear_matrix();
            }
        }

    }

    delete mini_batch;
    delete shuffler;

    return true;
}


bool DNN::mini_batch_update(ccma::algebra::BaseMatrixT<real>* mini_batch, real eta){
    std::vector<ccma::algebra::DenseColMatrixT<real>*> batch_weights;
    std::vector<ccma::algebra::DenseColMatrixT<real>*> batch_biases;
    init_parameter(&batch_weights, 0.0);
    init_parameter(&batch_biases, 0.0);

    ccma::algebra::LabeledDenseMatrixT<real>* label_data = new ccma::algebra::LabeledDenseMatrixT<real>();
    for(int i = 0; i < mini_batch->get_rows(); i++){

        real* data = new real[_sizes[0]];
        memcpy(data, mini_batch->get_row_data(i)->get_data(), sizeof(real) * _sizes[0]);
        real label = mini_batch->get_row_data(i)->get_data()[_sizes[0]];
        label_data->set_shallow_data(data, &label, 1, _sizes[0]);

        std::vector<ccma::algebra::DenseColMatrixT<real>*> train_weight;
        std::vector<ccma::algebra::DenseColMatrixT<real>*> train_bias;

        back_propagation(label_data, &train_weight, &train_bias);

        //sum weights & biases
        for(int i = 0; i < _num_layers; i++){
            batch_weights[i]->add(train_weight[i]);
            batch_biases[i]->add(train_bias[i]);
        }

    }
    delete label_data;

    //batch update with average grad
    real eta_average = eta/mini_batch->get_rows();
    for(int i = 0; i < _num_layers; i++){
        batch_weights[i]->multiply(eta_average);
        _weights[i]->subtract(batch_weights[i]);

        batch_biases[i]->multiply(eta_average);
        _biases[i]->subtract(batch_biases[i]);
    }
}

void DNN::back_propagation(const ccma::algebra::LabeledDenseMatrixT<real>* train_data,
                           std::vector<ccma::algebra::DenseColMatrixT<real>*>* out_weights,
                           std::vector<ccma::algebra::DenseColMatrixT<real>*>* out_biases){
    std::vector<ccma::algebra::DenseColMatrixT<real>*> local_weights;
    std::vector<ccma::algebra::DenseColMatrixT<real>*> local_biases;
    std::vector<ccma::algebra::DenseColMatrixT<real>*> activations;
    init_parameter(&local_weights, 0.0);
    init_parameter(&local_biases, 0.0);

    

}

void DNN::init_parameter(std::vector<ccma::algebra::DenseColMatrixT<real>*>* out_parameters,real init_value){
    for(int i = 1; i < _num_layers; i++){
        out_parameters->push_back(new ccma::algebra::DenseColMatrixT<real>(_sizes[i-1], init_value));
    }
}

}//namespace nn
}//namespace algorithm
}//namespace ccma
