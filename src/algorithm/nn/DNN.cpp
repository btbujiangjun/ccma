/*********************************************
* Author: Jun Jiang - jiangjun4@sina.com
* Created: 2017-04-06 17:27
* Last modified: 2017-04-06 17:27
* Filename: DNN.cpp
* Description: 
**********************************************/

#include "algorithm/nn/DNN.h"
#include <random>
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
              real lamda,
              uint mini_batch_size,
              ccma::algebra::BaseMatrixT<real>* test_data,
              ccma::algebra::BaseMatrixT<real>* test_label){

    uint num_train_data = train_data->get_rows();
    uint num_test_data = 0;
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

    auto now = []{return std::chrono::system_clock::now();};

    for(uint i = 0; i < epochs; i++){

        auto start_time = now();

        shuffler->shuffle();

        for(uint j = 0; j < num_train_data; j++){

            if( j % 100 == 0){
                printf("Epoch[%d][%d/%d]training...\r", i, j, num_train_data);
            }

            train_data->get_row_data(shuffler->get_row(j), row_data);
            mini_batch_data->set_row_data(row_data, j % mini_batch_size);

            train_label->get_row_data(shuffler->get_row(j), row_label);
            mini_batch_label->set_row_data(row_label, j % mini_batch_size);

            if( j % mini_batch_size == mini_batch_size - 1 || j == (num_train_data - 1) ){
                mini_batch_update(mini_batch_data, mini_batch_label, eta, lamda, num_train_data);

                mini_batch_data->clear_matrix();
                mini_batch_label->clear_matrix();
            }
        }

        auto training_time = now();
        printf("Epoch %d train runtime: %lld ms\n", i, std::chrono::duration_cast<std::chrono::milliseconds>(training_time - start_time).count());

        if(num_test_data > 0){
            printf("Epoch %d: %d / %d\n", i, evaluate(test_data, test_label), num_test_data);
            printf("Epoch %d predict runtime: %lld ms\n", i, std::chrono::duration_cast<std::chrono::milliseconds>(now() - training_time).count());
        }

        printf("Epoch %d runtime: %lld ms\n", i, std::chrono::duration_cast<std::chrono::milliseconds>(now() - start_time).count());
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

void DNN::mini_batch_update(ccma::algebra::BaseMatrixT<real>* mini_batch_data,
                            ccma::algebra::BaseMatrixT<real>* mini_batch_label,
                            real eta,
                            real lamda,
                            uint n){

    std::vector<ccma::algebra::BaseMatrixT<real>*> batch_weights;
    std::vector<ccma::algebra::BaseMatrixT<real>*> batch_biases;
    init_parameter(&batch_weights, 0.0, &batch_biases, 0.0);

    uint row = mini_batch_data->get_rows();
    uint weight_size = _weights.size();

    uint num_thread = _num_hardware_concurrency;
    if(num_thread > row){
        num_thread = row;
    }

    auto train_data  = new ccma::algebra::DenseMatrixT<real>[num_thread];
    auto train_label = new ccma::algebra::DenseMatrixT<real>[num_thread];

    if(num_thread > 1){
        //multithread parallel training
        uint thread_epochs = row / num_thread;
        if(row % num_thread != 0){
            thread_epochs += 1;
        }

        std::vector<std::thread> threads(num_thread);

        for(uint i = 0; i < thread_epochs; i++){
            uint thread_size = num_thread;
            if(i == thread_epochs - 1){
                thread_size = row - i * num_thread;
            }

            for(uint j = 0; j < thread_size; j++){
                mini_batch_data->get_row_data(i * num_thread + j, &train_data[j]);
                mini_batch_label->get_row_data(i * num_thread + j, &train_label[j]);
                threads[j] = std::thread(std::mem_fn(&DNN::back_propagation), this, &train_data[j], &train_label[j], &batch_weights, &batch_biases);
            }

            for(uint j = 0; j < thread_size; j++){
                threads[j].join();
            }
        }
    }else{
        //main thread training
        for(uint i = 0; i < row; i++){
            mini_batch_data->get_row_data(i, &train_data[0]);
            mini_batch_label->get_row_data(i, &train_label[0]);

            back_propagation(&train_data[0], &train_label[0], &batch_weights, &batch_biases);
        }
    }

    delete[] train_data;
    delete[] train_label;

    /*
     * batch update with average grad
     * w_k --> w'_k = w_k - eta/m * batch_weights
     * b_k --> b'_k = b_k - eta/m * batch_biases
    */
    real weight_decay = 1.0 - eta * (lamda / n);
    for(uint i = 0; i < weight_size; i++){
        batch_weights[i]->multiply(eta);
        batch_weights[i]->division(mini_batch_data->get_rows());
        _weights[i]->multiply(weight_decay);
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
                           std::vector<ccma::algebra::BaseMatrixT<real>*>* batch_weights,
                           std::vector<ccma::algebra::BaseMatrixT<real>*>* batch_biases){

    std::vector<ccma::algebra::BaseMatrixT<real>*> train_weights;
    std::vector<ccma::algebra::BaseMatrixT<real>*> train_biases;
    init_parameter(&train_weights, 0.0, &train_biases, 0.0);

    std::vector<ccma::algebra::BaseMatrixT<real>*> zs;
    std::vector<ccma::algebra::BaseMatrixT<real>*> activations;

    auto activation = new ccma::algebra::DenseMatrixT<real>();
    train_data->clone(activation);

    auto as = new ccma::algebra::DenseMatrixT<real>();
    activation->clone(as);
    activations.push_back(as);//store all activation layer by layer

    /*
     * feedforward
     * a_l = sigmoid(w_l * a_l-1 + b_l)
     */
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

    /*
     * backpropagation
     * L layer(last layer) Error
     * Error δL = cost->delta
     */
    int last_layer = activations.size() - 1;
    auto delta = new ccma::algebra::DenseMatrixT<real>();

    _cost->delta(zs[last_layer -1], activations[last_layer], train_label, delta);

    train_biases[last_layer - 1]->set_data(delta);

    /*
     * Derivative(Cw) = a_in * δ_out
     * a_in = a_L-1, δ_out = delta
     */
    auto act = activations[last_layer - 1];
    act->transpose();
    act->dot(delta);
    train_weights[last_layer - 1]->set_data(act);

    delete delta;

    auto mat = new ccma::algebra::DenseMatrixT<real>();
    auto delta_weight = new ccma::algebra::DenseMatrixT<real>();
    auto delta_bias = new ccma::algebra::DenseMatrixT<real>();

    /*
     * δ_l = ( (w_l+1).T * δ_l+1 ) * Derivative(z_l)
     */
    for(int i = _weights.size() - 2; i >= 0; i--){
        _weights[i + 1]->clone(delta_weight);//w_l+1
        delta_weight->transpose();

        train_biases[i + 1]->clone(delta_bias);//δ_l+1

        helper.dot(delta_bias, delta_weight, mat);

        //_cost->derivative_sigmoid(zs[i]);//Derivative(z_l)
        zs[i]->derivative_sigmoid();//Derivative(z_l)
        mat->multiply(zs[i]);

        train_biases[i]->set_data(mat);//Derivative(Cb) = δ

        /*
         * Derivative(Cw) = a_in * δ_out
         * a_in = a_l-1, δ_out = mat
         * activations include input layer, so l-1 is i.
         */
        activations[i]->transpose();
        helper.dot(activations[i], train_biases[i], mat);
        train_weights[i]->set_data(mat);
    }

    delete mat;
    delete delta_weight;
    delete delta_bias;

    clear_parameter(&zs);
    clear_parameter(&activations);

    //sum weights & biases
    for(uint i = 0; i < _weights.size(); i++){
        batch_weights->at(i)->add(train_weights[i]);
        batch_biases->at(i)->add(train_biases[i]);
    }

    clear_parameter(&train_weights);
    clear_parameter(&train_biases);;
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
