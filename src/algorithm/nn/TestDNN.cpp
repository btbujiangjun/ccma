/*********************************************
* Author: Jun Jiang - jiangjun4@sina.com
* Created: 2017-01-12 17:37
* Last modified:	2017-04-07 16:12
* Filename:		TestDNN.cpp
* Description: DNN test
**********************************************/

#include "DNN.h"
#include <iostream>
#include "utils/MnistHelper.h"

int main(int argc, char** argv){
    ccma::algorithm::nn::DNN* dnn = new ccma::algorithm::nn::DNN();
    dnn->add_layer(784);
    dnn->add_layer(30);
    dnn->add_layer(10);
    dnn->init_networks_weights();

    ccma::utils::MnistHelper<real> helper;

    auto train_data     = new ccma::algebra::DenseMatrixT<real>();
    helper.read_image("data/mnist/train-images-idx3-ubyte",train_data, 100);

    auto train_label    = new ccma::algebra::DenseMatrixT<real>();
    helper.read_vec_label("data/mnist/train-labels-idx1-ubyte", train_label, 100);

    auto test_data      = new ccma::algebra::DenseMatrixT<real>();
    helper.read_image("data/mnist/t10k-images-idx3-ubyte", test_data, 100);

    auto test_label     = new ccma::algebra::DenseMatrixT<real>();
    helper.read_label("data/mnist/t10k-labels-idx1-ubyte", test_label, 100);

    dnn->sgd(train_data, train_label, 30, 3, 10, test_data, test_label);

    delete train_data, train_label;
    delete test_data, test_label;
    delete dnn;
}
