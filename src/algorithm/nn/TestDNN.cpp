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

    ccma::utils::MnistHelper helper;
    ccma::algebra::LabeledDenseMatrixT<real>* train_mat;
    ccma::algebra::LabeledDenseMatrixT<real>* test_mat;

    train_mat = helper.read<real>("data/mnist/train-images-idx3-ubyte", "data/mnist/train-labels-idx1-ubyte", 1000);
    test_mat = helper.read<real>("data/mnist/t10k-images-idx3-ubyte", "data/mnist/t10k-labels-idx1-ubyte", 1000);

    dnn->sgd(train_mat, 30, 3, 10, test_mat);

    delete train_mat;
    delete test_mat;
    delete dnn;
}
