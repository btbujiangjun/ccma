/*********************************************
* Author: Jun Jiang - jiangjun4@sina.com
* Created: 2017-06-01 11:22
* Last modified: 2017-06-01 11:22
* Filename: TestCNN.cpp
* Description: 
**********************************************/
#include <iostream>
#include <algorithm/cnn/CNN.h>
#include <utils/MnistHelper.h>

int main(int argc, char** argv){
    auto cnn = new ccma::algorithm::cnn::CNN();
    cnn->add_layer(new ccma::algorithm::cnn::DataLayer(28, 28));
    cnn->add_layer(new ccma::algorithm::cnn::ConvolutionLayer(5, 1, 3));
    cnn->add_layer(new ccma::algorithm::cnn::SubSamplingLayer(2));
    cnn->add_layer(new ccma::algorithm::cnn::FullConnectionLayer(10));

    ccma::utils::MnistHelper<real> helper;
    auto train_data     = new ccma::algebra::DenseMatrixT<real>();
    helper.read_image("data/mnist/train-images-idx3-ubyte",train_data, -1);
    auto train_label    = new ccma::algebra::DenseMatrixT<real>();
    helper.read_vec_label("data/mnist/train-labels-idx1-ubyte", train_label, -1);

    auto test_data      = new ccma::algebra::DenseMatrixT<real>();
    helper.read_image("data/mnist/t10k-images-idx3-ubyte", test_data, -1);

    auto test_label     = new ccma::algebra::DenseMatrixT<real>();
    helper.read_label("data/mnist/t10k-labels-idx1-ubyte", test_label, -1);
    cnn->train(train_data, train_label, 2, test_data, test_label);

    delete cnn;
    delete train_data;
    delete train_label;
    delete test_data;
    delete test_label;
}
