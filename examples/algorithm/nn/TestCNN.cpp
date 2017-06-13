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
    if(!(cnn->add_layer(new ccma::algorithm::cnn::DataLayer(28, 28)) &&
        cnn->add_layer(new ccma::algorithm::cnn::ConvolutionLayer(5, 1, 6)) &&
        cnn->add_layer(new ccma::algorithm::cnn::SubSamplingLayer(2, new ccma::algorithm::cnn::MeanPooling())) &&
        cnn->add_layer(new ccma::algorithm::cnn::ConvolutionLayer(5, 1, 12)) &&
        cnn->add_layer(new ccma::algorithm::cnn::SubSamplingLayer(2, new ccma::algorithm::cnn::MeanPooling())) &&
        cnn->add_layer(new ccma::algorithm::cnn::FullConnectionLayer(10)))){
        delete cnn;
        return -1;
    }

    ccma::utils::MnistHelper<real> helper;
    const uint training_cnt = 10000;
    const uint test_cnt = 1000;
    auto train_data     = new ccma::algebra::DenseMatrixT<real>();
    helper.read_image("data/mnist/train-images-idx3-ubyte",train_data, training_cnt);
    auto train_label    = new ccma::algebra::DenseMatrixT<real>();
    helper.read_vec_label("data/mnist/train-labels-idx1-ubyte", train_label, training_cnt);

    auto test_data      = new ccma::algebra::DenseMatrixT<real>();
    helper.read_image("data/mnist/t10k-images-idx3-ubyte", test_data, test_cnt);

    auto test_label     = new ccma::algebra::DenseMatrixT<real>();
    helper.read_label("data/mnist/t10k-labels-idx1-ubyte", test_label, test_cnt);
    cnn->train(train_data, train_label, 50, test_data, test_label);
//    cnn->train(train_data, train_label, 1);

    delete cnn;
    delete train_data;
    delete train_label;
    delete test_data;
    delete test_label;
}
