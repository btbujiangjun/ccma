/*********************************************
* Author: Jun Jiang - jiangjun4@sina.com
* Created: 2017-04-19 15:54
* Last modified: 2017-04-19 15:54
* Filename: TestMnistHelper.cpp
* Description: 
**********************************************/
#include "utils/MnistHelper.h"

int main(int argc, char** argv){
    ccma::utils::MnistHelper<int> helper;

    auto image_mat = new ccma::algebra::DenseMatrixT<int>();
    helper.read_image("data/mnist/train-images-idx3-ubyte",image_mat, 10);
    image_mat->display("\t");
    delete image_mat;

    auto label_mat = new ccma::algebra::DenseMatrixT<int>();
    helper.read_label("data/mnist/train-labels-idx1-ubyte", label_mat, 10);
    label_mat->display();

    helper.read_vec_label("data/mnist/train-labels-idx1-ubyte", label_mat, 10);
    label_mat->display();

    delete label_mat;
}
