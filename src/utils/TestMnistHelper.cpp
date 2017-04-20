/*********************************************
* Author: Jun Jiang - jiangjun4@sina.com
* Created: 2017-04-19 15:54
* Last modified: 2017-04-19 15:54
* Filename: TestMnistHelper.cpp
* Description: 
**********************************************/
#include "MnistHelper.h"

int main(int argc, char** argv){
    ccma::utils::MnistHelper helper;
    ccma::algebra::LabeledDenseMatrixT<int>* mat;

    mat = helper.read<int>("data/mnist/train-images-idx3-ubyte", "data/mnist/train-labels-idx1-ubyte", 1);
    mat->display();
    delete mat;
}
