/*********************************************
* Author: Jun Jiang - jiangjun4@sina.com
* Created: 2017-04-19 15:54
* Last modified: 2017-04-19 15:54
* Filename: TestMnistHelper.cpp
* Description: 
**********************************************/
#include "MnistHelper.h"
#include <memory>

int main(int argc, char** argv){
    ccma::utils::MnistHelper<int> helper;
    ccma::algebra::LabeledDenseMatrixT<int>* mat;

    mat = helper.read("data/mnist/train-images-idx3-ubyte", "data/mnist/train-labels-idx1-ubyte", 10);
    for(int i =0; i < mat->get_rows(); i++){
        int* data = mat->get_data_matrix()->get_row_data(i)->get_data();
        auto m = new ccma::algebra::DenseMatrixT<int>(data, 28, 28);
        delete data;
        m->display("");
        delete m;
        auto n = new ccma::algebra::DenseMatrixT<int>(helper.vectorize_label(mat->get_label(i), 10), 1, 10);
        n->display("");
        delete n;
    }
//    mat->display();
    delete mat;
}
