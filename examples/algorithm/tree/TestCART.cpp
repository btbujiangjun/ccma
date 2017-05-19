/*********************************************
* Author: Jun Jiang - jiangjun4@sina.com
* Created: 2017-01-12 17:37
* Last modified: 2017-01-12 17:37
* Filename: TestCART.cpp
* Description: tree-based regression
**********************************************/

#include "algorithm/tree/CART.h"
#include "utils/FileOp.h"


int main(int argc, char** argv){
    uint* d = new uint[0];
    auto file_op = new ccma::utils::DenseFileOp();
    auto mat = new ccma::algebra::LabeledDenseMatrixT<real>();

    if(file_op->read_data("./data/ex0.txt", mat)){
        auto reg_tree = new ccma::algorithm::tree::ClassificationAndRegressionTree();
        auto model = new ccma::algorithm::tree::CartModel();

        reg_tree->train(mat, 1, 0, model);
        model->display();

        delete reg_tree;
        delete model;
    }

    delete mat;
    delete file_op;
}

