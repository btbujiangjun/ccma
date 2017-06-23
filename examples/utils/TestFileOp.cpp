/*********************************************
* Author: Jun Jiang - jiangjun4@sina.com
* Created: 2016-12-12 19:32
* Last modified: 2016-12-12 19:32
* Filename: TestFileOp.cpp
* Description: 
**********************************************/

#include "utils/FileOp.h"

int main(int argc, char** argv){
    ccma::utils::DenseFileOp* fo = new ccma::utils::DenseFileOp();
    ccma::algebra::DenseMatrixT<float>* mat = new ccma::algebra::DenseMatrixT<float>();
    if(fo->read_data<float>("./data/ex0.txt", mat)){
        mat->display();
    }
    ccma::algebra::LabeledDenseMatrixT<float>* lmat = new ccma::algebra::LabeledDenseMatrixT<float>();
    if(fo->read_data<float>("./data/ex0.txt", lmat)){
        lmat->display();
    }
    delete mat;
    delete lmat;
    delete fo;
}
