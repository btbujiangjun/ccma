/*********************************************
* Author: Jun Jiang - jiangjun4@sina.com
* Created: 2016-12-12 19:32
* Last modified: 2016-12-12 19:32
* Filename: TestFileOp.cpp
* Description: 
**********************************************/

#include "FileOp.h"

int main(int argc, char** argv){
    ccma::utils::DenseFileOp* fo = new ccma::utils::DenseFileOp();
    ccma::algebra::BaseMatrixT<float>* mat = new ccma::algebra::BaseMatrixT<float>();
    if(fo->read_data<float>("./data/ex0.txt", mat)){
        mat->display();
    }
    ccma::algebra::LabeledMatrixT<float, float, char>* lmat = new ccma::algebra::LabeledMatrixT<float, float, char>();
    if(fo->read_data<float, float>("./data/ex0.txt", lmat)){
        lmat->display();
    }
    delete mat;
    delete lmat;
    delete fo;
}
