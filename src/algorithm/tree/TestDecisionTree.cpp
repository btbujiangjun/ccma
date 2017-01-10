/*********************************************
* Author: Jun Jiang - jiangjun4@sina.com
* Created: 2016-11-24 15:41
* Last modified:	2016-11-24 17:20
* Filename:		TestDecisionTree.cpp
* Description: decision tree algorithm test class 
**********************************************/

#include "stdio.h"
#include "algebra/BaseMatrix.h"
#include "algorithm/tree/DecisionTree.h"

int main(int argc, char** argv){
    //uint a[15] = {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    int a[15] = {1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 2, 1, 0};
    int b[5] = {1, 1, 0, 0, 0};

    ccma::algebra::LabeledDenseMatrixT<int> lm(a, b, 5, 3);
    ccma::algorithm::tree::DecisionTree dt;
    ccma::algorithm::tree::DecisionTreeModel* model = new ccma::algorithm::tree::DecisionTreeModel("root", std::to_string(lm.get_rows()));
    dt.train(&lm, model);
    model->display();
    delete model;

    return 0;
}
