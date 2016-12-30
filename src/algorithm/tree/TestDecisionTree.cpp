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
    //char b[5] = {'y', 'y', 'n', 'n', 'n'};
    char c[3] = {'A', 'B', 'C'};
    ccma::algebra::LabeledDenseMatrixT<int> lm(a, b, 5, 3);
    ccma::algorithm::tree::DecisionTree dt;
    //for(uint i = 0; i < 100000; i++){
    //    printf("[%d]\n", i);
        dt.train(&lm);
    //}
    //
    
    /*
    ccma::algebra::OneDenseMatrixT<int>* one = new ccma::algebra::OneDenseMatrixT<int>(5);
    printf("[%d]\n", one->get_data(4));
    delete one;

    int m[6] = {1,2,3,4,5,6};
    ccma::algebra::BaseMatrixT<int>* m1 = new ccma::algebra::DenseMatrixT<int>(2,3, m);
    ccma::algebra::BaseMatrixT<int>* m2 = new ccma::algebra::DenseMatrixT<int>(3,2, m);
    ccma::algebra::BaseMatrixT<int>* m3 = new ccma::algebra::DenseMatrixT<int>();
    ccma::algebra::BaseMatrixT<int>* m4 = new ccma::algebra::DenseMatrixT<int>();
    m1->product(m2, m3);

    m3->display();

    m3->transpose(m4);
    m4->display();

    m1->transpose(m4);
    m1->display();
    m4->display();
    */

    return 0;
}
