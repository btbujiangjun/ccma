/*********************************************
* Author: Jun Jiang - jiangjun4@sina.com
* Created: 2016-12-07 17:26
* Last modified: 2016-12-07 17:26
* Filename: TestLinearRegress.cpp
* Description: 
**********************************************/
#include <stdio.h>
#include <string.h>
#include "utils/FileOp.h"
#include "LinearRegress.h"

int main(int argc, char** argv){
    /*
    //int a[16] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16};
    real a[9] = {1,2,3,2,2,1,3,4,3};
//    real a[4] = {1,2,3,2};
    ccma::algebra::BaseMatrixT<real> bm(3, 3, a);
    real result = 0;
    bm.det(&result);
    printf("det:%f\n", result);

    bm.display();

    ccma::algebra::BaseMatrixT<real>* res_mat = new ccma::algebra::BaseMatrixT<real>();
    bm.inverse(res_mat);
    res_mat->display();

    ccma::algebra::BaseMatrixT<real>* product_mat = new ccma::algebra::BaseMatrixT<real>();
    bm.inner_product(res_mat, product_mat);
    product_mat->display();
    */

    ccma::algebra::LabeledMatrixT<real, real, char>* lmat = new ccma::algebra::LabeledMatrixT<real, real, char>();
    ccma::utils::DenseFileOp* fo = new ccma::utils::DenseFileOp();
    if(fo->read_data("./data/ex0.txt", lmat)){
        ccma::algorithm::regression::LinearRegression<real, real, char>* regression = new ccma::algorithm::regression::LinearRegression<real, real, char>(lmat);
        if(regression->standard_regression()){
            ccma::algebra::ColMatrixT<real>* weights = regression->get_weights();
            weights->display();
        }
        delete regression;
    }
    delete fo;
    delete lmat;

}
