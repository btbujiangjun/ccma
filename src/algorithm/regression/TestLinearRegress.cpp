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

    ccma::algebra::LabeledDenseMatrixT<real>* lmat = new ccma::algebra::LabeledDenseMatrixT<real>();
    ccma::utils::DenseFileOp* fo = new ccma::utils::DenseFileOp();
    if(fo->read_data("./data/ex0.txt", lmat)){
        lmat->display();
        ccma::algorithm::regression::LinearRegression* regression = new ccma::algorithm::regression::LinearRegression();
        ccma::algebra::DenseColMatrixT<real>* weights = new ccma::algebra::DenseColMatrixT<real>(lmat->get_cols(), 0.0);
        if(regression->standard_regression<real>(lmat, weights)){
            printf("weights:\n");
            weights->display();
        }
        delete weights;
        delete regression;
    }
    delete fo;
    delete lmat;

}
