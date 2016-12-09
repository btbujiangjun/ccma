/*********************************************
* Author: Jun Jiang - jiangjun4@sina.com
* Created: 2016-12-07 17:26
* Last modified: 2016-12-07 17:26
* Filename: TestLinearRegress.cpp
* Description: 
**********************************************/
#include <stdio.h>
#include <string.h>
#include "algebra/BaseMatrix.h"

int main(int argc, char** argv){
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

}
