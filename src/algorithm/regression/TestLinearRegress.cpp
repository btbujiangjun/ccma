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
    ccma::algebra::LabeledDenseMatrixT<real>* lmat = new ccma::algebra::LabeledDenseMatrixT<real>();
    ccma::utils::DenseFileOp* fo = new ccma::utils::DenseFileOp();
    if(fo->read_data("./data/ex0.txt", lmat)){
//        lmat->display();
        ccma::algorithm::regression::LinearRegression* regression = new ccma::algorithm::regression::LinearRegression();
        ccma::algebra::DenseColMatrixT<real>* weights = new ccma::algebra::DenseColMatrixT<real>(lmat->get_cols(), 0.0);
        if(regression->standard_regression<real>(lmat, weights)){
            printf("weights:\n");
            weights->display();
        }
        delete weights;
        delete regression;

        auto predict_mat = new ccma::algebra::DenseMatrixT<real>();
        lmat->get_data_matrix(predict_mat);
        ccma::algebra::DenseColMatrixT<real> predict_labels(1, predict_mat->get_rows());
        if(regression->local_weight_logistic_regression(lmat, predict_mat, 1, &predict_labels)){
            printf("predict_labels:\n");
            predict_labels.display();
        }
        delete predict_mat;
    }
    delete fo;
    delete lmat;

}
