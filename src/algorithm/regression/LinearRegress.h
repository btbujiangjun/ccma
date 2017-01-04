/*********************************************
* Author: Jun Jiang - jiangjun4@sina.com
* Created: 2016-12-07 15:14
* Last modified: 2016-12-07 15:14
* Filename: LinearRegress.h
* Description:linear regression
**********************************************/
#ifndef _CCMA_ALGORITHM_REGRESSION_LINEARREGRESS_H_
#define _CCMA_ALGORITHM_REGRESSION_LINEARREGRESS_H_

#include "algebra/BaseMatrix.h"

namespace ccma{
namespace algorithm{
namespace regression{

class LinearRegression{
public:
    /*
     * simple linear regression, samples >> features is need.
     * sum(yi - xiTw)^2 [i = i-->m]
     * w^ = (xTx)-1 * xTy
     */
    template<class T>
    bool standard_regression(ccma::algebra::LabeledDenseMatrixT<T>* train_data, ccma::algebra::DenseColMatrixT<real>* weights);

    /*
     * locally weighted linear regression.
     * to set a personal weight for every train data nearby predict data point
     * w^ = (xTWX)-1 * xTWy
     * gaussian kernal: w(i, i) = exp(|x(i)-x| / 2k^2)
     */
    template<class T>
    bool local_weight_logistic_regresion(ccma::algebra::LabeledDenseMatrixT<T>* train_data,
                                         ccma::algebra::DenseMatrixT<T>* predict_data,
                                         real k,
                                         ccma::algebra::DenseColMatrixT<real>* predict_labels);

};//class LinearRegression


}//regression
}//namespace
}//namespace ccma

#endif //_CCMA_ALGORITHM_REGRESSION_LINEARREGRESS_H_
