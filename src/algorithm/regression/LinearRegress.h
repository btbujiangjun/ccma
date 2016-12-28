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
     */
    template<class T>
    bool standard_regression(ccma::algebra::LabeledDenseMatrixT<T>* train_data, ccma::algebra::DenseColMatrixT<real>* weights);


};//class LinearRegression


}//regression
}//namespace
}//namespace ccma

#endif //_CCMA_ALGORITHM_REGRESSION_LINEARREGRESS_H_
