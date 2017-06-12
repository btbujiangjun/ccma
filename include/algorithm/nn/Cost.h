/*********************************************
* Author: Jun Jiang - jiangjun4@sina.com
* Created: 2017-05-15 16:10
* Last modified: 2017-05-15 16:10
* Filename: Cost.h
* Description: Abstact class of NN cost func
**********************************************/

#ifndef _CCMA_ALGORITHM_NN_COST_H_
#define _CCMA_ALGORITHM_NN_COST_H_

#include "algebra/BaseMatrix.h"

namespace ccma{
namespace algorithm{
namespace nn{

class Cost{
public:
    virtual ~Cost(){}
    virtual void delta(ccma::algebra::BaseMatrixT<real>* z,
                       ccma::algebra::BaseMatrixT<real>* a,
                       ccma::algebra::BaseMatrixT<real>* y,
                       ccma::algebra::BaseMatrixT<real>* out_cost) = 0;

    void derivative_sigmoid(ccma::algebra::BaseMatrixT<real>* mat);
};//class Cost

/*
 * C = (y - a)^2 / 2
 */
class QuadraticCost:public Cost{
public:
    /*
     * derivative C_w (a-y) * sigmoid(z) * (1 - sigmoid(z))
     * sigmoid(z) = 1.0/(1.0+exp(-z))
     */
    void delta(ccma::algebra::BaseMatrixT<real>* z,
               ccma::algebra::BaseMatrixT<real>* a,
               ccma::algebra::BaseMatrixT<real>* y,
               ccma::algebra::BaseMatrixT<real>* out_cost);
};//class QuadraticCost

/*
 * C = 1/n sum(y*lna + (1-y)*ln(1 - a))
 */
class CrossEntropyCost:public Cost{
public:
    /*
     * derivative C_w = 1/n * ∑(x_j(σ(z) - y))
     */
    void delta(ccma::algebra::BaseMatrixT<real>* z,
               ccma::algebra::BaseMatrixT<real>* a,
               ccma::algebra::BaseMatrixT<real>* y,
               ccma::algebra::BaseMatrixT<real>* out_cost);
};//class CrossEntropyCost

}//namespace
}//namespace algorithm
}//namespace ccma

#endif
