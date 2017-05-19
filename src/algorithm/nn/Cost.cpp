/*********************************************
* Author: Jun Jiang - jiangjun4@sina.com
* Created: 2017-05-16 11:45
* Last modified: 2017-05-16 11:45
* Filename: Cost.cpp
* Description: Implemention of nn cost
**********************************************/

#include "algorithm/nn/Cost.h"

namespace ccma{
namespace algorithm{
namespace nn{

/*
 * sigmoid(z) * (1-sigmoid(z))
 */
void Cost::derivative_sigmoid(ccma::algebra::BaseMatrixT<real>* mat){
    auto sigz  = new ccma::algebra::DenseMatrixT<real>();

    mat->sigmoid();

    mat->clone(sigz);
    sigz->multiply(-1);
    sigz->add(1);

    mat->multiply(sigz);
    delete sigz;
}

/*
 * derivative C_w (a-y) * sigmoid(z) * (1 - sigmoid(z))
 * sigmoid(z) = 1.0/(1.0+exp(-z))
 */
void QuadraticCost::delta(ccma::algebra::BaseMatrixT<real>* z,
                          ccma::algebra::BaseMatrixT<real>* a,
                          ccma::algebra::BaseMatrixT<real>* y,
                          ccma::algebra::BaseMatrixT<real>* out_cost){
    auto cost   = new ccma::algebra::DenseMatrixT<real>();
    auto sigz  = new ccma::algebra::DenseMatrixT<real>();

    a->clone(cost);
    z->clone(sigz);

    cost->subtract(y);

    this->derivative_sigmoid(sigz);

    cost->multiply(sigz);

    cost->clone(out_cost);

    delete cost;
    delete sigz;
}

/*
 * derivative C_w = 1/n * ∑(x_j(σ(z) - y))
 */
void CrossEntropyCost::delta(ccma::algebra::BaseMatrixT<real>* z,
                             ccma::algebra::BaseMatrixT<real>* a,
                             ccma::algebra::BaseMatrixT<real>* y,
                             ccma::algebra::BaseMatrixT<real>* out_cost){
    auto cost   = new ccma::algebra::DenseMatrixT<real>();
    a->clone(cost);
    cost->subtract(y);

    cost->clone(out_cost);

    delete cost;
}

}
}//namespace algorithm
}//namespace ccma
