/*********************************************
* Author: Jun Jiang - jiangjun4@sina.com
* Created: 2017-01-11 14:18
* Last modified: 2017-01-11 14:18
* Filename: CART.cpp
* Description: implement classify and regress tree
**********************************************/
#include "algorithm/tree/CART.h"

namespace ccma{
namespace algorithm{
namespace tree{

template<class T>
void ClassificationAndRegressionTree::train(ccma::algebra::LabeledDenseMatrixT<T>* train_data,
                                            const uint& min_sub_mat_rows,
                                            const real& min_diff_var,
                                            CartModel* model){
    create_tree(train_data, min_sub_mat_rows, min_diff_var, model);
}

template<class T>
void ClassificationAndRegressionTree::create_tree(ccma::algebra::LabeledDenseMatrixT<T>* train_data,
                                                  const uint& min_sub_mat_rows,
                                                  const real& min_diff_var,
                                                  CartModel* node){
    uint feature_idx = 0;
    real best_value = 0.0;

    bool flag = choose_best_split(train_data, min_sub_mat_rows, min_diff_var, &feature_idx, &best_value);

    real* data = new real[1];
    data[0] = best_value;
    ccma::algebra::DenseColMatrixT<real>* mat = new ccma::algebra::DenseColMatrixT<real>(data, 1);
    node->set_value(mat);
    delete[] data;

    if(flag){//suitable for binary split
        node->set_name(std::to_string(train_data->get_feature_name(feature_idx)));

        ccma::algebra::LabeledDenseMatrixT<T>* lmat = new ccma::algebra::LabeledDenseMatrixT<T>();
        ccma::algebra::LabeledDenseMatrixT<T>* rmat = new ccma::algebra::LabeledDenseMatrixT<T>();

        train_data->binary_split(feature_idx, static_cast<T>(best_value), lmat, rmat);

        node->set_left_child(new CartModel());
        node->set_right_child(new CartModel());

        create_tree(lmat, min_sub_mat_rows, min_diff_var, node->get_left_child());
        delete lmat;

        create_tree(rmat, min_sub_mat_rows, min_diff_var, node->get_right_child());
        delete rmat;
    }
}

template<class T>
bool ClassificationAndRegressionTree::choose_best_split(ccma::algebra::LabeledDenseMatrixT<T>* mat,
                                                        const uint& min_sub_mat_rows,
                                                        const real& min_diff_var,
                                                        uint* feature_idx,
                                                        real* best_value){
    if(mat->is_unique_label()){
        *best_value = mat->label_mean();
        return false;
    }

    real best_var_value = ccma::utils::get_max_value<real>();
    uint best_feature_idx = 0;
    T best_split_value = 0;

    for(uint i = 0; i < mat->get_cols(); i++){

        T split_value = 0;
        for(uint j = 0; j < mat->get_rows(); j++){

            if( j > 0 && split_value == mat->get_data(j, i)){
                continue;
            }
            split_value = mat->get_data(j, i);

            ccma::algebra::LabeledDenseMatrixT<T>* lmat = new ccma::algebra::LabeledDenseMatrixT<T>();
            ccma::algebra::LabeledDenseMatrixT<T>* rmat = new ccma::algebra::LabeledDenseMatrixT<T>();

            mat->binary_split(i, split_value, lmat, rmat);
            if(lmat->get_rows() < min_sub_mat_rows || rmat->get_rows() < min_sub_mat_rows){
                delete lmat;
		delete rmat;
                continue;
            }

            real var_sum = lmat->label_var() + rmat->label_var();
            if(best_var_value > var_sum){
                best_feature_idx = i;
                best_split_value = split_value;
                best_var_value = var_sum;
            }

            delete lmat;
	    delete rmat;
        }
    }

    if(mat->label_var() - best_var_value < min_diff_var || best_var_value == ccma::utils::get_max_value<real>()){
        *best_value = mat->label_mean();
        return false;
    }

    *feature_idx = best_feature_idx;
    *best_value = static_cast<real>(best_split_value);
    return true;
}


template<class T>
bool ClassificationAndRegressionTree::linear_regression(ccma::algebra::LabeledDenseMatrixT<T>* mat, ccma::algebra::DenseColMatrixT<real>* weights){
    auto x = new ccma::algebra::DenseMatrixT<T>();
    mat->get_data_matrix(x);
    x->add_x0();

    auto y = new ccma::algebra::DenseMatrixT<T>();
    mat->get_labels(y);

    ccma::algebra::DenseMatrixT<T>* xT = new ccma::algebra::DenseMatrixT<T>();
    _helper->transpose(x, xT);

    ccma::algebra::DenseMatrixT<T>* xTx = new ccma::algebra::DenseMatrixT<T>();
    _helper->dot(xT, x, xTx);

    T det = 0;
    if(!xTx->det(&det) || det == static_cast<T>(0)){
        delete x;
	delete y;
	delete xT;
	delete xTx;
        return false;
    }

    ccma::algebra::DenseMatrixT<real>* xTxI = new ccma::algebra::DenseMatrixT<real>();
    xTx->inverse(xTxI);

    ccma::algebra::DenseMatrixT<T>* xTy = new ccma::algebra::DenseMatrixT<T>();
    _helper->dot(xT, y, xTy);

    _helper->dot(xTxI, xTy, weights);

    return true;
}

template<class T>
bool ClassificationAndRegressionTree::model_error(ccma::algebra::LabeledDenseMatrixT<T>* mat, real* error){
    ccma::algebra::DenseColMatrixT<real>* weights = new ccma::algebra::DenseColMatrixT<real>(mat->get_cols(), 1.0);
    if(linear_regression(mat, weights)){
        auto x = new ccma::algebra::DenseMatrixT<T>();
        mat->get_data_matrix(x);

        auto y = new ccma::algebra::DenseMatrixT<T>();
        mat->get_labels(y);

        ccma::algebra::DenseMatrixT<real>* y_predict = new ccma::algebra::DenseMatrixT<real>();
        _helper->dot(x, weights, y_predict);

        ccma::algebra::DenseMatrixT<real>* y_diff = new ccma::algebra::DenseMatrixT<real>();
        _helper->subtract(y, y_predict, y_diff);

        y_diff->pow(2);

        *error = y_diff->sum();

        delete weights;
	delete x;
	delete y;
	delete y_predict;
	delete y_diff;

        return true;
    }

    delete weights;

    return false;
}

template void ClassificationAndRegressionTree::train(ccma::algebra::LabeledDenseMatrixT<int>* train_data,const         uint& min_sub_mat_rows, const real& min_diff_var, CartModel* model);
template void ClassificationAndRegressionTree::train(ccma::algebra::LabeledDenseMatrixT<real>* train_data,const         uint& min_sub_mat_rows, const real& min_diff_var, CartModel* model);

template void ClassificationAndRegressionTree::create_tree(ccma::algebra::LabeledDenseMatrixT<int>* train_data,const uint& min_sub_mat_rows, const real& min_diff_var, CartModel* parent);
template void ClassificationAndRegressionTree::create_tree(ccma::algebra::LabeledDenseMatrixT<real>* train_data,const uint& min_sub_mat_rows, const real& min_diff_var, CartModel* parent);

template bool ClassificationAndRegressionTree::choose_best_split(ccma::algebra::LabeledDenseMatrixT<int>* mat,const     uint& min_sub_mat_rows, const real& min_diff_var, uint* feature_idx, real* best_value);
template bool ClassificationAndRegressionTree::choose_best_split(ccma::algebra::LabeledDenseMatrixT<real>* mat,const     uint& min_sub_mat_rows, const real& min_diff_var, uint* feature_idx, real* best_value);

template bool ClassificationAndRegressionTree::linear_regression(ccma::algebra::LabeledDenseMatrixT<int>* mat, ccma::algebra::DenseColMatrixT<real>* weights);
template bool ClassificationAndRegressionTree::linear_regression(ccma::algebra::LabeledDenseMatrixT<real>* mat, ccma::algebra::DenseColMatrixT<real>* weights);

template bool ClassificationAndRegressionTree::model_error(ccma::algebra::LabeledDenseMatrixT<int>* mat, real* error);
template bool ClassificationAndRegressionTree::model_error(ccma::algebra::LabeledDenseMatrixT<real>* mat, real* error);

}//namespace tree
}//namespace algorithm
}//namespace ccma
