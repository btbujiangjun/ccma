/*********************************************
* Author: Jun Jiang - jiangjun4@sina.com
* Created: 2017-01-09 19:36
* Last modified:	2017-01-10 15:57
* Filename:		CART.h
* Description: Classification and Regression Tree
**********************************************/

#ifndef _CCMA_ALGORITHM_TREE_CART_H_
#define _CCMA_ALGORITHM_TREE_CART_H_

#include <string.h>
#include "algebra/BaseMatrix.h"
#include "utils/MatrixHelper.h"

namespace ccma{
namespace algorithm{
namespace tree{

class CartModel{
public:
    CartModel(){
        _name = "";
        _value = nullptr;
        _is_leaf = true;
        _is_root = true;
        _left_child = nullptr;
        _right_child = nullptr;
    }
    CartModel(ccma::algebra::DenseColMatrixT<real>* value):_value(value){
        _name = "";
        _is_leaf = true;
        _is_root = true;
        _left_child = nullptr;
        _right_child = nullptr;
    }
    CartModel(const std::string& name, ccma::algebra::DenseColMatrixT<real>* value):_name(name), _value(value){
        _is_leaf = true;
        _is_root = true;
        _left_child = nullptr;
        _right_child = nullptr;
    }

    ~CartModel(){
        if(_value != nullptr){
            delete _value;
        }
        if(_left_child != nullptr){
            delete _left_child;
        }
        if(_right_child != nullptr){
            delete _right_child;
        }
    }

    inline std::string get_name() const{
        return _name;
    }
    inline void set_name(const std::string& name){
        _name = name;
    }
    inline ccma::algebra::DenseColMatrixT<real>* get_value() const{
        return _value;
    }
    inline void set_value(ccma::algebra::DenseColMatrixT<real>* value){
        if(_value != nullptr){
            delete _value;
        }
        _value = value;
    }

    inline void set_left_child(CartModel* left_child){
        left_child->set_is_root(false);
        _left_child = left_child;
        _is_leaf = false;
    }
    inline bool has_left_child() const{
        return _left_child != nullptr;
    }
    inline CartModel* get_left_child() const{
        return _left_child;
    }
    inline void set_right_child(CartModel* right_child){
        right_child->set_is_root(false);
        _right_child = right_child;
        _is_leaf = false;
    }
    inline bool has_right_child() const{
        return _right_child != nullptr;
    }
    inline CartModel* get_right_child() const{
        return _right_child;
    }

    inline void set_is_leaf(bool is_leaf){
        _is_leaf = is_leaf;
    }
    inline bool get_is_leaf() const{
        return _is_leaf;
    }
    inline void set_is_root(bool is_root){
        _is_root = is_root;
    }
    inline bool get_is_root() const{
        return _is_root;
    }

    void display(){
        display("", this, "root");
    }
private:
    std::string _name;
    ccma::algebra::DenseColMatrixT<real>* _value;
    bool _is_leaf;
    bool _is_root;
    CartModel* _left_child;
    CartModel* _right_child;

    void display(const std::string& prefix,
                 CartModel* node,
                 const std::string& node_type){
        if(node->get_is_leaf()){
            if(node->get_value()->get_rows() == 1 && node->get_value()->get_cols() == 1){
                printf("%s[%s]value:[%s]\n", prefix.c_str(), node_type.c_str(), std::to_string(node->get_value()->get_data(0)).c_str());
            }else{
                printf("%s[%s]value:[%s]\n", prefix.c_str(), node_type.c_str(), node->get_value()->to_string()->c_str());
            }
        }else{
            printf("%s[%s]feature_name:[%s]split_value[%s]\n", prefix.c_str(), node_type.c_str(), node->get_name().c_str(), node->get_value()->to_string()->c_str());
        }

        if(node->has_left_child()){
            display(prefix+"\t", node->get_left_child(), "left");
        }

        if(node->has_right_child()){
            display(prefix+"\t", node->get_right_child(), "right");
        }
    }
};//class CartModel

class ClassificationAndRegressionTree{
public:
    ClassificationAndRegressionTree(){
        _helper = new ccma::utils::MatrixHelper();
    }
    ~ClassificationAndRegressionTree(){
        if(_helper != nullptr){
            delete _helper;
        }
    }

    template<class T>
    void train(ccma::algebra::LabeledDenseMatrixT<T>* train_data,
               const uint& min_sub_mat_rows,
               const real& min_dif_var,
               CartModel* model);
private:
    ccma::utils::MatrixHelper* _helper = nullptr;

    template<class T>
    void create_tree(ccma::algebra::LabeledDenseMatrixT<T>* train_data,
                     const uint& min_sub_mat_rows,
                     const real& min_diff_var,
                     CartModel* node);

    template<class T>
    bool choose_best_split(ccma::algebra::LabeledDenseMatrixT<T>* mat,
                           const uint& min_sub_mat_rows,
                           const real& min_diff_var,
                           uint* feature_idx,
                           real* best_value);

    template<class T>
    bool linear_regression(ccma::algebra::LabeledDenseMatrixT<T>* mat, ccma::algebra::DenseColMatrixT<real>* weights);

    template<class T>
    bool model_error(ccma::algebra::LabeledDenseMatrixT<T>* mat, real* error);
};// class ClassificationAndRegressionTree

}//namespace tree
}//namespace algorithm
}//namespace ccma

#endif //_CCMA_ALGORITHM_TREE_CART_H_
