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
#include "algebra/BaseMatrixT.h"

namespace ccma{
namespace algorithm{
namespace tree{

class CartModel{
public:
    CartModel(const std:string& name, ccma::algebra::DenseColMatrixT<real>* value):_name(name), _value(value){
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

    inline string get_name() const{
        return _name;
    }
    inline ccma::algebra::DenseColMatrixT<real>* get_value() const{
        return _value;
    }

    inline void set_left_child(CartModel* left_child){
        left_child->set_is_root(false);
        _left_child = left_child;
        _is_leaf = false;
    }
    inline CartModel* get_left_child() const{
        return _left_child;
    }
    inline void set_right_child(CartModel* right_child){
        right_child->set_is_root(false);
        _right_child = right_child;
        _is_leaf = false;
    }
    inline CartModel* get_right_child() const{
        return _right_child;
    }

    inline set_is_leaf(bool is_leaf){
        _is_leaf = is_leaf;
    }
    inline bool get_is_leaf() const{
        return _is_leaf;
    }
    inline set_is_root(bool is_root){
        _is_root = is_root;
    }
    inline get_is_root() const{
        return _is_root;
    }

    inline bool has_left_child() const{
        return _left_child != nullptr;
    }
    inline bool has_right_child() const{
        return _right_child != nullptr;
    }

    void display(){
        display("", this);
    }
private:
    std::string _name;
    ccma::algebra::DenseColMatrixT<real>* _value;
    bool _is_leaf;
    bool _is_root;
    CartModel* _left_child;
    CartModel* _right_child;

    void display(const std::string& prefix, CartModel* node){
        if(node->is_root()){
            printf("%sroot:[%s]samplses[%s]\n", prefix, node->get_name(), node->get_value()->to_string());
        }else if(node->is_leaf()){
            printf("%svalue:[%s]samples[%s]\n", prefix, node->get_name(), node->get_value()->to_string());
        }else{
            printf("%sfeature:[%s]value[%s]\n", prefix, node->get_name(), node->get_value()->to_string());
        }
        if(node->has_left_child()){
            display(prefix+"\t", node->get_left_child());
        }
        if(node->has_right_child()){
            display(prefix+"\t", node->get_right_child());
        }
    }
};//class CartModel

class ClassificationAndRegressionTree{
/*
public:
    template<class T>
    void train(ccma::algebra::LabeledDenseMatrixT<T>* train_data, CartModel* model);
private:
    template<class T>
    void create_tree(ccma::algebra::LabeledDenseMatrixT<T>* train_data, CartModel* parent);

    template<class T>
*/
};// class ClassificationAndRegressionTree

}//namespace tree
}//namespace algorithm
}//namespace ccma

#endif //_CCMA_ALGORITHM_TREE_CART_H_
