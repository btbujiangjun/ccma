/*********************************************
* Author: Jun Jiang - jiangjun4@sina.com
* Created: 2016-11-22 12:02
* Last modified:	2016-11-24 17:02
* Filename:		DecisionTree.h
* Description:  decision tree model
**********************************************/
#ifndef _CCMA_ALGORITHM_TREE_DECISIONTREE_H_
#define _CCMA_ALGORITHM_TREE_DECISIONTREE_H_

#include <vector>
#include <string.h>
#include "algebra/BaseMatrix.h"

namespace ccma{
namespace algorithm{
namespace tree{

class DecisionTreeModel{
public:
    DecisionTreeModel(const std::string& name, const std::string& value):_name(name), _value(value){
        _children = new std::vector<DecisionTreeModel*>();
        _is_leaf = true;
        _is_root = true;
    }
    ~DecisionTreeModel(){
        typename std::vector<DecisionTreeModel*>::iterator it = _children->begin();
        while(it != _children->end()){
            DecisionTreeModel* node = *it;
            delete node;
            it = _children->erase(it);
        }
        _children->clear();
        _children = nullptr;
    }

    inline std::string get_name() const{
        return _name;
    }

    inline std::string get_value() const{
        return _value;
    }

    void add_child(DecisionTreeModel* child){
        child->set_root(false);
        _children->push_back(child);
        _is_leaf = false;
    }

    inline bool is_leaf() const{
        return _is_leaf;
    }

    inline bool is_root() const{
        return _is_root;
    }

    inline void set_root(bool is_root){
        _is_root = is_root;
    }

    inline std::vector<DecisionTreeModel*>* get_children(){
        return _children;
    }

    void display(){
        display("", this);
    }
private:
    std::string _name;
    std::string _value;
    bool _is_leaf;
    bool _is_root;
    std::vector<DecisionTreeModel*>* _children;

    void display(const std::string prefix, DecisionTreeModel* node){
        if(node->is_root()){
            printf("%sroot:[%s]samples:[%s]\n", prefix.c_str(), node->get_name().c_str(), node->get_value().c_str());
        }else if(node->is_leaf()){
            printf("%slabel:[%s]samples:[%s]\n", prefix.c_str(), node->get_name().c_str(), node->get_value().c_str());
        }else{
            printf("%sfeature:[%s]value:[%s]\n", prefix.c_str(), node->get_name().c_str(), node->get_value().c_str());
        }

        std::vector<DecisionTreeModel*>* children = node->get_children();
        typename std::vector<DecisionTreeModel*>::iterator it = children->begin();
        while(it != children->end()){
            display(prefix+"\t", *it++);
        }
    }
};

class DecisionTree{
public:
    template<class T>
    void train(ccma::algebra::LabeledDenseMatrixT<T>* train_data, DecisionTreeModel* model);

private:
    template<class T>
    void create_tree(ccma::algebra::LabeledDenseMatrixT<T>* mat, DecisionTreeModel* parent);

    template<class T>
    uint search_best_feature(ccma::algebra::LabeledDenseMatrixT<T>* mat);
};//class DecisionTree


}//namespace tree
}//namespace algorithm
}//namespace ccma

#endif //_CCMA_ALGORITHM_TREE_DECISIONTREE_H_
