/*********************************************
* Author: Jun Jiang - jiangjun4@sina.com
* Created: 2016-11-22 12:02
* Last modified:	2016-11-24 17:02
* Filename:		DecisionTree.cpp
* Description:  decision tree model
**********************************************/

#include "DecisionTree.h"

namespace ccma{
namespace algorithm{
namespace tree{

DecisionTreeModel::DecisionTreeModel(const std::string& name, const std::string& value) : _name(name), _value(value){
    _children = new std::vector<DecisionTreeModel*>();
    _is_leaf = true;
    _is_root = true;
}

DecisionTreeModel::~DecisionTreeModel(){
    typename std::vector<DecisionTreeModel*>::iterator it = _children->begin();
    while(it != _children->end()){
        DecisionTreeModel* node = *it;
        delete node;
        it = _children->erase(it);
    }
    _children->clear();
    _children = nullptr;
}

void DecisionTreeModel::add_child(DecisionTreeModel* child){
    child->set_root(false);
    _children->push_back(child);
    _is_leaf = false;
}

void DecisionTreeModel::display(const std::string prefix, DecisionTreeModel* node){
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


template<class T>
void DecisionTree::train(ccma::algebra::LabeledDenseMatrixT<T>* train_data, DecisionTreeModel* model){
    create_tree(train_data, model);
}

template<class T>
void DecisionTree::create_tree(ccma::algebra::LabeledDenseMatrixT<T>* mat, DecisionTreeModel* parent){

    ccma::algebra::CCMap<T>* label_cnt_map = mat->get_label_cnt_map();

    //all labels are the same or have no feature
    if(label_cnt_map->size() == 1 || mat->get_cols()== 0){
        DecisionTreeModel* node = new DecisionTreeModel(std::to_string(label_cnt_map->get_max_key()), std::to_string(label_cnt_map->get_max_value()));
        parent->add_child(node);
    }else{
        uint n_best_feature = search_best_feature(mat);

        ccma::algebra::CCMap<T>* uvm = mat->get_feature_cnt_map(n_best_feature);
        typename ccma::algebra::CCMap<T>::iterator it;
        for(it = uvm->begin(); it != uvm->end(); it++){

            ccma::algebra::LabeledDenseMatrixT<T>* sub_matrix = new ccma::algebra::LabeledDenseMatrixT<T>();
            mat->split(n_best_feature, it->first, sub_matrix);

            DecisionTreeModel* node = new DecisionTreeModel(std::to_string(mat->get_feature_name(n_best_feature)), std::to_string(it->first));
            parent->add_child(node);

            create_tree(sub_matrix, node);

            delete sub_matrix;
        }
    }

}


template<class T>
uint DecisionTree::search_best_feature(ccma::algebra::LabeledDenseMatrixT<T>* mat){

    real best_info_gain = 0.0;
    uint best_feature_idx = 0;
    real base_shannon_entropy = mat->get_shannon_entropy();

    for(uint i = 0; i < mat->get_cols(); i++){

        real ent = 0;
        ccma::algebra::CCMap<T>* uvm = mat->get_feature_cnt_map(i);
        typename ccma::algebra::CCMap<T>::iterator it;

        for(it = uvm->begin(); it != uvm->end(); it++){
            ccma::algebra::LabeledDenseMatrixT<T>* sub_matrix = new ccma::algebra::LabeledDenseMatrixT<T>();
            if(mat->split(i, it->first, sub_matrix)){
                real prob = (real)sub_matrix->get_rows()/mat->get_rows();
                ent += prob * sub_matrix->get_shannon_entropy();
            }
            delete sub_matrix;
        }

        real info_gain = base_shannon_entropy - ent;
        if(info_gain >= best_info_gain){
            best_feature_idx = i;
            best_info_gain = info_gain;
        }
    }
    return best_feature_idx;
}

template void DecisionTree::train(ccma::algebra::LabeledDenseMatrixT<int>* train_data, DecisionTreeModel* model);
template void DecisionTree::train(ccma::algebra::LabeledDenseMatrixT<real>* train_data, DecisionTreeModel* model);

template void DecisionTree::create_tree(ccma::algebra::LabeledDenseMatrixT<int>* mat, DecisionTreeModel* parent);
template void DecisionTree::create_tree(ccma::algebra::LabeledDenseMatrixT<real>* mat, DecisionTreeModel* parent);

template uint DecisionTree::search_best_feature(ccma::algebra::LabeledDenseMatrixT<int>* mat);
template uint DecisionTree::search_best_feature(ccma::algebra::LabeledDenseMatrixT<real>* mat);

}//namespace tree
}//namespace algorithm
}//namespace ccma
