/*********************************************
* Author: Jun Jiang - jiangjun4@sina.com
* Created: 2016-11-22 12:02
* Last modified:	2016-11-24 17:02
* Filename:		DecisionTree.h
* Description:  decision tree model
**********************************************/
#ifndef _CCMA_ALGORITHM_TREE_DECISIONTREE_H_
#define _CCMA_ALGORITHM_TREE_DECISIONTREE_H_

#include "algebra/BaseMatrix.h"

namespace ccma{
namespace algorithm{
namespace tree{

class DecisionTree{
public:
    template<class T>
    void train(ccma::algebra::LabeledDenseMatrixT<T>* train_data){
        train_data->display();
        create_tree(train_data);
    }

private:
    template<class T>
    void create_tree(ccma::algebra::LabeledDenseMatrixT<T>* mat);

    template<class T>
    uint search_best_feature(ccma::algebra::LabeledDenseMatrixT<T>* mat);
};//class DecisionTree


template<class T>
void DecisionTree::create_tree(ccma::algebra::LabeledDenseMatrixT<T>* mat){

    ccma::algebra::CCMap<T>* label_cnt_map = mat->get_label_cnt_map();

    //all labels are the same or have no feature
    if(label_cnt_map->size() == 1 || mat->get_cols()== 0){
        printf("\t\tleaf node [%d][%d]\n", label_cnt_map->get_max_key(), label_cnt_map->get_max_value());
    }else{
        uint n_best_feature = search_best_feature(mat);
        printf("best feature[%d]rows[%d]\n", mat->get_feature_name(n_best_feature), mat->get_rows());

        ccma::algebra::CCMap<T>* uvm = mat->get_feature_cnt_map(n_best_feature);
        typename ccma::algebra::CCMap<T>::iterator it;
        for(it = uvm->begin(); it != uvm->end(); it++){

            ccma::algebra::LabeledDenseMatrixT<T>* sub_matrix = new ccma::algebra::LabeledDenseMatrixT<T>();
            mat->split(n_best_feature, it->first, sub_matrix);
            printf("\ttree[%d][%d]\n", mat->get_feature_name(n_best_feature), it->first);

            create_tree(sub_matrix);

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

}//namespace tree
}//namespace algorithm
}//namespace ccma

#endif //_CCMA_ALGORITHM_TREE_DECISIONTREE_H_
