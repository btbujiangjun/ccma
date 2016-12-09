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

template<class T, class LT, class FT>
class DecisionTree{
public:
    explicit DecisionTree(ccma::algebra::LabeledMatrixT<T, LT, FT>* train_data) : _train_data(train_data){}

    void train(){
        create_tree(_train_data);
    }

private:
    void create_tree(ccma::algebra::LabeledMatrixT<T, LT, FT>* mat);
    uint search_best_feature(ccma::algebra::LabeledMatrixT<T, LT, FT>* mat);

    ccma::algebra::LabeledMatrixT<T, LT, FT>* _train_data;
};//class DecisionTree


template<class T, class LT, class FT>
void DecisionTree<T, LT, FT>::create_tree(ccma::algebra::LabeledMatrixT<T, LT, FT>* mat){

    ccma::algebra::CCMap<LT>* label_map = mat->get_label_map();

    //all labels are the same or have no feature
    if(label_map->size() == 1 || mat->get_cols()== 0){
        printf("\t\tleaf node [%c][%d]\n", label_map->get_max_key(), label_map->get_max_value());
    }else{
        uint n_best_feature = search_best_feature(mat);
        printf("best feature[%c]rows[%d]\n", mat->get_feature_label(n_best_feature), mat->get_rows());

        ccma::algebra::CCMap<T>* uvm = mat->get_feature_map(n_best_feature);
        typename ccma::algebra::CCMap<T>::iterator it;
        for(it = uvm->begin(); it != uvm->end(); it++){

            ccma::algebra::LabeledMatrixT<T, LT, FT>* sub_matrix = new ccma::algebra::LabeledMatrixT<T, LT, FT>();
            mat->split(n_best_feature, it->first, sub_matrix);
            printf("\ttree[%c][%d]\n", mat->get_feature_label(n_best_feature), it->first);

            create_tree(sub_matrix);

            delete sub_matrix;
        }
    }

}


template<class T, class LT, class FT>
uint DecisionTree<T, LT, FT>::search_best_feature(ccma::algebra::LabeledMatrixT<T, LT, FT>* mat){

    real best_info_gain = 0.0;
    uint best_feature_idx = 0;
    real base_shannon_entropy = mat->get_shannon_entropy();

    for(uint i = 0; i < mat->get_cols(); i++){

        real ent = 0;
        ccma::algebra::CCMap<T>* uvm = mat->get_feature_map(i);
        typename ccma::algebra::CCMap<T>::iterator it;

        for(it = uvm->begin(); it != uvm->end(); it++){

            ccma::algebra::LabeledMatrixT<T, LT, FT>* sub_matrix = new ccma::algebra::LabeledMatrixT<T, LT, FT>();
            mat->split(i, it->first, sub_matrix);

            real prob = (real)sub_matrix->get_rows()/mat->get_rows();
            ent += prob * sub_matrix->get_shannon_entropy();

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
