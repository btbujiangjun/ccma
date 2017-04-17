/*********************************************
* Author: Jun Jiang - jiangjun4@sina.com
* Created: 2017-04-10 16:33
* Last modified: 2017-04-10 16:33
* Filename: MatrixShuffle.h
* Description: matrix shuffle
**********************************************/

#ifndef _CCMA_ALGEBRA_MATRIXSUFFLE_H_
#define _CCMA_ALGEBRA_MATRIXSUFFLE_H_

#include <random>
#include <vector>
#include <ctime>
#include "BaseMatrix.h"

namespace ccma{
namespace algebra{

template<class T>
class MatrixShuffler{
public:
    MatrixShuffler(ccma::algebra::BaseMatrixT<T>* mat);

    ~MatrixShuffler();

    void shuffle();

    ccma::algebra::BaseMatrixT<T>* get_row(int row_id);

private:
    ccma::algebra::BaseMatrixT<T>* m_mat;
    std::vector<int> m_shuffle_idx;
};//class MatrixShuffler


template<class T>
class LabeledMatrixShuffler : public MatrixShuffler<T>{
public:
    LabeledMatrixShuffler(ccma::algebra::LabeledDenseMatrixT<T>* mat);

    ccma::algebra::BaseMatrixT<T>* get_row(int row_id);
    T get_label(int row_id);
private:
    ccma::algebra::LabeledDenseMatrixT<real>* m_mat;
};//class LabeledMatrixShuffler

template<class T>
MatrixShuffler<T>::MatrixShuffler(ccma::algebra::BaseMatrixT<T>* mat){
    m_mat = mat;
    shuffle();
}

template<class T>
MatrixShuffler<T>::~MatrixShuffler(){
    m_shuffle_idx.clear();
}

template<class T>
void MatrixShuffler<T>::shuffle(){
    m_shuffle_idx.clear();

    int num_data = m_mat->get_rows();
    std::vector<int> idx;

    for(int i = 0; i < num_data; i++){
        idx.push_back(i);
    }

    while(idx.size() > 0){
        std::default_random_engine generator(time(0));
        std::uniform_int_distribution<int> dis(0, idx.size() - 1);

        int idx_value = idx[dis(generator)];
        m_shuffle_idx.push_back(idx_value);

        std::vector<int>::iterator it = idx.begin();
        while(it != idx.end()){
            if(*it == idx_value){
                idx.erase(it);
                break;
            }
            it++;
        }
    }

}

template<class T>
ccma::algebra::BaseMatrixT<T>* MatrixShuffler<T>::get_row(int row_id){
    return m_mat[m_shuffle_idx[row_id]];
}

template<class T>
LabeledMatrixShuffler<T>::LabeledMatrixShuffler(ccma::algebra::LabeledDenseMatrixT<T>* mat){
    m_mat = mat;
    shuffle();
}

template<class T>
ccma::algebra::BaseMatrixT<T>* LabeledMatrixShuffler::get_row(int row_id){
    return m_mat[m_shuffle_idx[row_id]];
}

template<class T>
T LabeledMatrixShuffler:get_label(int row_id){
    return m_mat->get_label(row_id);
}

}//namespace algebra
}//namespace ccma

#endif
