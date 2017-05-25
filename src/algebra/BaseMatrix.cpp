/*********************************************
* Author: Jun Jiang - jiangjun4@sina.com
* Created: 2017-04-13 15:47
* Last modified: 2017-04-13 15:47
* Filename: BaseMatrix.cpp
* Description: Implemention of BaseMatrixT
**********************************************/

#include "algebra/BaseMatrix.h"
#include <vector>

namespace ccma{
namespace algebra{

template<class T>
bool BaseMatrixT<T>::add(const T value){
    uint size = get_size();
    T* data = get_data();
    for(uint i = 0; i != size; i++){
        data[i] += value;
    }
    return true;
}
template<class T>
bool BaseMatrixT<T>::add(BaseMatrixT<T>* mat){
    uint row = mat->get_rows();
    uint col = mat->get_cols();

    if( (_rows != row && row != 1)
            || _cols != col){
        printf("Add matrix dim Error:[%d-%d][%d-%d]\n", _rows, _cols, row, col);
        return false;
    }

    bool is_diff_rows = (_rows != row);

    uint size = get_size();
    T* data_a = get_data();
    T* data_b = mat->get_data();

    uint num_thread = get_num_thread(size);
    if(num_thread == 1){
        if(!is_diff_rows){
            for(uint i = 0; i != size; i++){
                data_a[i] += data_b[i];
            }
        }else{
            for(uint i = 0; i != size; i++){
                data_a[i] += data_b[i % col];
            }
        }
    }else{
        uint block_size = size/num_thread;
        if(size % num_thread != 0){
            block_size += 1;
        }

        std::vector<std::thread> threads(num_thread);
        for(uint i = 0; i != num_thread; i++){
            threads[i] = std::thread(
                    [&data_a, &data_b, is_diff_rows, col](uint start_idx, uint end_idx){
                        if(!is_diff_rows){
                            for(uint ti = start_idx; ti < end_idx; ti++){
                                data_a[ti] += data_b[ti];
                            }
                        }else{
                            for(uint ti = start_idx; ti < end_idx; ti++){
                                data_a[ti] += data_b[ti % col];
                            }
                        }
                        }, i * block_size , std::min(size, (i + 1) * block_size)
                    );
        }

        for(auto& thread : threads){
            thread.join();
        }
    }

    return true;
}

template<class T>
bool BaseMatrixT<T>::subtract(const T value){
    uint size = get_size();
    T* data = get_data();
    for(uint i = 0; i != size; i++){
        data[i] -= value;
    }

    return true;
}
template<class T>
bool BaseMatrixT<T>::subtract(BaseMatrixT<T>* mat){
    uint row = mat->get_rows();
    uint col = mat->get_cols();
    if((_rows != row && row != 1) || _cols != col){
        printf("Subtract matrix dim Error:[%d-%d][%d-%d]\n", _rows, _cols, row, col);
        return false;
    }

    uint size = get_size();

    T* data_a = get_data();
    T* data_b = mat->get_data();

    bool is_diff_row = (_rows != row);
    if(!is_diff_row){
        for(uint i = 0; i != size; i++){
            data_a[i] -= data_b[i];
        }
    }else{
        for(uint i = 0; i != size; i++){
            data_a[i] -= data_b[i % col];
        }
    }

    return true;
}

template<class T>
bool BaseMatrixT<T>::multiply(const T value){
    uint size = get_size();
    T* data = get_data();

    for(uint i = 0; i != size; i++){
        data[i] *= value;
    }

    return true;
}
template<class T>
bool BaseMatrixT<T>::multiply(BaseMatrixT<T>* mat){
    uint row = mat->get_rows();
    uint col = mat->get_cols();
    if((_rows != row && row != 1) || _cols != col){
        printf("multiply matrix dim Error:[%d-%d][%d-%d]\n", _rows, _cols, row, col);
        return false;
    }

    uint size = get_size();
    T* data_a = get_data();
    T* data_b = mat->get_data();

    bool is_diff_row = (_rows != row);
    if(!is_diff_row){
        for(uint i = 0; i != size; i++){
            data_a[i] *= data_b[i];
        }
    }else{
        for(uint i = 0; i != size; i++){
            data_a[i] *= data_b[i % col];
        }
    }

    return true;
}

template<class T>
bool BaseMatrixT<T>::division(const T value){

    uint size   = get_size();
    T* data     = get_data();

    for(uint i = 0; i != size; i++){
        data[i] /=  value;
    }

    return true;
}

template<class T>
void BaseMatrixT<T>::sigmoid(){

    uint size   = get_size();
    T one       = static_cast<T>(1);
    T* data     = get_data();

    uint num_thread = get_num_thread(size);
    if(num_thread == 1){
        for(uint i = 0; i != size; i++){
            data[i] = one / (one + std::exp(-data[i]));
        }
    }else{
        uint block_size = (size % num_thread == 0) ? size / num_thread : (size / num_thread + 1);

        printf("threads[%d]-block size[%d]\n", num_thread, block_size);

        std::vector<std::thread> threads(num_thread);
        for(uint i = 0; i != num_thread; i++){
            threads[i] = std::thread([&data, &one](uint start_idx, uint end_idx){
                                            for(uint ti = start_idx; ti != end_idx; ti++){
                                                data[ti] = one / (one + std::exp(-data[ti]));
                                            }
                                        }, i * block_size, std::min(size, (i + 1) * block_size)
                                    );
        }

        for(auto& thread : threads){
            thread.join();
        }
    }
}

/*
 *sigmoid(z) * (1-sigmoid(z)
 */
template<class T>
void BaseMatrixT<T>::derivative_sigmoid(){
    sigmoid();

    auto clone_mat = new ccma::algebra::DenseMatrixT<T>();
    this->clone(clone_mat);

    clone_mat->multiply(-1);
    clone_mat->add(1);

    multiply(clone_mat);
    delete clone_mat;
}

template<class T>
void BaseMatrixT<T>::x_sum(){
    if(_rows > 1){
        T* data = get_data();
        T* new_data = new T[_cols];
        for(int i = 0; i != _cols; i++){
            for(int j = 0; j != _rows; j++){
                new_data[i] += data[j * _cols + i];
            }
        }
        set_shallow_data(new_data, 1, _cols);
    }
}
template<class T>
void BaseMatrixT<T>::y_sum(){
    if(_cols > 1){
        T* data = get_data();
        T* new_data = new T[_rows];
        for(int i = 0; i != _rows; i++){
            for(int j = 0; j != _cols; j++){
                new_data[i] += data[i * _cols + j];
            }
        }
        set_shallow_data(new_data, _rows, 1);
    }
}


template class BaseMatrixT<int>;
template class BaseMatrixT<real>;

}//namespace ccma
}//namespace algebra
