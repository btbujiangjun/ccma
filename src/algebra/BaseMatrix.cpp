/*********************************************
* Author: Jun Jiang - jiangjun4@sina.com
* Created: 2017-04-13 15:47
* Last modified: 2017-04-13 15:47
* Filename: BaseMatrix.cpp
* Description: Implemention of BaseMatrixT
**********************************************/

#include "BaseMatrix.h"
#include <vector>

namespace ccma{
namespace algebra{

template<class T>
bool BaseMatrixT<T>::add(const T value){
    uint size = get_size();
    T* data = get_data();
    for(uint i = 0; i < size; i++){
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

    bool is_diff_rows = (_rows > row);

    uint size = get_size();
    T* data_a = get_data();
    T* data_b = mat->get_data();

    uint num_thread = get_num_thread(size);
    if(num_thread == 1){
        if(!is_diff_rows){
            for(uint i = 0; i < size; i++){
                data_a[i] += data_b[i];
            }
        }else{
            for(uint i = 0; i < size; i++){
                data_a[i] += data_b[i % col];
            }
        }
    }else{
        uint block_size = size/num_thread;
        if(size % num_thread != 0){
            block_size += 1;
        }

        //printf("size[%d]thread[%d]block_size[%d]\n", size, num_thread, block_size);
        std::vector<std::thread> threads(num_thread);
        for(uint i = 0; i < num_thread; i++){
            threads[i] = std::thread(
                    [&data_a, &data_b](uint start_idx, uint end_idx){
                            for(uint ti = start_idx; ti < end_idx; ti++){
                                data_a[ti] += data_b[ti];
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
    for(uint i = 0; i < size; i++){
        data[i] -= value;
    }

    return true;
}
template<class T>
bool BaseMatrixT<T>::subtract(BaseMatrixT<T>* mat){
    if( _rows != mat->get_rows() || _cols != mat->get_cols()){
        printf("Subtract matrix dim Error:[%d-%d][%d-%d]\n", _rows, _cols, mat->get_rows(), mat->get_cols());
        return false;
    }

    uint size = get_size();

    T* data_a = get_data();
    T* data_b = mat->get_data();

    for(uint i = 0; i < size; i++){
        data_a[i] -= data_b[i];
    }

    return true;
}

template<class T>
bool BaseMatrixT<T>::multiply(const T value){
    uint size = get_size();
    T* data = get_data();

    for(uint i = 0; i < size; i++){
        data[i] *= value;
    }

    return true;
}
template<class T>
bool BaseMatrixT<T>::multiply(BaseMatrixT<T>* mat){
    if(_rows != mat->get_rows() || _cols != mat->get_cols()){
        printf("multiply matrix dim Error:[%d-%d][%d-%d]\n", _rows, _cols, mat->get_rows(), mat->get_cols());
        return false;
    }

    uint size = get_size();
    T* data_a = get_data();
    T* data_b = mat->get_data();

    for(uint i = 0; i < size; i++){
        data_a[i] *= data_b[i];
    }

    return true;
}

template<class T>
bool BaseMatrixT<T>::division(const T value){

    uint size   = get_size();
    T* data     = get_data();

    for(uint i = 0; i < size; i++){
        data[i] /=  value;
    }

    return true;
}

template<class T>
bool BaseMatrixT<T>::sigmoid(){

    uint size   = get_size();
    T one       = static_cast<T>(1);
    T* data     = get_data();

    /*
    for(uint i = 0; i < size; i++){
        data[i] = one / (one + std::exp(-data[i]));
    }
    */

    uint num_thread = get_num_thread(size);
    if(num_thread == 1){
        for(uint i = 0; i < size; i++){
            data[i] = one / (one + std::exp(-data[i]));
        }
    }else{
        uint block_size = (size % num_thread == 0) ? size / num_thread : (size / num_thread + 1);

        printf("threads[%d]-block size[%d]\n", num_thread, block_size);

        std::vector<std::thread> threads(num_thread);
        for(uint i = 0; i < num_thread; i++){
            threads[i] = std::thread([&data, &one](uint start_idx, uint end_idx){
                                            for(uint ti = start_idx; ti < end_idx; ti++){
                                                data[ti] = one / (one + std::exp(-data[ti]));
                                            }
                                        }, i * block_size, std::min(size, (i + 1) * block_size)
                                    );
        }

        for(auto& thread : threads){
            thread.join();
        }
    }
    return true;
}


template class BaseMatrixT<int>;
template class BaseMatrixT<real>;

}//namespace ccma
}//namespace algebra
