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

    if(_rows == 0 && _cols == 0){
        set_data(mat->get_data(), row, col);
        return true;
    }

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
        memset(new_data, 0, sizeof(T)*_cols);

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
        memset(new_data, 0, sizeof(T)*_rows);

        for(int i = 0; i != _rows; i++){
            for(int j = 0; j != _cols; j++){
                new_data[i] += data[i * _cols + j];
            }
        }
        set_shallow_data(new_data, _rows, 1);
    }
}

template<class T>
void BaseMatrixT<T>::expand(uint row_dim, uint col_dim){
    if(row_dim * col_dim > 1){//not allowed 0 and all of 1
        T* data = get_data();
        T* new_data = new T[_rows * _cols * row_dim * col_dim];
        memset(new_data, 0, sizeof(T)*_rows * _cols * row_dim * col_dim);

        uint row = _rows * row_dim;
        uint col = _cols * col_dim;
        for(uint i = 0; i != row; i++){
            for(uint j = 0; j != col; j++){
                new_data[i * col + j] = data[i/row_dim * _cols + j/col_dim];
            }
        }
        set_shallow_data(new_data, row, col);
    }
}

template<class T>
bool BaseMatrixT<T>::convn(ccma::algebra::BaseMatrixT<T>* kernal,
                           uint stride,
                           std::string shape){
    uint kernal_row = kernal->get_rows();
    uint kernal_col = kernal->get_cols();

    uint data_row;
    uint data_col;

    T* data;
    T* new_data;
    T* src_data = get_data();

    if(shape == "full" && kernal_row * kernal_col != 1){//padding 0, size is kernel_row - 1 and kernel_col - 1
        data_row = _rows + 2 * (kernal_row - 1);
        data_col = _cols + 2 * (kernal_col - 1);

        data = new T[data_row * data_col];
        memset(data, 0, sizeof(T) * data_row * data_col);//padding 0
        for(uint i = 0; i != _rows; i++){
            memcpy(&data[(i + kernal_row - 1) * data_col + kernal_col - 1], &src_data[i * _cols], sizeof(T)* _cols);
        }

    }else{//"valid"
        data_row = _rows;
        data_col = _cols;
        data     = src_data;
        if(data_row < kernal_row || data_col < kernal_col){
            printf("Convn error: kernel dim large than mat.\n");
            return false;
        }
    }

    uint conv_row = (data_row - kernal_row) % stride == 0 ? (data_row - kernal_row) / stride + 1 : (data_row - kernal_row) / stride + 2;
    uint conv_col = (data_col - kernal_col) % stride == 0 ? (data_col - kernal_col) / stride + 1 : (data_col - kernal_col) / stride + 2;

    new_data = new T[conv_row * conv_col];

//    kernal->flip180();
    T* kernal_data = kernal->get_data();

    for(uint i = 0; i != conv_row; i++){
        for(uint j = 0; j != conv_col; j++){
            T sum = 0;
            for(uint k_i = 0; k_i != kernal_row; k_i++){
                uint row = i * stride + k_i;
                for(uint k_j = 0; k_j != kernal_col; k_j++){
                    uint col = j * stride + k_j;
                    //skip out of range, in other word, fill 0
                    if(row < data_row && col < data_col){
                        T a = data[row * data_col + col];
                        T b = kernal_data[k_i * kernal_col + k_j];
                        if(a != 0 && b != 0){
                            sum += a * b;
                        }
                    }
                }
            }
            new_data[i * conv_col + j] = sum;
        }
    }
    this->set_shallow_data(new_data, conv_row, conv_col);
  //  kernal->flip180();

    return true;
}

template<class T>
void BaseMatrixT<T>::flipdim(uint dim){
    T* data = new T[_rows * _cols];
    T* src_data = this->get_data();

    for(uint i = 0; i != _rows; i++){
        for(uint j = 0; j != _cols; j++){
            uint idx;
            if(dim == 1){//row convert
                int new_row = i + 1 - _rows;
                if(new_row < 0){
                    new_row = -new_row;
                }
                idx = new_row * _cols + j;
            }else{//col convert
                int new_col = j + 1 - _cols;
                if(new_col < 0){
                    new_col = -new_col;
                }
                idx = i * _cols + new_col;
            }
            data[idx] = src_data[i * _cols + j];
        }
    }
    this->set_shallow_data(data, _rows, _cols);
}

template<class T>
void BaseMatrixT<T>::flip180(){
    T* data = new T[_rows * _cols];
    T* src_data = this->get_data();

    for(uint i = 0; i != _rows; i++){
        for(uint j = 0; j != _cols; j++){
            int new_row = i + 1 - _rows;
            if(new_row < 0){
                new_row = -new_row;
            }
            int new_col = j + 1 - _cols;
            if(new_col < 0){
                new_col = -new_col;
            }
            data[new_row * _rows + new_col] = src_data[i * _cols + j];
        }
    }
    this->set_shallow_data(data, _rows, _cols);
}


template class BaseMatrixT<int>;
template class BaseMatrixT<real>;

}//namespace ccma
}//namespace algebra
