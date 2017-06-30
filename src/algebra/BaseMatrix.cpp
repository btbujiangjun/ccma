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
void BaseMatrixT<T>::outer(BaseMatrixT<T>* mat){
	uint size1= get_size();
	uint size2 = mat->get_size();
	T* data1 = this->get_data();
	T* data2 = mat->get_data();

	uint k = 0;
	T* data = new T[size1 * size2];
	for(uint i = 0; i != size1; i++){
		for(uint j = 0; j != size2; j++){
			data[k++] = data1[i] * data2[j];
		}
	}
	this->set_shallow_data(data, size1, size2);
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
void BaseMatrixT<T>::pow(const T exponent){
    uint size   = get_size();
    T* data     = get_data();

    for(uint i = 0; i != size; i++){
        data[i] = std::pow(data[i], exponent);
    }
}

template<class T>
void BaseMatrixT<T>::log(){
    uint size   = get_size();
    T* data     = get_data();

    for(uint i = 0; i != size; i++){
        data[i] =  std::log(data[i]);
    }
}


template<class T>
void BaseMatrixT<T>::exp(){
    uint size   = get_size();
    T* data     = get_data();

    for(uint i = 0; i != size; i++){
        data[i] =  std::exp(data[i]);
    }
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

/*
 * softmax
 */
template<class T>
void BaseMatrixT<T>::softmax(){
	T e_sum = 0;
	uint size = get_size();
	T* data = new T[size];
	T* src_data = this->get_data();
	for(uint i = 0; i != size; i++){
		data[i] = std::exp(src_data[i]);
		e_sum += data[i];
	}
	for(uint i = 0; i != size; i++){
		src_data[i] = data[i] / e_sum;
	}
    delete[] data;
}


template<class T>
void BaseMatrixT<T>::tanh(){
	uint size = get_size();
	T* data = this->get_data();
    for(uint i = 0; i != size; i++){
        data[i] = std::tanh(data[i]);
    }
}


template<class T>
void BaseMatrixT<T>::relu(){
	uint size = get_size();
	T* data = this->get_data();
	for(uint i = 0; i != size; i++){
		if(data[i] < 0){
			data[i] = 0;
		}
	}
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
BaseMatrixT<int>* BaseMatrixT<T>::argmax(const uint axis){
    T* data = this->get_data();
    uint size = (axis == 0)? _rows : _cols;
    int* idx_data = new int[size];
    for(uint i = 0; i != size; i++){
        T max_value = 0;
        uint max_idx = 0;
	
        uint end_idx = (axis == 0) ? _cols : _rows;
        for(int j = 0; j != end_idx; j++){
            T value = (axis == 0) ? data[i * _cols + j] : data[j * _cols + i];
            if( j == 0 || value > max_value){
                max_value = value;
                max_idx = j;
            }
        }
        idx_data[i] = max_idx;
    }

    uint rows = (axis == 0) ? size : 1;
    uint cols = (axis == 0) ? 1 : size;

    auto mat = new DenseMatrixT<int>();
    mat->set_shallow_data(idx_data, rows, cols);

    return mat;
}


template<class T>
uint BaseMatrixT<T>::argmax(const uint id, const uint axis){
	T max_value = 0;
	uint max_idx = 0;
	T* data = this->get_data();

	uint end_idx = (axis == 0) ? _cols : _rows;
	for(uint i = 0; i != end_idx; i++){
		T value = (axis == 0) ? data[id * _cols + i] : data[i * _rows + id];
		if( i == 0 || value > max_value){
			max_value = value;
			max_idx = i;
		}
	}
	return max_idx;
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

    }else if(shape == "valid"){//"valid"
        data_row = _rows;
        data_col = _cols;
        data     = src_data;
        if(data_row < kernal_row || data_col < kernal_col){
            printf("Convn error: kernel dim large than mat.\n");
            return false;
        }
    }else{
        printf("Convn error: not support shape [%s]\n", shape.c_str());
        return false;
    }

    uint conv_row = (data_row - kernal_row) % stride == 0 ? (data_row - kernal_row) / stride + 1 : (data_row - kernal_row) / stride + 2;
    uint conv_col = (data_col - kernal_col) % stride == 0 ? (data_col - kernal_col) / stride + 1 : (data_col - kernal_col) / stride + 2;

    new_data = new T[conv_row * conv_col];

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

    if(shape == "full"){
    	delete[] data;
    }

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


template<class T>
bool BaseMatrixT<T>::get_col_data(const uint col_id, BaseMatrixT<T>* out_mat){
    if(col_id >= 0 && col_id < this->_cols){
        T* data = new T[this->_rows];
        T* src_data = this->get_data();
        for(uint i = 0; i != this->_rows; i++){
            data[i] = src_data[i * this->_cols + col_id];
        }
        out_mat->set_shallow_data(data, this->_rows, 1);
        return true;
    }
    return false;
}

template<class T>
bool BaseMatrixT<T>::set_col_data(const uint col_id, BaseMatrixT<T>* mat){
    uint mat_col = mat->get_cols();
	if(this->_rows == mat->get_rows() && this->_cols >= (col_id + mat_col)){
        T* data = this->get_data();
        T* mat_data = mat->get_data();

        for(uint i = 0; i != this->_rows; i++){
		    memcpy(&data[i * this->_cols + col_id], &mat_data[i * mat_col], sizeof(T) * mat_col);
        }
		return true;
	}
	return false;
}


template<class T>
void BaseMatrixT<T>::reset(const T value){
    T* data = this->get_data();
    uint size = get_size();
    if(value == (T)0 || value == (T)-1){
       memset(data, value, sizeof(T) * size); 
    }else{
        for(uint i = 0; i != size; i++){
            data[i] =  value;
        }
    }
}

template<class T>
void BaseMatrixT<T>::reset(const T value,
                           uint rows,
                           uint cols){
    uint size = rows * cols;
    if(get_size() == size){
        reset(value);
        reshape(rows, cols);
    }else{
        T* data = new T[size];
        if(value == (T)0 || value == (T)-1){
           memset(data, value, sizeof(T) * size); 
        }else{
            for(uint i = 0; i != size; i++){
                data[i] =  value;
            }
        }
        this->set_shallow_data(data, rows, cols);
    }
}

template class BaseMatrixT<int>;
template class BaseMatrixT<real>;

}//namespace ccma
}//namespace algebra
