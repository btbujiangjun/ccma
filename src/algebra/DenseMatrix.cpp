/**********************************************
* Author: Jun Jiang - jiangjun4@sina.com
* Created: 2016-11-22 11:30
* Last modified:	2016-12-15 10:54
* Filename:		DenseMatrixT.cpp
* Description: Dense Data Structure of training/predict data
**********************************************/

#include "algebra/BaseMatrix.h"
#include <stdio.h>

namespace ccma{
namespace algebra{

template<class T>
DenseMatrixT<T>::DenseMatrixT() : BaseMatrixT<T>(){
    _data = nullptr;
    _cache_matrix_det = ccma::utils::get_max_value<T>();
    _cache_to_string = nullptr;
}

template<class T>
DenseMatrixT<T>::DenseMatrixT(const uint rows, const uint cols) : BaseMatrixT<T>(rows, cols){
    _data = new T[rows * cols];
    memset(_data, 0, sizeof(T)* rows * cols);

    _cache_matrix_det = ccma::utils::get_max_value<T>();
    _cache_to_string = nullptr;
}


template<class T>
DenseMatrixT<T>::DenseMatrixT(const T* data,
                              const uint rows,
                              const uint cols):BaseMatrixT<T>(rows, cols){
    _data = new T[rows * cols];
    memcpy(_data, data, sizeof(T) * rows * cols);

    _cache_matrix_det = ccma::utils::get_max_value<T>();
    _cache_to_string = nullptr;
}


template<class T>
DenseMatrixT<T>::~DenseMatrixT(){
    clear_matrix();
}

template<class T>
void DenseMatrixT<T>::clone(BaseMatrixT<T>* out_mat){
    out_mat->set_data(_data, this->_rows, this->_cols);
}

template<class T>
void DenseMatrixT<T>::clear_matrix(){
    if(_data != nullptr){
        delete[] _data;
        _data = nullptr;

        this->_rows = 0;
        this->_cols = 0;

        clear_cache();
    }
}

template<class T>
void DenseMatrixT<T>::set_data(const T* data,
                               const uint rows,
                               const uint cols){
    if(rows * cols != this->_rows * this->_cols){
        if(_data != nullptr){
            delete[] _data;
            _data = nullptr;
        }
        _data = new T[rows * cols];
    }
    memcpy(_data, data, sizeof(T)* rows * cols);
    this->_rows = rows;
    this->_cols = cols;
    clear_cache();
}

template<class T>
void DenseMatrixT<T>::set_shallow_data(T* data,
                                       const uint rows,
                                       const uint cols){
    if(_data != nullptr){
        delete[] _data;
        _data = nullptr;

        clear_cache();
    }

    _data = data;
    this->_rows = rows;
    this->_cols = cols;
}


template<class T>
bool DenseMatrixT<T>::get_row_data(const int row, BaseMatrixT<T>* out_mat){
    int r = row, c = 0;
    if(check_range(&r, &c)){
        if(out_mat->get_size() == this->_cols){
            memcpy(out_mat->get_data(), &_data[r * this->_cols], sizeof(T) * this->_cols);
            out_mat->reshape(1, this->_cols);
        }else{
            T* data = new T[this->_cols];
            memcpy(data, &_data[r * this->_cols], sizeof(T) * this->_cols);
            out_mat->set_shallow_data(data, 1, this->_cols);
        }
        return true;
    }
    return false;
}

template<class T>
bool DenseMatrixT<T>::set_row_data(const uint row_id, BaseMatrixT<T>* mat){
	if(this->_cols == mat->get_cols() && this->_rows >= (row_id + mat->get_rows())){
		memcpy(&_data[row_id * this->_cols], mat->get_data(), sizeof(T) * mat->get_size());
		return true;
	}
	printf("set_row_data error:[%d-%d][%d-%d]\n", this->_cols, mat->get_cols(), this->_rows, row_id + mat->get_rows());
	return false;
}

template<class T>
bool DenseMatrixT<T>::insert_row_data(const int row_id, BaseMatrixT<T>* mat){
    int r = row_id;
    if(this->_rows > 0 && this->_cols != mat->get_cols()){
        return false;
    }

    if(this->_rows == 0){
        this->_cols = mat->get_cols();
    }

    if(r < 0){
        r += this->_rows;
    }
    if(r < 0){
        r = 0;
    }

    if(r > this->_rows){
        r = this->_rows;
    }

    T* data = new T[(this->_rows + mat->get_rows()) * this->_cols];
    if(r > 0){
        memcpy(data, _data, sizeof(T) * this->_cols * r);
    }
    memcpy(&data[r * this->_cols], mat->get_data(), sizeof(T) * mat->get_rows() * mat->get_cols());
    if(r < this->_rows){
        memcpy(&data[(r + mat->get_rows()) * this->_cols], &_data[this->_cols * r], sizeof(T) * (this->_rows - r) * this->_cols);
    }

    if(_data != nullptr){
        delete[] _data;
        _data = nullptr;

        clear_cache();
    }

    _data = data;
    this->_rows += mat->get_rows();
    this->_cols = mat->get_cols();

    return true;
}

template<class T>
bool DenseMatrixT<T>::extend(BaseMatrixT<T>* mat, bool col_dim){
    uint row = mat->get_rows();
    uint col = mat->get_cols();

    if(this->_rows == 0){
        T* data = new T[row * col];
        memcpy(data, mat->get_data(), sizeof(T) * row * col);
        this->set_shallow_data(data, row, col);
        return true;
    }

    if(col_dim){
        if(this->_rows != row){
            return false;
        }

        T* data = new T[this->_rows * (this->_cols + col)];
        for(uint i = 0; i < this->_rows; i++){
            memcpy(&data[i * (this->_cols + col)], &_data[i * this->_cols], sizeof(T) * this->_cols);
            memcpy(&data[(i+1) * (this->_cols + col) - this->_cols], &mat->get_data()[i * col], sizeof(T) * col);
        }

        set_shallow_data(data, this->_rows, (this->_cols + col));

        return true;
    }else{
        if(this->_cols != col){
            return false;
        }
        T* data = new T[(this->_rows + row) * this->_cols];
        memcpy(data, _data, sizeof(T) * this->_rows * this->_cols);
        memcpy(&data[this->_rows * this->_cols], mat->get_data(), sizeof(T)* row * col);
        set_shallow_data(data, this->_rows + row, col);
        return true;
    }
}


/* 
 * A(m,p) * B(p,n) = C(m,n)
 */
template<class T>
bool DenseMatrixT<T>::dot(BaseMatrixT<T>* mat){

    uint row_a = this->_rows;
    uint col_a = this->_cols;
    uint row_b = mat->get_rows();
    uint col_b = mat->get_cols();

    if(col_a != row_b){
        printf("Dot Matrix Dim Error[%d:%d][%d:%d]\n", row_a,col_a, row_b, col_b);
        return false;
    }

    T* data = new T[row_a * col_b];
    T zero = static_cast<T>(0);
    T value;
    uint a_init_idx, a_idx, row_cursor = 0;
    uint j, k;
    T* data_a = _data;
    T* data_b = mat->get_data();

    for(uint i = 0; i != row_a; i++){
        a_init_idx = i * col_a;
        for(j = 0; j != col_b; j++){
            value = zero;
            a_idx = a_init_idx;
            for(k = 0; k != col_a; k++){
                //a_idx  == i * col_a + k
                value += data_a[a_idx++] * data_b[k * col_b + j];
            }
            data[row_cursor++] = value;
        }
    }

    this->set_shallow_data(data, row_a, col_b);

    return true;
}

template<class T>
T DenseMatrixT<T>::sum() const{
    T value = static_cast<T>(0);
    uint size = this->get_size();
    for(uint i = 0; i != size; i++){
        value += _data[i];
    }
    return value;
}


template<class T>
bool DenseMatrixT<T>::swap(const uint aa_row,
                           const uint aa_col,
                           const uint bb_row,
                           const uint bb_col){
    int a_row = aa_row, a_col = aa_col, b_row = bb_row, b_col = bb_col;
    if(!check_range(&a_row, &a_col) || !check_range(&b_row, &b_col)){
        return false;
    }

    T value = get_data(a_row, a_col);
    set_data(get_data(b_row, b_col),a_row, a_col);
    set_data(value, b_row, b_col);

    clear_cache();

    return true;
}
template<class T>
bool DenseMatrixT<T>::swap_row(const uint a, const uint b){

    if(a == b){
        return true;
    }

    if(a >= this->_rows || b >= this->_rows){
        return false;
    }

    T* data = new T[this->_cols];
    memcpy(data, &_data[a * this->_cols], sizeof(T) * this->_cols);
    memcpy(&_data[a * this->_cols], &_data[b * this->_cols], sizeof(T) * this->_cols);
    memcpy(&_data[b * this->_cols], data, sizeof(T) * this->_cols);

    delete[] data;
    data = nullptr;

    clear_cache();

    return true;
}
template<class T>
bool DenseMatrixT<T>::swap_col(const uint a, const uint b){
    if(a >= this->_cols || b >= this->_cols){
        return false;
    }

    if(a == b){
        return true;
    }

    for(uint i = 0; i != this->_rows; i++){
        swap(i, a, i, b);
    }

    return true;
}

template<class T>
BaseMatrixT<T>* DenseMatrixT<T>::transpose(){
    uint row = this->_rows;
    uint col = this->_cols;
    if(row == 1 || col == 1){
        this->_rows = col;
        this->_cols = row;
    }else{
        T* data = new T[row * col];
        uint new_data_idx = 0;

        for(uint i = 0; i != col; i++){
            for(uint j = 0; j != row; j++){
                data[new_data_idx++] = _data[j * col + i];
            }
        }

        set_shallow_data(data, col, row);
    }
    return this;
}

template<class T>
void DenseMatrixT<T>::add_x0(){
    add_x0(this);
}
template<class T>
void DenseMatrixT<T>::add_x0(BaseMatrixT<T>* result){
    T* data = new T[this->_rows * this->_cols];
    for(uint i = 0; i < this->_rows; i++){
        data[i * (this->_cols + 1)] = 1;
        memcpy(&data[i * (this->_cols + 1) + 1], &this->_data[i * this->_cols], sizeof(T) * this->_cols);
    }

    result->set_shallow_data(data, this->_rows, this->_cols + 1);
}

template<class T>
bool DenseMatrixT<T>::det(T* result){

    //must be phalanx N*N
    if(this->_rows != this->_cols){
        return false;
    }

    if(_cache_matrix_det != ccma::utils::get_max_value<T>()){
        *result = _cache_matrix_det;
        return true;
    }

    if(this->_rows == 1){
        _cache_matrix_det = this->_data[0];
        *result = _cache_matrix_det;
        return true;
    }

    /*
     *二阶对角,高阶画圈
     */
    if(this->_rows == 2){
        _cache_matrix_det = get_data(0, 0) * get_data(1, 1) - get_data(0, 1) * get_data(1, 0);
        *result = _cache_matrix_det;
        return true;
    }

    T sum1 = static_cast<T>(0), sum2 = static_cast<T>(0);
    for(uint i = 0; i < this->_cols; i++){
        T s = static_cast<T>(1);
        for(uint j = 0; j < this->_rows; j++){
            s *= get_data(j, (i+ j) % this->_rows);
        }
        sum1 += s;
    }

    for(uint i = 0; i < this->_cols; i++){
        T s = static_cast<T>(1);
        for(uint j = 0; j < this->_rows; j++){
            int row = i - j;
            if(row < 0){
                row += this->_rows;
            }
            s *= get_data(j, row);
        }
        sum2 += s;
    }

    _cache_matrix_det = sum1 - sum2;
    *result = _cache_matrix_det;

    return true;
}

template<class T>
real DenseMatrixT<T>::mean(){
    uint size = this->get_size();
    if(size == 0){
        return 0.0;
    }
    T sum = 0;
    for(uint i = 0; i != size; i++){
        sum += _data[i];
    }
    return static_cast<real>(sum)/size;
}

template<class T>
real DenseMatrixT<T>::mean(uint col){
    if(this->_rows == 0 or col >= this->_cols){
        return 0.0;
    }

    T sum = 0;
    for(uint i = 0; i != this->_rows; i++){
        sum += _data[i * this->_cols + col];
    }

    return static_cast<real>(sum) / this->_rows;
}

template<class T>
real DenseMatrixT<T>::var(){
    uint size =  this->get_size();

    if(size == 0){
        return 0.0;
    }

    real mean_value = mean();
    real var_sum = 0.0;
    for(uint i = 0; i < size; i++){
        var_sum += std::pow(this->_data[i] - mean_value, 2);
    }

    return var_sum / size;
}
template<class T>
real DenseMatrixT<T>::var(uint col){
    if(this->_rows == 0 or col >= this->_cols){
        return 0.0;
    }

    real mean_value = mean(col);
    real var_sum = 0.0;
    for(uint i = 0; i < this->_rows; i++){
        var_sum += std::pow(_data[i * this->_cols + col] - mean_value, 2);
    }

    return (var_sum / this->_rows);
}

template<class T>
bool DenseMatrixT<T>::inverse(BaseMatrixT<real>* result){
    if(this->_rows != this->_cols){
        return false;
    }

    uint size = this->get_size();

    //copy src matrix
    auto extend_mat = new DenseMatrixT<real>();
    auto data = new real[size];
    if(typeid(T) == typeid(real)){
        memcpy(data, _data, sizeof(T) * size);
    }else{
        for(uint i = 0; i != size ; i++){
            data[i] = static_cast<real>(_data[i]);
        }
    }
    extend_mat->set_shallow_data(data, this->_rows, this->_cols);

    //extend eye matrix
    auto eye_mat = new DenseEyeMatrixT<real>(this->_rows);
    extend_mat->extend(eye_mat, true);
    delete eye_mat;

    //adjust row
    uint extend_mat_rows = extend_mat->get_rows();
    uint extend_mat_cols = extend_mat->get_cols();
    auto zero = static_cast<T>(0);

    for(uint i = 0; i != extend_mat_rows; i++){
        if(extend_mat->get_data(i, i) == zero){
            uint j;
            for(j = 0; j != extend_mat_rows; j++){
                if(extend_mat->get_data(j, i) != zero){
                    extend_mat->swap_row(i, j);
                    break;
                }
            }
            if(j >= extend_mat_rows){//every element is zero in col i, cannot be inverse.
                return false;
            }
        }
    }

    //calc extend matrix
    for(uint i = 0; i != extend_mat_rows; i++){

        //all element div the first element,to make diagonal elemnt is 1
        real diagonal_element = extend_mat->get_data(i, i);
        for(uint j = 0; j < extend_mat_cols; j++){
            extend_mat->set_data(extend_mat->get_data(i, j)/diagonal_element, i, j);
        }

        //to make the element of other rows in col i is 0
        for(uint m = 0; m != extend_mat_rows; m++){
            if(m == i){//skip itself
                continue;
            }
            real element = extend_mat->get_data(m, i);
            for(uint n = 0; n != extend_mat_cols; n++){
                extend_mat->set_data(extend_mat->get_data(m, n) - extend_mat->get_data(i, n) * element, m, n);
            }
        }
    }

    //calc inverse matrix
    auto new_data = new real[size];
    auto extend_data = extend_mat->get_data();
    for(uint i = 0; i != extend_mat_rows; i++){
        memcpy(&new_data[i * this->_cols], &extend_data[i * extend_mat_cols + this->_cols], sizeof(real)*this->_cols);
    }
    delete extend_mat;
    result->set_shallow_data(new_data, this->_rows, this->_cols);

    return true;
}

template<class T>
bool DenseMatrixT<T>::operator==(BaseMatrixT<T>* mat) const{
    if(this->_rows == mat->get_rows() && this->_cols == mat->get_cols()){
        uint size = this->get_size();
        for(uint i = 0; i != size; i++){
            if(_data[i] != mat->get_data(i)){
                return false;
            }
        }
        return true;
    }
    return false;
}


template<class T>
void DenseMatrixT<T>::display(const std::string& split){
    printf("[%d*%d][\n", this->_rows, this->_cols);
    for(uint i = 0; i < this->_rows; i++){
        printf("row[%d][",i);
        for(uint j = 0; j < this->_cols; j++){
            printf("%s", std::to_string(get_data(i, j)).c_str());
            if(j != this->_cols - 1){
                printf("%s", split.c_str());
            }
        }
        printf("]\n");
    }
    printf("]\n");
}

template<class T>
std::string* DenseMatrixT<T>::to_string(){
    if(_cache_to_string != nullptr){
        return _cache_to_string;
    }
    std::string* str = new std::string("[" + std::to_string(this->_rows) + "*"+std::to_string(this->_cols)+"][");
    for(uint i = 0 ; i < this->_rows; i++){
        str->append("[");
        for(uint j = 0; j < this->_cols; j++){
            if(j > 0){
                str->append(",");
            }
            str->append(std::to_string(get_data(i, j)));
        }
        str->append("]");
    }
    str->append("]");
    _cache_to_string = str;
    return _cache_to_string;
}

template<class T>
LabeledDenseMatrixT<T>::LabeledDenseMatrixT(){
    _labels = nullptr;
    _feature_names = nullptr;

    _cache_shannon_entropy = 0.0;
    _cache_label_cnt_map = new CCMap<T>();
    _cache_feature_cnt_map = new std::unordered_map<uint, CCMap<T>* >();
}

template<class T>
LabeledDenseMatrixT<T>::LabeledDenseMatrixT(const T* data,
                                            const T* labels,
                                            const uint rows,
                                            const uint cols) : DenseMatrixT<T>(data, rows, cols){
    _labels = new T[rows];
    memcpy(_labels, labels, sizeof(T) * rows);

    _feature_names = new uint[cols];
    for(uint i = 0; i < cols; i++){
        _feature_names[i] = i;
    }

    _cache_shannon_entropy = 0.0;
    _cache_label_cnt_map = new CCMap<T>();
    _cache_feature_cnt_map = new std::unordered_map<uint, CCMap<T>* >();
}

template<class T>
LabeledDenseMatrixT<T>::LabeledDenseMatrixT(const T* data,
                                            const T* labels,
                                            const uint* feature_names,
                                            const uint rows,
                                            const uint cols) : DenseMatrixT<T>(data, rows, cols){
    _labels = new T[rows];
    memcpy(_labels, labels, sizeof(T) * rows);

    _feature_names = new uint[cols];
    memcpy(_feature_names, feature_names, sizeof(uint) * cols);

    _cache_shannon_entropy = 0.0;
    _cache_label_cnt_map = new CCMap<T>();
    _cache_feature_cnt_map = new std::unordered_map<uint, CCMap<T>* >();
}



template<class T>
LabeledDenseMatrixT<T>::~LabeledDenseMatrixT(){
    if(_labels != nullptr){
        delete[] _labels;
        _labels = nullptr;
    }

    if(_feature_names != nullptr){
        delete[] _feature_names;
        _feature_names = nullptr;
    }

    clear_cache();

    if(_cache_label_cnt_map != nullptr){
        delete _cache_label_cnt_map;
    }

    if(_cache_feature_cnt_map != nullptr){
        delete _cache_feature_cnt_map;
    }
}

template<class T>
void LabeledDenseMatrixT<T>::get_data_matrix(DenseMatrixT<T>* out_mat){
    return out_mat->set_data(this->_data, this->_rows, this->_cols);
}

template<class T>
void LabeledDenseMatrixT<T>::get_labels(DenseMatrixT<T>* out_mat){
    return out_mat->set_data(_labels, this->_rows, 1);
}

template<class T>
void LabeledDenseMatrixT<T>::set_data(const T* data,
                                      const T* labels,
                                      const uint rows,
                                      const uint cols){
    uint* feature_names = new uint[cols];
    for(uint i = 0; i < cols; i++){
        feature_names[i] = i;
    }

    set_data(data, labels, feature_names, rows, cols);

    delete[] feature_names;
    feature_names = nullptr;
}

template<class T>
bool LabeledDenseMatrixT<T>::get_row_data(const int row_id, LabeledDenseMatrixT<T>* row_data){

    int id = row_id;
    if(row_id < 0){
        id += this->_rows;
    }

    if(id < 0 || id >= this->_rows){
        return false;
    }

    T* data = new T[this->_rows];
    T* label = new T[1];
    label[0] = get_label(id);

    row_data->set_shallow_data(data, label, 1, this->_cols);

    return row_data;
}

template<class T>
void LabeledDenseMatrixT<T>::set_row_data(LabeledDenseMatrixT<T>* mat, const int row_id){
    int id = row_id;
    if(id < 0){
        id += this->_rows;
    }

    auto data_mat = new DenseMatrixT<T>();
    mat->get_data_matrix(data_mat);
    DenseMatrixT<T>::set_row_data(id, data_mat);
    delete data_mat;

    auto label_mat = new DenseMatrixT<T>();
    mat->get_labels(label_mat);
    T* labels = new T[mat->get_rows() + this->_rows];
    if(id > 0){
        memcpy(labels, _labels, sizeof(T) * id);
    }
    memcpy(&labels[id], label_mat->get_data(), sizeof(T) * data_mat->get_rows());
    if(id < this->_rows){
        memcpy(&labels[id + label_mat->get_rows()], &label_mat->get_data()[id], sizeof(T) * (this->_rows - id));
    }
    delete label_mat;

    if(_labels){
        delete[] _labels;
    }
    _labels = labels;
}

template<class T>
void LabeledDenseMatrixT<T>::set_data(const T* data,
                                      const T* labels,
                                      const uint* feature_names,
                                      const uint rows,
                                      const uint cols){
    DenseMatrixT<T>::set_data(data, rows, cols);

    if(_labels != nullptr){
        delete[] _labels;
    }
    _labels = new T[rows];
    memcpy(_labels, labels, sizeof(T) * rows);

    if(_feature_names != nullptr){
        delete[] _feature_names;
    }
    _feature_names = new uint[cols];
    memcpy(_feature_names, feature_names, cols);

    clear_cache();
}

template<class T>
void LabeledDenseMatrixT<T>::set_shallow_data(T* data,
                                              T* labels,
                                              const uint rows,
                                              const uint cols){
    uint* feature_names = new uint[cols];
    for(uint i = 0; i < cols; i++){
        feature_names[i] = i;
    }

    set_shallow_data(data, labels, feature_names, rows, cols);
}

template<class T>
void LabeledDenseMatrixT<T>::set_shallow_data(T* data,
                                              T* labels,
                                              uint* feature_names,
                                              const uint rows,
                                              const uint cols){
    DenseMatrixT<T>::set_shallow_data(data, rows, cols);

    if(_labels != nullptr){
        delete[] _labels;
    }
    _labels = labels;

    if(_feature_names != nullptr){
        delete[] _feature_names;
    }
    _feature_names = feature_names;

    clear_cache();
}

template<class T>
void LabeledDenseMatrixT<T>::clear_matrix(){
    DenseMatrixT<T>::clear_matrix();
    delete[] _labels;
    _labels = nullptr;
    clear_cache();
}

template<class T>
real LabeledDenseMatrixT<T>::get_shannon_entropy(){
    if(_cache_shannon_entropy != 0.0){
        return _cache_shannon_entropy;
    }

    real ent = 0.0;
    CCMap<T>* vm = get_label_cnt_map();
    typename CCMap<T>::const_iterator it;
    for(it = vm->begin(); it != vm->end(); it++){
        real prob = (real)it->second/this->_rows;
        ent -= prob * (log(prob)/log(2));
    }
    _cache_shannon_entropy = ent;
    return ent;
}

template<class T>
bool LabeledDenseMatrixT<T>::split(uint feature_idx,
                                   T split_value,
                                   LabeledDenseMatrixT<T>* sub_mat){
    CCMap<T>* feature_cnt_map = get_feature_cnt_map(feature_idx);
    typename CCMap<T>::const_iterator it;
    it = feature_cnt_map->find(split_value);
    if(it == feature_cnt_map->end()){
        return false;
    }

    uint new_rows = it->second;
    uint new_cols = this->_cols -1;

    T* new_data = new T[new_rows * new_cols];
    T* new_label = new T[new_rows];

    uint new_data_idx = 0;
    uint new_label_idx = 0;

    for(uint i = 0; i < this->_rows; i++){
        if(this->get_data(i, feature_idx) !=  split_value){
            continue;
        }

        for(uint j = 0; j < this->_cols; j++){
            if(j != feature_idx){
                new_data[new_data_idx++] = this->get_data(i, j);
            }
        }

        new_label[new_label_idx++] = _labels[i];
    }

    uint* new_feature_names = new uint[this->_cols - 1];
    if(feature_idx > 0){
        memcpy(new_feature_names, _feature_names, feature_idx * sizeof(uint));
    }
    if(feature_idx < this->_cols - 1){//skip feature_idx
        memcpy(&new_feature_names[feature_idx], &_feature_names[feature_idx+1], (this->_cols -1 - feature_idx) * sizeof(uint));
    }

    if(sub_mat == nullptr){
        sub_mat = new LabeledDenseMatrixT<T>();
    }
    sub_mat->set_shallow_data(new_data, new_label, new_feature_names, new_rows, new_cols);

    return true;
}

template<class T>
bool LabeledDenseMatrixT<T>::binary_split(uint feature_idx,
                                          T split_value,
                                          LabeledDenseMatrixT<T>* lt_mat,
                                          LabeledDenseMatrixT<T>* gt_mat){

    uint lt_rows = 0, gt_rows = 0;
    for(uint i = 0; i < this->_rows; i++){
        if(this->get_data(i, feature_idx) <= split_value){
            lt_rows++;
        }
    }
    gt_rows = this->_rows - lt_rows;

    T* lt_data = new T[lt_rows * this->_cols];
    T* lt_labels = new T[lt_rows];

    T* gt_data = new T[gt_rows * this->_cols];
    T* gt_labels = new T[gt_rows];

    uint lt_idx = 0;
    for(uint i = 0; i < this->_rows; i++){
        if(this->get_data(i, feature_idx) <=  split_value){
            memcpy(&lt_data[lt_idx * this->_cols], &this->_data[i * this->_cols], this->_cols * sizeof(T));
            lt_labels[lt_idx] = this->_labels[i];
            lt_idx++;
        }else{
            memcpy(&gt_data[(i - lt_idx) * this->_cols], &this->_data[i * this->_cols], this->_cols * sizeof(T));
            gt_labels[i - lt_idx] = this->_labels[i];
        }
    }

    if(lt_mat == nullptr){
        lt_mat = new LabeledDenseMatrixT<T>();
    }
    if(gt_mat == nullptr){
        gt_mat = new LabeledDenseMatrixT<T>();
    }

    uint* lt_feature_names = new uint[this->_cols];
    uint* gt_feature_names = new uint[this->_cols];
    memcpy(lt_feature_names, _feature_names, sizeof(uint) * this->_cols);
    memcpy(gt_feature_names, _feature_names, sizeof(uint) * this->_cols);

    lt_mat->set_shallow_data(lt_data, lt_labels, lt_feature_names, lt_rows, this->_cols);
    gt_mat->set_shallow_data(gt_data, gt_labels, gt_feature_names, gt_rows, this->_cols);

    return true;
}

template<class T>
CCMap<T>* LabeledDenseMatrixT<T>::get_label_cnt_map(){
    if(_cache_label_cnt_map->size() > 0){
        return _cache_label_cnt_map;
    }

    typename CCMap<T>::iterator it;
    for(uint i = 0; i < this->_rows; i++){
        it = _cache_label_cnt_map->find(_labels[i]);
        if(it == _cache_label_cnt_map->end()){
            _cache_label_cnt_map->insert(std::make_pair(_labels[i], 1));
        }else{
            it->second += 1;
        }
    }

    return _cache_label_cnt_map;
}
template<class T>
bool LabeledDenseMatrixT<T>::is_unique_label(){
    CCMap<T>* label_cnt_map = get_label_cnt_map();
    return (label_cnt_map->size() == 1);
}

template<class T>
real LabeledDenseMatrixT<T>::label_mean(){
    if(this->_rows == 0){
        return 0.0;
    }

    real sum = 0.0;
    for(uint i = 0; i < this->_rows; i++){
        sum += _labels[i];
    }

    return (sum / this->_rows);
}

template<class T>
real LabeledDenseMatrixT<T>::label_var(){
    if(this->_rows == 0){
        return 0.0;
    }

    real mean = label_mean();
    real var_sum = 0.0;
    for(uint i = 0; i < this->_rows; i++){
        var_sum += std::pow(this->_labels[i] - mean, 2);
    }

    return (var_sum / this->_rows);
}

template<class T>
CCMap<T>* LabeledDenseMatrixT<T>::get_feature_cnt_map(uint feature_idx){
    typename std::unordered_map<uint, CCMap<T>*>::iterator it;
    it = _cache_feature_cnt_map->find(feature_idx);
    if(it != _cache_feature_cnt_map->end()){
        return it->second;
    }

    CCMap<T>* feature_cnt_map = new CCMap<T>();
    if(feature_idx >= this->_cols){
        return feature_cnt_map;
    }

    typename CCMap<T>::iterator itv;
    for(uint i = 0; i < this->_rows; i++){
        T data = this->get_data(i, feature_idx);
        itv = feature_cnt_map->find(data);
        if(itv == feature_cnt_map->end()){
            feature_cnt_map->insert(std::make_pair(data, 1));
        }else{
            itv->second += 1;
        }
    }
    _cache_feature_cnt_map->insert(std::make_pair(feature_idx, feature_cnt_map));

    return feature_cnt_map;
}

template<class T>
void LabeledDenseMatrixT<T>::clear_cache(){
    DenseMatrixT<T>::clear_cache();
    _cache_shannon_entropy = 0;
    if(_cache_label_cnt_map != nullptr){
        _cache_label_cnt_map->clear();
    }

    if(_cache_feature_cnt_map != nullptr){
        CCMap<T>* item;
        typename std::unordered_map<uint, CCMap<T>*>::iterator it;
        it = _cache_feature_cnt_map->begin();
        while(it != _cache_feature_cnt_map->end()){
            item = it->second;
            it = _cache_feature_cnt_map->erase(it);
            delete item;
        }
        _cache_feature_cnt_map->clear();
    }
}

template<class T>
bool LabeledDenseMatrixT<T>::operator==(LabeledDenseMatrixT<T>* mat){
    if(this->_rows == mat->get_rows() && this->_cols == mat->get_cols()){
        for(uint i = 0; i < this->_rows; i++){
            if(this->get_label(i) == mat->get_label(i)){
                return false;
            }
            for(uint j = 0; j < this->_cols; j++){
                if(this->get_data(i, j) != this->get_data(i, j)){
                    return false;
                }
            }
        }
        return true;
    }
    return false;
}

template<class T>
void LabeledDenseMatrixT<T>::display(const std::string& split){
    for(uint i = 0; i < this->_cols; i++){
        printf("col:%d\t", get_feature_name(i));
    }
    printf("label\n");
    for(uint i = 0; i < this->_rows; i++){
        for(uint j = 0; j < this->_cols; j++){
            printf("%s%s", std::to_string(this->get_data(i, j)).c_str(), split.c_str());
        }
        printf("%s\n", std::to_string(this->get_label(i)).c_str());
    }
}


template class DenseMatrixT<int>;
template class DenseMatrixT<real>;

template class DenseColMatrixT<int>;
template class DenseColMatrixT<real>;

template class LabeledDenseMatrixT<int>;
template class LabeledDenseMatrixT<real>;

}//namespace algebra
}//namespace ccma
