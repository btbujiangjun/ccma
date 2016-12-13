/*********************************************
* Author: Jun Jiang - jiangjun4@sina.com
* Created: 2016-12-12 11:25
* Last modified: 2016-12-12 11:25
* Filename: FileOp.cpp
* Description: 
**********************************************/
#include "FileOp.h"

#include <stdio.h>
#include <stdexcept>

namespace ccma{
namespace utils{

inline FILE* FileOp::open_file(const std::string& path, const std::string& open_mode){
    FILE* f = fopen(path.c_str(), open_mode.c_str());
    if(!f){
        throw std::runtime_error(std::string("Cannot open file:") + path);
    }
    return f;
}

template<class T>
bool DenseFileOp::read_data(const std::string& path, ccma::algebra::BaseMatrixT<T>* mat){
    return read_data<T>(path, mat, -1);
}

template<class T>
bool DenseFileOp::read_data(const std::string& path,
                            const int col_size,
                            ccma::algebra::BaseMatrixT<T>* mat){
    uint32_t rows = get_file_rows(path);
    uint32_t cols = get_file_fields(path);
    if(rows <= 1 || cols <= 1 ||(col_size != -1 && cols != col_size)){
        return false;
    }

    char buff[this->BUFF_SIZE];
    bool read_flag = true;
    uint32_t row, col;
    T* raw_data = new T[rows * cols];
    T* row_data = new T[cols];

    FILE* f = this->open_file(path, "r");
    for(row = 0; fgets(buff, this->BUFF_SIZE, f) != nullptr;){
        if(!this->split<T>(&buff, "\t", cols, row_data)){
            read_flag = false;
            break;
        }
        memcpy(&row_data[row * cols], row_data, sizeof(T) * cols);
        row++;
    }
    fclose(f);

    delete[] row_data;

    if(!read_flag){
        delete[] raw_data;

        return false;
    }
    if(row < rows){
        T* data = new T[row * cols];
        memcpy(data, raw_data, sizeof(T) * row * cols);
        mat->set_data(data, row, cols);
        delete[] data;
    }else{
        mat->set_data(raw_data, rows, cols);
    }
    delete[] raw_data;

    return true;
}


/*
template<class T, class LT>
bool DenseFileOp::read_data(std::string& const path, ccma::algebra::LabeledMatrixT<T, LT>* mat){
 return read_data(path, mat, -1, -1);
}


template<class T, class LT>
bool DenseFileOp::read_data(std::string& const path, ccma::algebra::LabeledMatrixT<T, LT>* mat, int col_size, int label_idx){
    uint32_t rows = get_file_rows(path);
    uint32_t cols = get_file_fields(path);
    if(col_size < 0){
        col_size = cols - 1;
    }
    if(label_idx < 0){
        label_idx = cols - 1;
    }
    if(rows <= 0 || col_size <= 1 || col_size != (cols - 1) || label_idx >= cols){
        return false;
    }

    char buff[this->BUFF_SIZE];
    T* raw_data = new T[rows * (cols - 1)];
    F* row_data = new F(cols - 1];
    LT* label_data = new LT[cols - 1];
    bool read_flag;
    uint32_t row, col;
    LT label_value;

    FILE* f = this->this->open_file(path,"r");
    for(row = 0; fgets(buff, this->BUFF_SIZE, f) != nullptr;){
        if(!split(&buff, "\t", cols, labe_idx, row_data, &label_value )){
            read_flag = false;
            break;
        }
        memcpy(raw_data[row * col_size], row_data, sizeof(T) * col_size);
        label_data[row] = label_value;
        row++;
    }
    fclose(f);
    delete[] row_data;
    if(!read_flag){
        delete[] raw_data;
        delete[] label_data;
        return false;
    }

    if(row < rows){
        T* data = new T[row * col_size];
        LT* l_data = new LT[row];
        memcpy(data, raw_data, sizeof(T) * row * col_size);
        memcpy(l_data, label_data, sizeof(LT) * row);

        result->set_data(data, l_data, row, col_size);
        delete[] l_data;
    }else{
        result->set_data(raw_data, label_data, row, col_size);
    }

    delete[] raw_data;
    delete[] label_data;
    return true;
}
*/

uint32_t DenseFileOp::get_file_rows(const std::string& path){
    uint32_t rows = 0;
    FILE* f = this->open_file(path, "r");
    char buff[this->BUFF_SIZE];
    while(fgets(buff, this->BUFF_SIZE, f) != nullptr){
        rows++;
    }
    fclose(f);

    return rows;
}

uint32_t DenseFileOp::get_file_fields(const std::string& path){
    uint32_t fields = 0;
    FILE* f = this->open_file(path, "r");
    char* buff = new char[this->BUFF_SIZE];
    fgets(buff, this->BUFF_SIZE, f);
    fclose(f);

    char* p;
    while((p = strsep(&buff, "\t")) != nullptr){
        if(*p != '\t' && *p != ' '){
            fields++;
        }
    }
    delete[] buff;
    return fields;
}

template<class T>
bool DenseFileOp::split(char** data,
                        const char* delim,
                        const int size,
                        T* result){
    uint32_t read_size = 0;
    char* p;

    while((p = strsep(data, "\t")) != nullptr){
        if(*p != '\t' && *p != ' '){
            result[read_size++] = (T)p;
        }
    }
    return (read_size == size);
}


template<class T, class LT>
bool DenseFileOp::split(char** data,
                        const char* delim,
                        const int size,
                        uint32_t label_idx,
                        T* result,
                        LT* label_value){
    uint32_t read_size = 0;
    char* p;

    while((p = strsep(data, "\t")) != nullptr){
        if(*p != '\t' && *p != ' '){
            if(read_size == label_idx){
                label_value = (LT)p;
            }else{
                result[read_size++] = (T)p;
            }
        }
    }
    return (read_size == size);
}

}//namespace utils
}//namespace ccma
