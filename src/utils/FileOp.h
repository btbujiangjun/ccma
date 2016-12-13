/*********************************************
* Author: Jun Jiang - jiangjun4@sina.com
* Created: 2016-12-12 10:46
* Last modified: 2016-12-12 10:46
* Filename: FileOp.h
* Description: file operation
**********************************************/
#ifndef _CCMA_UTILS_FILEOP_H_
#define _CCMA_UTILS_FILEOP_H_

#include <stdint.h>
#include <cstdlib>
#include <string.h>
#include <memory.h>
#include <stdio.h>
#include <stdexcept>
#include <iostream>
#include "algebra/BaseMatrix.h"

namespace ccma{
namespace utils{

class BaseFileOp{
public:
    BaseFileOp(){
        BUFF_SIZE = 102400;
    }

protected:
    uint32_t BUFF_SIZE;

    inline FILE* open_file(const std::string& path,  const std::string& open_mode);

    template<class T>
    bool split(char** data,
               const char* delim,
               const int size,
               T* result);

    template<class T, class LT>
    bool split(char** data,
               const char* delim,
               const int size,
               uint32_t label_idx,
               T* result,
               LT* label_value);
};//class BaseFileOp

class DenseFileOp : public BaseFileOp{
public:
    template<class T>
    bool read_data(const std::string& path, ccma::algebra::BaseMatrixT<T>* mat);

    template<class T>
    bool read_data(const std::string& path,
                   const int col_size,
                   ccma::algebra::BaseMatrixT<T>* mat);

    template<class T, class LT, class FT>
    bool read_data(const std::string& path, ccma::algebra::LabeledMatrixT<T, LT, FT>* mat);

    template<class T, class LT, class FT>
    bool read_data(const std::string& path,
                   int col_size,
                   int label_idx,
                   ccma::algebra::LabeledMatrixT<T, LT, FT>* mat);

private:
    uint32_t get_file_rows(const std::string& path);
    uint32_t get_file_fields(const std::string& path);
};//class DenseFileOp


inline FILE* BaseFileOp::open_file(const std::string& path, const std::string& open_mode){
    FILE* f = fopen(path.c_str(), open_mode.c_str());
    if(!f){
        throw std::runtime_error(std::string("Cannot open file:") + path);
    }
    return f;
}

template<class T>
bool BaseFileOp::split(char** data,
                        const char* delim,
                        const int size,
                        T* result){
    uint32_t read_size = 0;
    char* p;
    while((p = strsep(data, "\t")) != nullptr && *p != '\n' && *p != '\r'){
        if(*p != '\t' && *p != ' ' && *p != '\0'){
            result[read_size++] = ccma::utils::ccma_cast<T>(p);
        }
    }
    return (read_size == size);
}

template<class T, class LT>
bool BaseFileOp::split(char** data,
                        const char* delim,
                        const int size,
                        uint32_t label_idx,
                        T* result,
                        LT* label_value){
    uint32_t read_size = 0;
    char* p;

    while((p = strsep(data, "\t")) != nullptr){
        if(*p != '\t' && *p != ' ' && *p != '\0' && *p != '\n' && *p != '\r'){
            if(read_size == label_idx){
                *label_value = ccma::utils::ccma_cast<LT>(p);
            }else{
                result[read_size++] = ccma::utils::ccma_cast<T>(p);
            }
        }
    }
    return (read_size == size);
}

template<class T>
bool DenseFileOp::read_data(const std::string& path, ccma::algebra::BaseMatrixT<T>* mat){
    return read_data<T>(path, -1, mat);
}

template<class T>
bool DenseFileOp::read_data(const std::string& path,
                            int col_size,
                            ccma::algebra::BaseMatrixT<T>* mat){
    uint32_t rows = get_file_rows(path);
    uint32_t cols = get_file_fields(path);
    if(rows <= 1 || cols <= 1 ||(col_size != -1 && cols != col_size)){
        return false;
    }

    char* buff = new char[this->BUFF_SIZE];
    bool read_flag = true;
    uint32_t row, col;
    T* raw_data = new T[rows * cols];
    T* row_data = new T[cols];
    char* buff_start = buff;

    FILE* f = this->open_file(path, "r");
    for(row = 0; fgets(buff, this->BUFF_SIZE, f) != nullptr;){
        if(!this->split<T>(&buff, "\t", cols, row_data)){
            read_flag = false;
            break;
        }
        buff = buff_start;
        memcpy(&raw_data[row * cols], row_data, sizeof(T) * cols);
        memset((void*)buff, 0, sizeof(char)*this->BUFF_SIZE);
        row++;
    }
    fclose(f);

    delete[] row_data;
    delete[] buff_start;

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


template<class T, class LT, class FT>
bool DenseFileOp::read_data(const std::string& path, ccma::algebra::LabeledMatrixT<T, LT, FT>* mat){
    return read_data<T, LT, FT>(path, -1, -1, mat);
}


template<class T, class LT, class FT>
bool DenseFileOp::read_data(const std::string& path,
                            int col_size,
                            int label_idx,
                            ccma::algebra::LabeledMatrixT<T, LT, FT>* mat){
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

    char* buff = new char[this->BUFF_SIZE];
    T* raw_data = new T[rows * (cols - 1)];
    T* row_data = new T[cols - 1];
    LT* label_data = new LT[rows];
    FT* fea_data = new FT[cols - 1];

    bool read_flag = true;
    uint32_t row, col;
    LT label_value;
    char* buff_start = buff;

    FILE* f = this->open_file(path,"r");
    for(row = 0; fgets(buff, this->BUFF_SIZE, f) != nullptr;){
        if(!this->split<T, LT>(&buff, "\t", col_size, label_idx, row_data, &label_value )){
            read_flag = false;
            break;
        }

        memcpy(&raw_data[row * col_size], row_data, sizeof(T) * col_size);
        label_data[row] = label_value;

        buff = buff_start;//pointer be changed by strsep
        memset((void*)buff, 0, sizeof(char) * this->BUFF_SIZE);
        row++;
    }
    fclose(f);

    delete[] buff_start;
    delete[] row_data;

    if(!read_flag){
        delete[] raw_data;
        delete[] label_data;
        delete[] fea_data;
        return false;
    }

    for(int i = 0; i < col_size; i++){
        fea_data[i++] = (FT)('A'+1);
    }

    if(row < rows){
        T* data = new T[row * col_size];
        LT* l_data = new LT[row];
        memcpy(data, raw_data, sizeof(T) * row * col_size);
        memcpy(l_data, label_data, sizeof(LT) * row);

        mat->set_data(data, l_data, fea_data, row, col_size);
        delete[] l_data;
    }else{
        mat->set_data(raw_data, label_data, fea_data, row, col_size);
    }

    delete[] raw_data;
    delete[] label_data;
    delete[] fea_data;

    return true;
}

uint32_t DenseFileOp::get_file_rows(const std::string& path){
    uint32_t rows = 0;
    char* buff = new char[this->BUFF_SIZE];

    FILE* f = this->open_file(path, "r");
    while(fgets(buff, this->BUFF_SIZE, f) != nullptr){
        rows++;
    }
    fclose(f);

    delete[] buff;

    return rows;
}

uint32_t DenseFileOp::get_file_fields(const std::string& path){
    uint32_t fields = 0;
    char* buff = new char[this->BUFF_SIZE];
    char* buff_start = buff;

    FILE* f = this->open_file(path, "r");
    fgets(buff, this->BUFF_SIZE, f);
    fclose(f);

    char* p;
    while((p = strsep(&buff, "\t")) != nullptr && *p != '\n' && *p != '\r'){
        if(*p != '\t' && *p != ' ' && *p != '\0'){
            fields++;
        }
    }
    delete[] buff_start;

    return fields;
}

}//namespace utils
}//namespace ccma


#endif //_CCMA_UTILS_FILEOP_H_
