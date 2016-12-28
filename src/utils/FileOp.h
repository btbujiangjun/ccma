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
#include "TypeDef.h"
#include "algebra/BaseMatrix.h"

namespace ccma{
namespace utils{

class BaseFileOp{
protected:
    static const uint BUFF_SIZE = 102400;

    inline FILE* open_file(const std::string& path,  const std::string& open_mode);

    template<class T>
    bool split(char** data,
               const char* delim,
               const int size,
               T* result);

    template<class T>
    bool split(char** data,
               const char* delim,
               const int size,
               uint label_idx,
               T* result,
               T* label_value);
};//class BaseFileOp

class DenseFileOp : public BaseFileOp{
public:
    template<class T>
    bool read_data(const std::string& path, ccma::algebra::DenseMatrixT<T>* mat);

    template<class T>
    bool read_data(const std::string& path,
                   const int col_size,
                   ccma::algebra::DenseMatrixT<T>* mat);

    template<class T>
    bool read_data(const std::string& path, ccma::algebra::LabeledDenseMatrixT<T>* mat);

    template<class T>
    bool read_data(const std::string& path,
                   int col_size,
                   int label_idx,
                   ccma::algebra::LabeledDenseMatrixT<T>* mat);

private:
    uint get_file_rows(const std::string& path);
    uint get_file_fields(const std::string& path);
};//class DenseFileOp


inline FILE* BaseFileOp::open_file(const std::string& path, const std::string& open_mode){
    FILE* file = fopen(path.c_str(), open_mode.c_str());
    if(!file){
        throw std::runtime_error(std::string("Cannot open file:") + path);
    }
    return file;
}

template<class T>
bool BaseFileOp::split(char** data,
                        const char* delim,
                        const int size,
                        T* result){
    uint read_size = 0;
    char* p;
    while((p = strsep(data, "\t")) != nullptr && *p != '\n' && *p != '\r'){
        if(*p != '\t' && *p != ' ' && *p != '\0'){
            result[read_size++] = ccma::utils::type_cast<T>(p);
        }
    }
    return (read_size == size);
}

template<class T>
bool BaseFileOp::split(char** data,
                        const char* delim,
                        const int size,
                        uint label_idx,
                        T* result,
                        T* label_value){
    uint read_size = 0;
    char* p;

    while((p = strsep(data, "\t")) != nullptr){
        if(*p != '\t' && *p != ' ' && *p != '\0' && *p != '\n' && *p != '\r'){
            if(read_size == label_idx){
                *label_value = ccma::utils::type_cast<T>(p);
            }else{
                result[read_size++] = ccma::utils::type_cast<T>(p);
            }
        }
    }
    return (read_size == size);
}

template<class T>
bool DenseFileOp::read_data(const std::string& path, ccma::algebra::DenseMatrixT<T>* mat){
    return read_data<T>(path, -1, mat);
}

template<class T>
bool DenseFileOp::read_data(const std::string& path,
                            int col_size,
                            ccma::algebra::DenseMatrixT<T>* mat){
    uint rows = get_file_rows(path);
    uint cols = get_file_fields(path);

    //range check
    if(rows <= 1 || cols <= 1 ||(col_size != -1 && cols != col_size)){
        return false;
    }

    T* data = new T[rows * cols];
    T* row_data = new T[cols];

    char* buff = new char[this->BUFF_SIZE];
    //split function will be change the pointer of buff, so stored.
    char* buff_start = buff;

    bool read_flag = true;
    uint row_cursor;

    FILE* file = this->open_file(path, "r");
    for(row_cursor = 0; fgets(buff, this->BUFF_SIZE, file) != nullptr;){
        if(!this->split<T>(&buff, "\t", cols, row_data)){
            read_flag = false;
            break;
        }
        buff = buff_start;//re-point the start position.
        memcpy(&data[row_cursor * cols], row_data, sizeof(T) * cols);
        memset((void*)buff, 0, sizeof(char)*this->BUFF_SIZE);
        row_cursor++;
    }
    fclose(file);

    delete[] row_data;
    delete[] buff_start;

    if(mat == nullptr){
        mat = new ccma::algebra::DenseMatrixT<T>();
    }

    if(!read_flag){
        delete[] data;
        return false;
    }
    if(row_cursor < rows){
        T* new_data = new T[row_cursor * cols];
        memcpy(new_data, data, sizeof(T) * row_cursor * cols);
        mat->set_shallow_data(new_data, row_cursor, cols);
        delete[] data;
    }else{
        mat->set_shallow_data(data, rows, cols);
    }

    return true;
}


template<class T>
bool DenseFileOp::read_data(const std::string& path, ccma::algebra::LabeledDenseMatrixT<T>* mat){
    return read_data<T>(path, -1, -1, mat);
}


template<class T>
bool DenseFileOp::read_data(const std::string& path,
                            int col_size,
                            int label_idx,
                            ccma::algebra::LabeledDenseMatrixT<T>* mat){
    uint rows = get_file_rows(path);
    uint cols = get_file_fields(path);

    //range check
    if(col_size < 0){
        col_size = cols - 1;
    }
    if(label_idx < 0){
        label_idx = cols - 1;
    }
    if(rows <= 0 || col_size <= 1 || col_size != (cols - 1) || label_idx >= cols){
        return false;
    }

    T* data = new T[rows * (cols - 1)];
    T* label_data = new T[rows];

    char* buff = new char[this->BUFF_SIZE];
    T* row_data = new T[cols - 1];

    T label_value;
    char* buff_start = buff;

    bool read_flag = true;
    uint row_cursor;

    FILE* file = this->open_file(path,"r");
    for(row_cursor = 0; fgets(buff, this->BUFF_SIZE, file) != nullptr;){
        if(!this->split<T>(&buff, "\t", col_size, label_idx, row_data, &label_value )){
            read_flag = false;
            break;
        }

        memcpy(&data[row_cursor * col_size], row_data, sizeof(T) * col_size);
        label_data[row_cursor] = label_value;

        buff = buff_start;//pointer be changed by strsep, re-point to the start addr.
        memset((void*)buff, 0, sizeof(char) * this->BUFF_SIZE);
        row_cursor++;
    }
    fclose(file);

    delete[] buff_start;
    delete[] row_data;

    if(!read_flag){
        delete[] data;
        delete[] label_data;
        return false;
    }

    if(mat == nullptr){
        mat = new ccma::algebra::LabeledDenseMatrixT<T>();
    }

    if(row_cursor < rows){
        T* new_data = new T[row_cursor * col_size];
        T* new_label_data = new T[row_cursor];
        memcpy(new_data, data, sizeof(T) * row_cursor * col_size);
        memcpy(new_label_data, label_data, sizeof(T) * row_cursor);

        mat->set_shallow_data(new_data, new_label_data, row_cursor, col_size);
        delete[] data, label_data;
    }else{
        mat->set_shallow_data(data, label_data, row_cursor, col_size);
    }

    return true;
}

uint DenseFileOp::get_file_rows(const std::string& path){
    uint rows = 0;
    char* buff = new char[this->BUFF_SIZE];

    FILE* f = this->open_file(path, "r");
    while(fgets(buff, this->BUFF_SIZE, f) != nullptr){
        rows++;
    }
    fclose(f);

    delete[] buff;

    return rows;
}

uint DenseFileOp::get_file_fields(const std::string& path){
    uint fields = 0;
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
