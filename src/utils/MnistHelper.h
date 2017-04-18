/*********************************************
* Author: Jun Jiang - jiangjun4@sina.com
* Created: 2017-04-18 17:41
* Last modified: 2017-04-18 17:41
* Filename: MnistHelper.h
* Description: read mnist file
**********************************************/

#ifndef _CCMA_UTILS_MNISTHELPER_H_
#define _CCMA_UTILS_MNISTHELPER_H_

#include <fstream>
#include <string>
#include "algebra/BaseMatrix.h"

namespace ccma{
namespace util{
class MnistHelper{
public:
    void read_mnist_file(const std::string& path, uint32_t key);

private:
    inline uint32_t read_header(const std::unique_ptr<char[]>& buffer, size_t position);
};//class MnistHelper


void MnistHelper::read_mnist_file(const std::string& path, uint32_t key){
    std::ifstream file;
    file.open(path, std::ios::in | std::ios::binary | std::ios::ate);

    if(!file){
        std::cout << "Can't open file:" << path << std::endl;
        return {};
    }

    auto size = file.tellg();
    std::unique_ptr<char[]> buffer(new char[size]);

    //read the entire file to buffer
    file.seekg(0, std::ios::beg);
    file.read(buffer.get(), size);
    file.close();

    auto magic_num = read_header(buffer, 0);

    if(magic_num != key){
        std::cout << "Invalid magic number, probably not  a MNIST file" << std::endl;
        return {};
    }

    auto count = read_header(buffer, 1);
}

inline uint32_t MnistHelper::read_header(const std::unique_ptr<char[]>& buffer, size_t position){
    auto header = reinterpret_cast<uint32_t*>(buffer.get());

    auto value = *(header + position);
    return (value << 24) | ((value << 8) & 0x00FF0000) | ((value >> 8) & 0X0000FF00) | (value >> 24);
}

}//namespace utils
}//namespace ccma
#endif
