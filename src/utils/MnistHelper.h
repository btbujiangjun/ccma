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


}

}//namespace utils
}//namespace ccma
#endif
