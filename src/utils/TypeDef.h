/*********************************************
* Author: Jun Jiang - jiangjun4@sina.com
* Created: 2016-11-22 11:23
* Last modified: 2016-11-22 11:23
* Filename: TypeDef.h
* Description:
**********************************************/

#ifndef _CCMA_UTILS_TYPEDEF_H_
#define _CCMA_UTILS_TYPEDEF_H_

#include <cstddef>

namespace ccma{
namespace utils{
#ifdef CCMA_TYPE_DOUBLE
    typedef double real;
#else
    typedef float real;
#endif

typedef size_t uint;

template<class T>
T ccma_cast(const char* data){
    if(typeid(T) == typeid(int)){
        return atoi(data);
    }else if(typeid(T) == typeid(real)){
        return atof(data);
    }else{
        return (T)*data;
    }
}

template<class T1, class T2>
bool ccma_type_compare(){
    return sizeof(T1) > sizeof(T2);
}


}//namespace utils
}//namespace ccma

using ccma::utils::real;
using ccma::utils::uint;

#endif
