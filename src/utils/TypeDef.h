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
#include <typeinfo>
#include <stdlib.h>
#include <limits.h>
#include <float.h>

namespace ccma{
namespace utils{
#ifdef CCMA_TYPE_DOUBLE
    typedef double real;
    #define MAX_REAL DBL_MAX;
    #define MIN_REAL DBL_MIN;
#else
    typedef float real;
    #define MAX_REAL FLT_MAX;
    #define MIN_REAL FLT_MIN;
#endif

typedef size_t uint;
#define MAX_INT INT_MAX;
#define MIN_INT INT_MIN;

template<class T>
T type_cast(const char* data){
    if(typeid(T) == typeid(int)){
        return atoi(data);
    }else if(typeid(T) == typeid(real)){
        return atof(data);
    }else{
        return static_cast<T>(*data);
    }
}

template<class T1, class T2>
bool ccma_type_compare(){
    return sizeof(T1) > sizeof(T2);
}

template<class T>
T get_max_value(){
    if(typeid(T) == typeid(int)){
        return MAX_INT;
    }else{
        return MAX_REAL;
    }
}
template<class T>
T get_min_value(){
    if(typeid(T) == typeid(int)){
        return MIN_INT;
    }else{
        return MIN_REAL;
    }
}

}//namespace utils
}//namespace ccma

using ccma::utils::real;
using ccma::utils::uint;

#endif //_CCMA_UTILS_TYPEDEF_H_
