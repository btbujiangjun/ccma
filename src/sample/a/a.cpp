#include "a.h"
template<class T>
T A::sum(T a, T b){
    return a + b;
}

template<class T>
A* A::sumsum(T a, T b){
    return new A();
}

template int A::sum(int, int);
template A* A::sumsum(int, int);

//template class A<int>;
//template class A<float>;
