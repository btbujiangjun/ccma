#include "b.h"
#include "stdlib.h"
#include "stdio.h"

int B::double_sum(int a, int b){
    A* aObj = new A();
//    int c = aObj->sum<int>(a, b);
//    float d = aObj->sum<float>(static_cast<float>(a), static_cast<float>(b));
    int c = 0;

    A* aa = aObj->sumsum(a, b);
    c = aa->sum(a, b);
    delete aObj, aa;
    return 2 * c;
}

int main(int argc, char** argv){
    B* b = new B();
    int sum = b->double_sum(2, 5);
    printf("[%d]\n", sum);
    delete b;
    return 0;
}
