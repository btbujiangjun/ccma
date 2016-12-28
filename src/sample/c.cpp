#include <stdio.h>

template<class T>
class a{
public:
    virtual void echo(){
        printf("it's a.\n");
    };
};

typedef a<int> ai;

class b :public ai{
public:
    void echo(){
        ai::echo();
        printf("it's b.\n");
    }
};

int main(void){
 ai* aa = new b();
 aa->echo();
 delete aa;
}
