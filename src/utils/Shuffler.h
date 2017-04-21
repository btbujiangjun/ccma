/*********************************************
* Author: Jun Jiang - jiangjun4@sina.com
* Created: 2017-04-10 16:33
* Last modified: 2017-04-10 16:33
* Filename: MatrixShuffle.h
* Description: matrix shuffle
**********************************************/

#ifndef _CCMA_UTILS_SHUFFLER_H_
#define _CCMA_UTILS_SHUFFLER_H_

#include <random>
#include <vector>
#include <ctime>

namespace ccma{
namespace utils{

class Shuffler{
public:
    Shuffler(uint size);

    ~Shuffler();

    void shuffle();

    uint get_row(uint row_id);

private:
    uint _size;
    std::vector<uint> _shuffler_idx;
};//class Shuffler


Shuffler::Shuffler(uint size){
    _size = size;
    shuffle();
}

Shuffler::~Shuffler(){
    _shuffler_idx.clear();
}

void Shuffler::shuffle(){

    _shuffler_idx.clear();

    std::vector<int> idx;
    for(int i = 0; i < _size; i++){
        idx.push_back(i);
    }

    while(idx.size() > 0){
        std::default_random_engine generator(time(0));
        std::uniform_int_distribution<int> dis(0, idx.size() - 1);

        int idx_value = idx[dis(generator)];
        _shuffler_idx.push_back(idx[idx_value]);

        std::vector<int>::iterator it = idx.begin();
        while(it != idx.end()){
            if(*it == idx_value){
                idx.erase(it);
                break;
            }
            it++;
        }
    }

}

uint Shuffler::get_row(uint row_id){
    return _shuffler_idx[row_id];
}

}//namespace algebra
}//namespace ccma

#endif
