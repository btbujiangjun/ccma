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

    ~Shuffler(){ _shuffler_idx.clear();}

    void shuffle();

    uint get_row(uint row_id);

private:
    uint _size;
    std::vector<uint> _shuffler_idx;
};//class Shuffler


Shuffler::Shuffler(uint size){
    _size = size;
    for(uint i = 0; i != _size; i++){
        _shuffler_idx.push_back(i);
    }
}

void Shuffler::shuffle(){

    std::random_device rd;
    uint random_idx, value;

    for(uint i = _size - 1; i != 0 ; i--){
        random_idx = rd() % i;
        value =  _shuffler_idx[random_idx];

        _shuffler_idx[random_idx] = _shuffler_idx[i];
        _shuffler_idx[i] = value;
    }

}

uint Shuffler::get_row(uint row_id){
    return _shuffler_idx[row_id];
}

}//namespace algebra
}//namespace ccma

#endif
