/*********************************************
* Author: Jun Jiang - jiangjun4@sina.com
* Created: 2017-01-12 17:37
* Last modified:	2017-04-07 16:12
* Filename:		TestDNN.cpp
* Description: DNN test
**********************************************/

#include "DNN.h"
#include <iostream>

int main(int argc, char** argv){
    ccma::algorithm::nn::DNN* dnn = new ccma::algorithm::nn::DNN();
    dnn->add_layer(4);
    dnn->add_layer(2);
    dnn->add_layer(1);
    dnn->init_networks();
    std::cout <<"finished\n" <<std::endl;
    delete dnn;
}
