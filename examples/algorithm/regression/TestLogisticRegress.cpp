/*********************************************
* Author: Jun Jiang - jiangjun4@sina.com
* Created: 2016-12-02 16:51
* Last modified: 2016-12-02 16:51
* Filename: TestRegress.cpp
* Description: 
**********************************************/

#include "stdio.h"
#include <fstream>
#include <iostream>
#include <string.h>
#include "algorithm/regression/LogisticRegress.h"

int main(int argc, char** argv){
    real* a = new real[300];
    real* b = new real[100];

    std::ifstream in_file;
    in_file.open("data/testSet.txt");

    uint k = 0;
    if(in_file.is_open()){
        char buff[1024];
        real x, y, z;
        while(in_file.good() && !in_file.eof() && k < 100){
            memset(buff, 0, 1024);
            in_file.getline(buff, 1024);
            if(sscanf(buff,"%f %f %f", &x, &y, &z) == 3){
                a[k*3] = 1.0f;
                a[k*3 + 1] = x;
                a[k*3 + 2] = y;
                b[k] = (int)z;
                k++;
            }
        }
        in_file.close();
    }
    ccma::algebra::LabeledDenseMatrixT<real> lm(a, b, 100, 3);
    ccma::algorithm::regression::LogisticRegress ls;

    lm.display();

    ls.batch_grad_desc(&lm, 150);
    ls.stoc_grad_desc(&lm, 150);
    ls.smooth_stoc_grad_desc(&lm, 150);
}
