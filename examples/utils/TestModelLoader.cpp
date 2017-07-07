/***********************************************
 * Author: Jun Jiang - jiangjun4@sina.com
 * Create: 2017-07-07 16:21
 * Last modified : 2017-07-07 16:21
 * Filename      : TestModelLoader.cpp
 * Description   : 
 **********************************************/
#include <string>
#include "utils/ModelLoader.h"

int main(int argc, char** argv){
    ccma::algebra::DenseMatrixT<int> m1;
    const uint mat_size = 1000;
    int* d1 = new int[mat_size];
    for(uint i = 0; i != mat_size; i++){
        d1[i] = i;
    }
    m1.set_shallow_data(d1, 10, 100);

    ccma::utils::ModelLoader<int> loader;
    const std::string path = "data/test.model";
    loader.write(&m1, path);
    loader.write(&m1, path, true);


    std::vector<ccma::algebra::BaseMatrixT<int>*> m2;
    loader.read(path, &m2);
    printf("mat_size[%ld]\n", m2.size());
    if(m2.size()){
        m2[0]->display("|");
        delete m2[0];
    }
}
