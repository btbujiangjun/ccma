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
    ccma::algebra::DenseRandomMatrixT<real> m1(10, 10, 0.0, 1);

    ccma::utils::ModelLoader loader;
    const std::string path = "data/test.model";
    loader.write<real>(&m1, path);
    loader.write<real>(&m1, path, true);


    std::vector<ccma::algebra::BaseMatrixT<real>*> m2;
    loader.read<real>(path, &m2);
    printf("mat_size[%ld]\n", m2.size());
    if(m2.size()){
        m2[0]->display("|");
        delete m2[0];
    }
}
