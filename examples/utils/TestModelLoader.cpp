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
    ccma::algebra::DenseMatrixT<real> m1;
    const uint mat_size = 1000;
    real* d1 = new real[mat_size];
    for(uint i = 0; i != mat_size; i++){
        d1[i] = i;
    }
    m1.set_shallow_data(d1, 10, 100);

    ccma::algebra::DenseRandomMatrixT<real> m2(100, 100, 0.0, 1);

    ccma::utils::ModelLoader loader;
    const std::string path = "data/test.model";
    loader.write<real>(&m1, path, false, "testmode");
    loader.write<real>(&m2, path, true, "testmode");


    std::vector<ccma::algebra::BaseMatrixT<real>*> ms;
    loader.read<real>(path, &ms, "testmode");
    for(auto&& m : ms){
        m->display();
        delete m;
    }
}
